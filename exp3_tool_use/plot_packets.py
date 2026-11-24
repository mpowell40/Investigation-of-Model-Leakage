"""Plot the packets of a capture in detail, or compare search / static / off / base.

    python plot_packets.py captures/tool_search_s1_pT01_r01.pcapng    one figure per file
    python plot_packets.py captures/tool_*_pT03_*.pcapng                several files
    python plot_packets.py processed/tool_static_s1_pT05_r02.csv        a processed capture
    python plot_packets.py FILE --zoom 5 15                             only 5-15 s
    python plot_packets.py FILE --around                                speech start - 2 s to reply end + 2 s
    python plot_packets.py --summary                                    feature distributions by class
    add --show to open a window as well as saving the PNG

Per-file figure (the voice session's port-443 flow):
  1. every packet: time vs size, upload and download, ACK-sized packets faded,
     your speech and Claude's reply shaded, the latency gap marked
  2. throughput (KB/s) per direction
  3. packet rate (pkt/s) per direction
  4. packet-size histograms of audio-sized packets (> 80 B), upload and download
The summary figure needs processed/ (run preprocess.py). Figures go to plots/.
"""
import argparse
from pathlib import Path

import numpy as np

import common
import features
import prompts
from preprocess import ROOT

PLOTS_DIR = ROOT / "plots"

# Same palette and styling as exp2_language_id/plot_packets.py (dataviz default palette).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"
SPEECH_FILL, REPLY_FILL = "#f0efec", "#e4e3df"
CLASS_COLOR = {"search": ORANGE, "static": BLUE, "off": AQUA, "base": INK_2}
CLASS_NAME = {"search": "search (current info)", "static": "static", "off": "search OFF",
              "base": "baseline"}
DIRS = ((True, BLUE, "Upload (your mic)"), (False, ORANGE, "Download (Claude)"))


def setup(show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 1.6, "font.size": 9,
    })
    return plt


def shade(ax, edges, label=False):
    if not edges:
        return
    ax.axvspan(edges["speech_start"], edges["speech_end"], color=SPEECH_FILL, lw=0, zorder=0)
    ax.axvspan(edges["reply_start"], edges["reply_end"], color=REPLY_FILL, lw=0, zorder=0)
    if label:
        tr = ax.get_xaxis_transform()
        ax.text((edges["speech_start"] + edges["speech_end"]) / 2, 1.0, "your speech",
                transform=tr, ha="center", va="bottom", fontsize=7, color=INK_2)
        ax.text((edges["reply_start"] + edges["reply_end"]) / 2, 1.0, "Claude's reply",
                transform=tr, ha="center", va="bottom", fontsize=7, color=INK_2)


def binned(t, values, lo, hi, width):
    edges = np.arange(lo, hi + width, width)
    sums = np.histogram(t, bins=edges, weights=values)[0]
    return edges[:-1] + width / 2, sums / width


def title_for(path):
    name = Path(path).stem
    meta = common.parse_name(name + ".pcapng")
    if not meta:
        return name, ""
    words = prompts.prompt_text(meta["class"], meta["prompt_id"])
    return (f"{name}   ({CLASS_NAME[meta['class']]}, speaker S{meta['speaker']}, "
            f"{meta['prompt_id']}, rep {meta['rep']})", f"“{words}”")


def plot_file(plt, path, zoom, around, out_dir):
    packets, edges, status, f = features.from_capture(path)
    if not packets:
        print(f"{path}: no port-443 traffic, skipped")
        return None
    t, size, up = features.as_arrays(packets)
    lo, hi = 0.0, float(t[-1])
    if around and edges:
        lo, hi = max(0.0, edges["speech_start"] - 2), edges["reply_end"] + 2
    elif zoom:
        lo, hi = zoom
    w = (t >= lo) & (t <= hi)
    width = 0.1 if hi - lo <= 40 else 0.25
    ack = size <= features.AUDIO_MIN

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.3, 1, 1, 1])
    ax_pk = fig.add_subplot(gs[0, :])
    ax_bw = fig.add_subplot(gs[1, :], sharex=ax_pk)
    ax_pr = fig.add_subplot(gs[2, :], sharex=ax_pk)
    ax_hu, ax_hd = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    title, words = title_for(path)
    fig.suptitle(title + (f"\n{words}" if words else ""), x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)

    # 1. Every packet.
    for is_up, color, name in DIRS:
        sel = w & (up == is_up)
        ax_pk.scatter(t[sel & ~ack], size[sel & ~ack], s=4, color=color, alpha=0.55, lw=0,
                      label=name, rasterized=True)
        ax_pk.scatter(t[sel & ack], size[sel & ack], s=3, color=MUTED, alpha=0.25, lw=0,
                      rasterized=True)
    ax_pk.scatter([], [], s=6, color=MUTED, label=f"ACK-sized (<= {features.AUDIO_MIN} B)")
    shade(ax_pk, edges, label=True)
    top = np.percentile(size[w], 99.5) if w.any() else 500
    ymax = max(top * 1.15, 500)
    ax_pk.set_ylim(0, ymax)
    if edges:
        y = ymax * 0.9
        ax_pk.annotate("", xy=(edges["reply_start"], y), xytext=(edges["speech_end"], y),
                       arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
        ax_pk.text(edges["reply_start"] + 0.1, y, f"latency {f['lat_s']:.2f} s", va="center",
                   fontsize=8, color=INK)
        ax_pk.axvline(edges["speech_end"] + f["lat_main_s"], color=INK, lw=0.8, ls="--")
        ax_pk.text(edges["speech_end"] + f["lat_main_s"] + 0.1, ymax * 0.78,
                   f"main answer +{f['lat_main_s']:.2f} s", fontsize=7, color=INK_2)
        sub = (f"reply {f['r_dur']:.1f} s, longest pause {f['r_pause_max']:.1f} s, "
               f"{f['r_early_kb']:.0f} KB in first {features.EARLY_S:g} s")
    else:
        sub = f"detection failed: {status}"
    ax_pk.set_title(f"Every packet (size on the wire). {sub}", pad=14)
    ax_pk.set_ylabel("bytes")
    ax_pk.legend(loc="lower right", ncol=3, markerscale=2.5)

    # 2. Throughput and 3. packet rate.
    for ax, weights, ylabel, ttl in (
            (ax_bw, size / 1000, "KB/s",
             f"Throughput ({width:g} s bins). Light gray = your speech, darker gray = Claude's reply"),
            (ax_pr, np.ones(len(size)), "packets/s", "Packet rate")):
        for is_up, color, name in DIRS:
            sel = w & (up == is_up)
            x, yv = binned(t[sel], weights[sel], lo, hi, width)
            ax.plot(x, yv, color=color, label=name.split(" ")[0])
        shade(ax, edges)
        ax.set_title(ttl)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", ncol=2)
    ax_pr.set_xlabel("seconds since the voice session's first packet")
    ax_pr.set_xlim(lo, hi)

    # 4. Size histograms (audio-sized packets only).
    for ax, (is_up, color, name) in zip((ax_hu, ax_hd), DIRS):
        s = size[w & (up == is_up) & ~ack]
        if len(s):
            lo_s, hi_s = np.percentile(s, [0.5, 99.5])
            ax.hist(s, bins=np.arange(int(lo_s), int(hi_s) + 2), color=color, lw=0)
            ax.axvline(np.median(s), color=INK, lw=1)
            ax.text(np.median(s), 1.0, f" median {np.median(s):.0f} B",
                    transform=ax.get_xaxis_transform(), va="top", fontsize=7, color=INK)
            ax.set_yscale("log")
        ax.set_title(f"{name.split(' ')[0]} packet sizes (> {features.AUDIO_MIN} B, n = {len(s):,})")
        ax.set_xlabel("bytes")
        ax.set_ylabel("packets (log)")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_around" if around else (f"_{lo:g}-{hi:g}s" if zoom else "")
    out = out_dir / f"{Path(path).stem}{suffix}.png"
    fig.savefig(out, dpi=130)
    print(f"{out}  ({status})")
    return fig


SUMMARY = [
    ("lat_s", "Latency: speech end -> first reply packet", "seconds"),
    ("lat_norm_s", "Latency minus sitting's base latency", "seconds"),
    ("lat_main_s", "Speech end -> main answer burst", "seconds"),
    ("r_dur", "Reply length", "seconds"),
    ("r_pause_max", "Longest download pause inside reply", "seconds"),
    ("r_early_kb", f"Reply audio in first {features.EARLY_S:g} s", "KB"),
    ("r_dn_size_std", "Reply download packet size, std", "bytes"),
    ("r_dn_size_h8", "Reply packets > 427 B (fraction)", "fraction"),
    ("r_dn_iat_cv", "Reply packet timing variability", "CV of gaps"),
    ("r_dn_bps", "Reply download rate", "bytes/s"),
    ("gap_dn_pkts", "Download packets in the latency gap", "packets"),
    ("idle_pre_up_frame_mean", "Idle upload frame size (red-flag check)", "bytes"),
]


def plot_summary(plt, out_dir):
    rows, _ = features.load_dataset()
    classes = [c for c in ("search", "static", "off", "base") if any(r["class"] == c for r in rows)]
    missing = [c for c in ("search", "static", "off", "base") if c not in classes]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(17, 11.5), constrained_layout=True)
    counts = ", ".join(f"{sum(r['class'] == c for r in rows)} {c}" for c in classes)
    note = f" (no {', '.join(missing)} captures yet)" if missing else ""
    fig.suptitle(f"Experiment 3: per-capture packet features by class ({counts}){note}\n"
                 "Each dot is one capture; filled = sittings 1-2, hollow = sitting 3 (held out); "
                 "black bars are medians.",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    for ax, (key, ttl, unit) in zip(axes.flat, SUMMARY):
        for i, c in enumerate(classes):
            for test in (False, True):
                v = np.array([r["features"][key] for r in rows
                              if r["class"] == c and (r["sitting"] == "3") == test])
                x = i + rng.uniform(-0.17, 0.17, len(v))
                kw = (dict(facecolors="none", edgecolors=CLASS_COLOR[c], linewidths=1.0) if test
                      else dict(color=CLASS_COLOR[c], lw=0))
                ax.scatter(x, v, s=20, alpha=0.8, label=None, **kw)
            v = [r["features"][key] for r in rows if r["class"] == c]
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=1.5)
        ax.set_xticks(range(len(classes)), [c for c in classes])
        ax.set_xlim(-0.6, len(classes) - 0.4)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary_by_class.png"
    fig.savefig(out, dpi=130)
    print(out)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".pcapng / .pcap captures or processed .csv files")
    ap.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"), help="seconds to show")
    ap.add_argument("--around", action="store_true",
                    help="zoom from 2 s before your speech to 2 s after the reply")
    ap.add_argument("--summary", action="store_true", help="compare features across classes")
    ap.add_argument("--out", default=str(PLOTS_DIR), help="folder for the PNGs")
    ap.add_argument("--show", action="store_true", help="also open the figures in a window")
    args = ap.parse_args()
    if not args.files and not args.summary:
        ap.error("give one or more files, or --summary")

    plt = setup(args.show)
    out_dir = Path(args.out)
    for path in args.files:
        fig = plot_file(plt, path, args.zoom, args.around, out_dir)
        if fig is not None and not args.show:
            plt.close(fig)
    if args.summary:
        plot_summary(plt, out_dir)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
