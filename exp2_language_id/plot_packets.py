"""Plot the packets of a capture in detail, or compare the three languages.

    python plot_packets.py sessions/lang_hi_s1_pALL_r01.pcapng     one figure per file
    python plot_packets.py sessions/*.pcapng                        every session
    python plot_packets.py processed/questions/es/lang_es_s1_pL03_r02.csv   one question
    python plot_packets.py FILE --zoom 30 45                        only 30-45 s
    python plot_packets.py FILE --turn 3                            zoom on turn 3
    python plot_packets.py --summary                                English vs Spanish vs Hindi
    add --show to open a window as well as saving the PNG

Per-file figure (the voice session's port-443 flow):
  1. every packet: time vs size, upload and download, ACK-sized packets faded
  2. throughput (KB/s) per direction, detected turns shaded and numbered
  3. packet rate (pkt/s) per direction
  4. packet-size histograms of audio-carrying packets, upload and download
The summary figure needs processed/ (run preprocess.py). Figures go to plots/.
"""
import argparse
from pathlib import Path

import numpy as np

import features
from preprocess import ROOT
from questions import LANG_NAME, LANGS

PLOTS_DIR = ROOT / "plots"

# Categorical slots 1-3 of the validated default palette (dataviz skill).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"
QUESTION_FILL, REPLY_FILL = "#f0efec", "#e4e3df"
LANG_COLOR = {"en": BLUE, "es": ORANGE, "hi": AQUA}
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


def shade_turns(ax, turns, lo, hi, label=False):
    for tr in turns:
        if tr["r_end"] < lo or tr["q_start"] > hi:
            continue
        ax.axvspan(tr["q_start"], tr["r_start"], color=QUESTION_FILL, lw=0, zorder=0)
        ax.axvspan(tr["r_start"], tr["r_end"], color=REPLY_FILL, lw=0, zorder=0)
        if label:
            ax.text((tr["r_start"] + tr["r_end"]) / 2, 1.0, f"T{tr['turn']}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7, color=INK_2)


def binned(t, values, lo, hi, width):
    edges = np.arange(lo, hi + width, width)
    sums = np.histogram(t, bins=edges, weights=values)[0]
    return edges[:-1] + width / 2, sums / width


def title_for(path):
    name = Path(path).name
    parts = name.split("_")
    if len(parts) >= 5 and parts[0] == "lang" and parts[1] in LANG_NAME:
        q = "all questions" if parts[3] == "pALL" else parts[3][1:]
        return f"{name}   ({LANG_NAME[parts[1]]}, speaker {parts[2].upper()}, {q}, rep {int(parts[4][1:3])})"
    return name


def plot_file(plt, path, zoom, turn_no, out_dir):
    packets, turns = features.turns_from_capture(path)
    if not packets:
        print(f"{path}: no port-443 traffic, skipped")
        return None
    t, size, up = features.as_arrays(packets)
    lo, hi = 0.0, float(t[-1])
    if turn_no:
        if not 1 <= turn_no <= len(turns):
            raise SystemExit(f"{path}: has {len(turns)} turns, no turn {turn_no}")
        tr = turns[turn_no - 1]
        lo, hi = max(0.0, tr["q_start"] - 2), tr["r_end"] + 2
    elif zoom:
        lo, hi = zoom
    w = (t >= lo) & (t <= hi)
    width = 0.1 if hi - lo <= 60 else 0.5          # finer bins when zoomed in
    ack = size <= features.ACK_MAX

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.3, 1, 1, 1])
    ax_pk = fig.add_subplot(gs[0, :])
    ax_bw = fig.add_subplot(gs[1, :], sharex=ax_pk)
    ax_pr = fig.add_subplot(gs[2, :], sharex=ax_pk)
    ax_hu, ax_hd = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    fig.suptitle(title_for(path), x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)

    # 1. Every packet.
    for is_up, color, name in DIRS:
        sel = w & (up == is_up)
        ax_pk.scatter(t[sel & ~ack], size[sel & ~ack], s=4, color=color, alpha=0.55, lw=0,
                      label=name, rasterized=True)
        ax_pk.scatter(t[sel & ack], size[sel & ack], s=3, color=MUTED, alpha=0.25, lw=0,
                      rasterized=True)
    ax_pk.scatter([], [], s=6, color=MUTED, label=f"ACK-sized (<= {features.ACK_MAX} B)")
    shade_turns(ax_pk, turns, lo, hi, label=True)
    ax_pk.set_title("Every packet (size on the wire)", pad=14)
    ax_pk.set_ylabel("bytes")
    top = np.percentile(size[w], 99.5) if w.any() else 300
    ax_pk.set_ylim(0, max(top * 1.15, 250))
    ax_pk.legend(loc="lower right", ncol=3, markerscale=2.5)

    # 2. Throughput and 3. packet rate.
    for ax, weights, ylabel, title in (
            (ax_bw, size / 1000, "KB/s",
             f"Throughput ({width:g} s bins). Light gray = question window, darker gray = Claude's reply"),
            (ax_pr, np.ones(len(size)), "packets/s", "Packet rate")):
        for is_up, color, name in DIRS:
            sel = w & (up == is_up)
            x, y = binned(t[sel], weights[sel], lo, hi, width)
            ax.plot(x, y, color=color, label=name.split(" ")[0])
        shade_turns(ax, turns, lo, hi)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", ncol=2)
    ax_pr.set_xlabel("seconds since start")
    ax_pr.set_xlim(lo, hi)

    # 4. Size histograms (audio-carrying packets only).
    for ax, (is_up, color, name) in zip((ax_hu, ax_hd), DIRS):
        s = size[w & (up == is_up) & ~ack]
        if len(s):
            lo_s, hi_s = np.percentile(s, [0.5, 99.5])
            ax.hist(s, bins=np.arange(int(lo_s), int(hi_s) + 2), color=color, lw=0)
            ax.axvline(np.median(s), color=INK, lw=1)
            ax.text(np.median(s), 1.0, f" median {np.median(s):.0f} B",
                    transform=ax.get_xaxis_transform(), va="top", fontsize=7, color=INK)
        ax.set_title(f"{name.split(' ')[0]} packet sizes (> {features.ACK_MAX} B, n = {len(s):,})")
        ax.set_xlabel("bytes")
        ax.set_ylabel("packets")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_turn{turn_no}" if turn_no else (f"_{lo:g}-{hi:g}s" if zoom else "")
    out = out_dir / f"{Path(path).stem}{suffix}.png"
    fig.savefig(out, dpi=130)
    print(f"{out}  ({len(turns)} turns)")
    return fig


SUMMARY = [
    ("r_dur", "Reply length", "seconds"),
    ("r_dn_bps", "Reply download rate", "bytes/s"),
    ("r_dn_size_mean", "Reply packet size, mean", "bytes"),
    ("r_dn_size_p10", "Reply packet size, 10th pct", "bytes"),
    ("r_dn_size_p90", "Reply packet size, 90th pct", "bytes"),
    ("r_dn_iat_cv", "Reply packet timing variability", "CV of gaps"),
    ("r_dn_pps", "Reply download packet rate", "packets/s"),
    ("r_up_bps", "Upload rate during reply", "bytes/s"),
]


def plot_summary(plt, out_dir):
    from train_model import load_dataset
    rows = load_dataset()
    langs = [l for l in LANGS if any(r["lang"] == l for r in rows)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    counts = ", ".join(f"{sum(r['lang'] == l for r in rows)} {LANG_NAME[l]}" for l in langs)
    fig.suptitle(f"Per-question packet features by language ({counts})\n"
                 "Each dot is one question/answer piece; black bars are medians.",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    for ax, (key, title, unit) in zip(axes.flat, SUMMARY):
        for i, lang in enumerate(langs):
            v = np.array([r[key] for r in rows if r["lang"] == lang])
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=18, color=LANG_COLOR[lang],
                       alpha=0.75, lw=0, label=LANG_NAME[lang])
            ax.hlines(np.median(v), i - 0.28, i + 0.28, color=INK, lw=1.5)
        ax.set_xticks(range(len(langs)), [LANG_NAME[l] for l in langs])
        ax.set_xlim(-0.6, len(langs) - 0.4)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
    axes.flat[0].legend(loc="best")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary_by_language.png"
    fig.savefig(out, dpi=130)
    print(out)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".pcapng / .pcap captures or processed .csv pieces")
    ap.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"), help="seconds to show")
    ap.add_argument("--turn", type=int, help="zoom on this detected turn (1 = first)")
    ap.add_argument("--summary", action="store_true", help="compare features across languages")
    ap.add_argument("--out", default=str(PLOTS_DIR), help="folder for the PNGs")
    ap.add_argument("--show", action="store_true", help="also open the figures in a window")
    args = ap.parse_args()
    if not args.files and not args.summary:
        ap.error("give one or more files, or --summary")

    plt = setup(args.show)
    out_dir = Path(args.out)
    for path in args.files:
        fig = plot_file(plt, path, args.zoom, args.turn, out_dir)
        if fig is not None and not args.show:
            plt.close(fig)
    if args.summary:
        plot_summary(plt, out_dir)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
