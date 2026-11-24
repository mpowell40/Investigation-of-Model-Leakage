"""Experiment 1 plots (matplotlib), written to figures/.

    python plot_packets.py captures/phrase_batchA_s1_r01.pcapng           # whole file
    python plot_packets.py captures/phrase_batchA_s1_r01.pcapng --turn 3  # one turn
    python plot_packets.py captures/phrase_batchA_s1_r01.pcapng --zoom 40 60
    python plot_packets.py --summary                                      # features by class

Per-file figure: every packet of the voice flow over time (upload and download),
throughput per direction, the smoothed download packet rate and upload frame size
with the adaptive thresholds used by preprocess.py, and size histograms. Turns
are shaded (green = your speech, grey = Claude's reply) and numbered with the
logged phrase.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import preprocess as P

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# Reference palette (dataviz skill), light mode.
UP, DOWN = "#2a78d6", "#eb6834"                # categorical slots 1, 2
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SPEECH_SHADE, REPLY_SHADE = "#1baf7a", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def shade(ax, turns, labels, numbers=False):
    for k, (t, lab) in enumerate(zip(turns, labels), start=1):
        s0, s1 = t["speech"]
        if not np.isnan(s0):
            ax.axvspan(s0, s1, color=SPEECH_SHADE, alpha=0.18, lw=0)
        for a, b in t["replies"]:
            ax.axvspan(a, b, color=REPLY_SHADE, alpha=0.14, lw=0)
        if numbers:
            x = s0 if not np.isnan(s0) else t["replies"][0][0]
            lo, hi = ax.get_xlim()
            if not lo <= x <= hi:
                continue
            ax.text(x, 1.01, f"{k}" + (f" {lab}" if lab else ""), transform=ax.get_xaxis_transform(),
                    color=INK, fontsize=8, fontweight="bold", va="bottom")


def plot_file(path, zoom=None, turn=None):
    meta = P.load_manifest().get(path.name)
    rows, frow, (up, dn, info, det, turns) = P.process_file(path, meta, write=False)
    labels = [r["label"] for r in rows]
    if turn is not None:
        if not 1 <= turn <= len(rows):
            raise SystemExit(f"--turn must be 1..{len(rows)}")
        r = rows[turn - 1]
        zoom = (r["piece_start"], r["piece_end"])
    t_bins = np.arange(0, len(det["dn_rate"])) * P.BIN

    fig = plt.figure(figsize=(13, 11), constrained_layout=True)
    gs = fig.add_gridspec(6, 2, height_ratios=[1.2, 1.2, 1, 1, 1, 1.3])
    ax_up = fig.add_subplot(gs[0, :])
    axes = [ax_up] + [fig.add_subplot(gs[i, :], sharex=ax_up) for i in range(1, 5)]
    ax_dn, ax_tp, ax_rate, ax_size = axes[1:]

    ax_up.scatter(up[:, 0], up[:, 1], s=1.5, color=UP, lw=0)
    ax_up.set_ylabel("upload\npacket size (B)")
    ax_dn.scatter(dn[:, 0], dn[:, 1], s=1.5, color=DOWN, lw=0)
    ax_dn.set_ylabel("download\npacket size (B)")
    for a in (ax_up, ax_dn):
        a.set_yscale("log")
        a.axhline(P.AUDIO_MIN, color=INK2, lw=0.8, ls="--")
    ax_up.text(0.995, P.AUDIO_MIN, f"{P.AUDIO_MIN} B: ACKs below", transform=ax_up.get_yaxis_transform(),
               ha="right", va="bottom", color=INK2, fontsize=7)

    tb = 0.25
    nb = int(info["duration"] / tb) + 1
    x = np.arange(nb) * tb
    for arr, col, name in ((up, UP, "upload"), (dn, DOWN, "download")):
        y = np.bincount((arr[:, 0] / tb).astype(int), weights=arr[:, 1], minlength=nb)[:nb] / tb / 1e3
        ax_tp.plot(x, y, color=col, lw=1.2, label=name)
    ax_tp.set_ylabel("throughput\n(kB/s, 0.25 s)")
    ax_tp.legend(loc="upper right", ncol=2)

    ax_rate.plot(t_bins, det["dn_rate"], color=DOWN)
    ax_rate.axhline(det["reply_thr"], color=INK, lw=0.9, ls="--")
    ax_rate.text(0.995, det["reply_thr"], f"reply threshold {det['reply_thr']:.0f} pkt/s",
                 transform=ax_rate.get_yaxis_transform(), ha="right", va="bottom", color=INK, fontsize=7)
    ax_rate.set_ylabel("download rate\n(pkt/s, 0.5 s smooth)")

    ax_size.plot(t_bins, det["up_size"], color=UP)
    ax_size.axhline(det["speech_thr"], color=INK, lw=0.9, ls="--")
    ax_size.text(0.995, det["speech_thr"], f"speech threshold {det['speech_thr']:.0f} B",
                 transform=ax_size.get_yaxis_transform(), ha="right", va="bottom", color=INK, fontsize=7)
    ax_size.set_ylabel("upload audio frame\n(mean B, 0.5 s smooth)")
    ax_size.set_xlabel("time from start of voice flow (s)")

    if zoom:
        ax_up.set_xlim(*zoom)
    else:
        ax_up.set_xlim(0, info["duration"])
    ax_size.set_ylim(100, max(230, det["speech_thr"] + 60))   # hide the start-up spike
    for i, a in enumerate(axes):
        shade(a, turns, labels, numbers=(i == 0))
        if a is not ax_size:
            plt.setp(a.get_xticklabels(), visible=False)

    lo, hi = zoom if zoom else (0, info["duration"])
    hbins = np.arange(80, 420, 4)
    for j, (arr, col, name) in enumerate(((up, UP, "upload"), (dn, DOWN, "download"))):
        a = fig.add_subplot(gs[5, j])
        w = arr[(arr[:, 0] >= lo) & (arr[:, 0] < hi), 1]
        a.hist(w[w < 420], bins=hbins, color=col, edgecolor=SURFACE, linewidth=0.3)
        a.set_title(f"{name} packet sizes ({len(w)} packets, {np.mean(w >= 420):.1%} ≥ 420 B not shown)",
                    loc="left", fontsize=9)
        a.set_xlabel("packet size (B)")
        a.set_ylabel("packets")

    span = f", {lo:.0f}-{hi:.0f} s" if zoom else ""
    title = (f"{path.name}{span} — {frow['status']}, {frow['turns']} turns"
             f" (expected {frow['expected_turns']}), flow {info['proto']} {info['server']}")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.99, 0.995, "green: your speech   grey: Claude's reply", ha="right", va="top",
             color=INK2, fontsize=8)
    FIGURES.mkdir(exist_ok=True)
    suffix = f"_turn{turn}" if turn else (f"_{lo:.0f}-{hi:.0f}s" if zoom else "")
    out = FIGURES / f"{path.stem}{suffix}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


SUMMARY_FEATURES = [
    ("speech_dur", "your speech duration (s)"),
    ("speech_excess_kB", "speech bytes above silence (kB)"),
    ("reply_dur", "Claude's reply duration (s)"),
    ("reply_dn_audio_kB", "reply audio received (kB)"),
    ("reply_early_kB", "reply audio in first 1.5 s (kB)"),
    ("pre_dn_msgs", "server messages while you speak"),
]


def plot_summary():
    path = RESULTS / "features.csv"
    if not path.exists():
        raise SystemExit("No results/features.csv. Run: python features.py")
    rows = [r for r in csv.DictReader(open(path, newline="")) if r["status"] == "ok" and r["label"]]
    outs = []
    for fset, title in (("batchA", "Set A: P02–P07 (filled = reps 1–16, hollow = held-out reps 17–20)"),
                        ("batchB", "Set B: D01 / D03 / D05 (all Oct 25)")):
        sub = [r for r in rows if r["set"] == fset]
        if not sub:
            continue
        classes = sorted({r["label"] for r in sub})
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        rng = np.random.default_rng(0)
        for ax, (feat, label) in zip(axes.flat, SUMMARY_FEATURES):
            data = [[float(r[feat]) for r in sub if r["label"] == c and r[feat] != ""] for c in classes]
            ax.boxplot(data, positions=range(len(classes)), widths=0.55, showfliers=False,
                       medianprops=dict(color=INK, lw=1.4), boxprops=dict(color=INK2, lw=0.8),
                       whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
            for i, c in enumerate(classes):
                for r in (r for r in sub if r["label"] == c and r[feat] != ""):
                    held = r["split"] == "heldout"
                    ax.scatter(i + rng.uniform(-0.18, 0.18), float(r[feat]), s=22,
                               facecolor="none" if held else SERIES[i % len(SERIES)],
                               edgecolor=SERIES[i % len(SERIES)], linewidth=1.1 if held else 0.6, zorder=3)
            ax.set_xticks(range(len(classes)), classes)
            ax.set_title(label, loc="left")
            ax.grid(axis="x", visible=False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
        FIGURES.mkdir(exist_ok=True)
        out = FIGURES / f"summary_{fset}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=".pcapng captures")
    ap.add_argument("--zoom", nargs=2, type=float, metavar=("T0", "T1"), help="time window in seconds")
    ap.add_argument("--turn", type=int, help="zoom to turn N (1-based)")
    ap.add_argument("--summary", action="store_true", help="feature distributions by class")
    args = ap.parse_args()
    if not args.files and not args.summary:
        ap.error("give a capture file or --summary")
    for f in args.files:
        print(plot_file(Path(f), args.zoom, args.turn))
    if args.summary:
        for o in plot_summary():
            print(o)


if __name__ == "__main__":
    sys.exit(main())
