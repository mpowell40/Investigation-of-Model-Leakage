"""Redraw results/exp3_io_search_vs_static.png from the pcaps.

Annotation positions (preamble, stall, catch-up burst) are computed from the
search capture's downstream bins, not placed by hand.
Usage: python plot_example.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze

CASES = [
    ("Static", "What's Atlanta's weather usually like in October?", "tool_static_s1_pT01_r01.pcapng"),
    ("Search", "What's the weather in Atlanta right now?", "tool_search_s1_pT01_r01.pcapng"),
]
HOT = analyze.REPLY_BIN_BYTES / 1024   # KB per bin that counts as reply audio


def series(pairs):
    xs = [t + analyze.BIN / 2 for t, _ in pairs]   # plot each bin at its centre
    ys = [v / 1024 for _, v in pairs]
    return xs, ys


fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, sharey=True)
for ax, (kind, prompt, f) in zip(axes, CASES):
    r = analyze.analyze_file("captures/" + f)
    ux, uy = series(r["_up"])
    dx, dy = series(r["_down"])
    ax.axvspan(-r["speech_s"], 0, color="tab:green", alpha=0.08)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.plot(ux, uy, color="tab:green", label="Upstream (you → Claude)")
    ax.plot(dx, dy, color="crimson", label="Downstream (Claude → you)")
    ax.set_title(f'{kind}: "{prompt}"   ({f})', loc="left", fontsize=10)
    ax.set_ylabel("KB per 0.25 s")
    ax.grid(alpha=0.3)

    if kind == "Search":
        pts = [(x, y) for x, y in zip(dx, dy) if x > 0]
        hot = [i for i, (_, y) in enumerate(pts) if y >= HOT]
        # preamble = first run of hot bins; stall = cold bins after it; burst = peak after stall
        i = hot[0]
        while i + 1 < len(pts) and pts[i + 1][1] >= HOT:
            i += 1
        pre = pts[hot[0]:i + 1]
        j = i + 1
        while pts[j][1] < HOT:
            j += 1
        stall = pts[i + 1:j]
        peak = max(pts[j:], key=lambda p: p[1])

        px = (pre[0][0] + pre[-1][0]) / 2
        ax.annotate("preamble", xy=(px, max(y for _, y in pre) + 1), xytext=(px - 1.2, 45),
                    arrowprops=dict(arrowstyle="->"), ha="center")
        sx = (stall[0][0] + stall[-1][0]) / 2
        ax.axvspan(stall[0][0] - analyze.BIN / 2, stall[-1][0] + analyze.BIN / 2,
                   color="gray", alpha=0.15)
        ax.annotate(f"stall while searching ({r['dip_s']} s)",
                    xy=(sx, 2), xytext=(sx + 1.6, 9), arrowprops=dict(arrowstyle="->"))
        ax.annotate("catch-up burst", xy=(peak[0] + 0.05, peak[1]), xytext=(peak[0] + 1.3, peak[1] - 5),
                    arrowprops=dict(arrowstyle="->"))
    else:
        ax.text(-5.8, 52, "shaded green = you speaking; dashed line = you stopped", fontsize=9, color="dimgray")
        ax.legend(loc="upper right", fontsize=9)

axes[0].set_xlim(-6, 12.5)
axes[0].set_ylim(-2, 68)
axes[1].set_xlabel("seconds relative to the end of your speech")
fig.suptitle("Claude voice connection: static vs search reply (Anthropic TCP flow only)", x=0.01, ha="left")
fig.tight_layout()
fig.savefig("results/exp3_io_search_vs_static.png", dpi=150)
print("saved")
