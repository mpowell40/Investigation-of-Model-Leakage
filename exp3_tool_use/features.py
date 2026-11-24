"""Packet-metadata features for one Experiment 3 capture (one prompt, one reply).

Shared by train_model.py, predict_tool.py and plot_packets.py. A capture is a
list of (t, size, direction) packets plus the edges preprocess.py found:
speech_start, speech_end (your prompt) and reply_start, reply_end (Claude).
Metadata only: size, direction, timing.

Feature groups, by prefix:
  lat_   latency: end of your speech -> first reply packet, and the same minus
         the median latency of the base_ captures in the same sitting (lat_norm_s);
         also end of speech -> start of the biggest burst (the main answer)
  r_     Claude's reply: duration, download bytes/s and packets/s, sizes and
         histograms of audio-sized packets (> 80 B), inter-arrival mean and CV,
         download pauses inside the reply, audio in the first 1.5 s
  gap_   the latency gap itself: traffic between your speech end and the reply
  idle_  silent stretches before your speech and after the reply. DIAGNOSTIC
         ONLY, never in the main model: in Experiment 2 idle keep-alives carried a
         recording artifact. train_model.py checks whether they predict anything.
"""
import csv
import statistics
from pathlib import Path

import numpy as np

import preprocess

AUDIO_MIN = 80            # packets above this carry audio (smaller: ACKs, keep-alives)
FINE = 0.1                # s, bins for pause detection
STALL_BYTES = 1500        # a 0.1 s download bin below this is a pause in reply audio
MIN_PAUSE = 0.2           # s, shorter dips are jitter
EARLY_S = 1.5             # s, window for early reply audio
BURST_BIN = 0.25          # s, bins for the main-answer burst
BURST_BYTES = 4000        # download bytes per BURST_BIN that count as reply audio
IDLE_GUARD = 0.5          # s kept clear of speech/reply edges in idle windows

# Histogram edges for audio-sized packets (bytes on the wire). Reply download
# packets are mostly full 426 B segments; upload frames during the reply ~106-128 B.
DOWN_EDGES = [80, 200, 300, 350, 384, 400, 420, 425, 427, 100000]
UP_EDGES = [80, 104, 108, 111, 114, 117, 120, 124, 130, 200, 100000]

GROUPS = ("lat_", "r_", "gap_", "idle_")
MAIN_GROUPS = ("lat_", "r_", "gap_")
LENGTH_FEATURES = ("r_dur", "r_dn_bytes", "r_dn_pkts")   # what "reply length" means


def as_arrays(packets):
    """(t, size, is_up) numpy arrays from a list of (t, size, direction)."""
    if not packets:
        return np.zeros(0), np.zeros(0, int), np.zeros(0, bool)
    t, size, d = zip(*packets)
    return np.array(t, float), np.array(size, int), np.array(d) == "up"


def _stats(prefix, x):
    if len(x) == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "std", "p10", "p50", "p90")}
    return {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_std": float(np.std(x)),
            f"{prefix}_p10": float(np.percentile(x, 10)),
            f"{prefix}_p50": float(np.percentile(x, 50)),
            f"{prefix}_p90": float(np.percentile(x, 90))}


def _hist(prefix, x, edges):
    h = np.histogram(x, bins=edges)[0].astype(float)
    h = h / h.sum() if h.sum() else h
    return {f"{prefix}_h{i}": float(v) for i, v in enumerate(h)}


def _iat(prefix, ts):
    d = np.diff(ts)
    if len(d) == 0:
        return {f"{prefix}_iat_mean": 0.0, f"{prefix}_iat_cv": 0.0}
    m = float(np.mean(d))
    return {f"{prefix}_iat_mean": m, f"{prefix}_iat_cv": float(np.std(d) / m) if m else 0.0}


def _rates(prefix, t, size, up, start, end):
    dur = max(end - start, 1e-3)
    w = (t >= start) & (t < end)
    f = {}
    for d, is_up in (("up", True), ("dn", False)):
        sel = w & (up == is_up)
        f[f"{prefix}_{d}_bps"] = float(size[sel].sum()) / dur
        f[f"{prefix}_{d}_pps"] = float(sel.sum()) / dur
    return f, w


def reply_features(t, size, up, a, b):
    f = {"r_dur": b - a}
    rates, w = _rates("r", t, size, up, a, b)
    f.update(rates)
    dn = w & ~up
    f["r_dn_bytes"] = float(size[dn].sum())
    f["r_dn_pkts"] = float(dn.sum())
    for d, is_up, edges in (("up", True, UP_EDGES), ("dn", False, DOWN_EDGES)):
        sel = w & (up == is_up)
        audio = sel & (size > AUDIO_MIN)
        f[f"r_{d}_audio_frac"] = float(audio.sum()) / sel.sum() if sel.sum() else 0.0
        f.update(_stats(f"r_{d}_size", size[audio]))
        f.update(_hist(f"r_{d}_size", size[audio], edges))
        f.update(_iat(f"r_{d}", t[audio]))

    # Pauses: runs of quiet 0.1 s download bins inside the reply that audio resumes after.
    n = max(int(np.ceil((b - a) / FINE)), 1)
    bins = np.bincount(np.clip(((t[dn] - a) / FINE).astype(int), 0, n - 1),
                       weights=size[dn], minlength=n)
    pauses, run = [], 0
    for i, v in enumerate(bins):
        if v < STALL_BYTES:
            run += 1
        else:
            if run * FINE >= MIN_PAUSE - 1e-9:
                pauses.append(((i - run) * FINE, run * FINE))
            run = 0
    longest = max(pauses, key=lambda p: p[1]) if pauses else (0.0, 0.0)
    f["r_pause_max"] = longest[1]
    f["r_pause_at"] = longest[0]
    f["r_pause_total"] = float(sum(p[1] for p in pauses))
    f["r_pause_count"] = float(len(pauses))
    early = float(bins[: int(round(EARLY_S / FINE))].sum())
    f["r_early_kb"] = early / 1024
    f["r_early_frac"] = early / f["r_dn_bytes"] if f["r_dn_bytes"] else 0.0
    coarse = np.bincount(((t[dn] - a) / 0.5).astype(int), weights=size[dn])
    f["r_dn_bin_cv"] = float(np.std(coarse) / np.mean(coarse)) if coarse.mean() else 0.0
    f["r_dn_bin_max"] = float(coarse.max()) / 0.5 if len(coarse) else 0.0
    return f


def main_answer_start(t, size, up, a, b):
    """Start (s) of the biggest download burst in the reply, at BURST_BIN resolution."""
    dn = (t >= a) & (t < b) & ~up
    if not dn.any():
        return a
    bins = np.bincount(((t[dn] - a) / BURST_BIN).astype(int), weights=size[dn])
    hot = np.nonzero(bins >= BURST_BYTES)[0]
    if len(hot) == 0:
        return a
    bursts, cur = [], [hot[0], hot[0], bins[hot[0]]]
    for i in hot[1:]:
        if (i - cur[1]) * BURST_BIN <= 0.5:
            cur[1], cur[2] = i, cur[2] + bins[i]
        else:
            bursts.append(cur)
            cur = [i, i, bins[i]]
    bursts.append(cur)
    return a + max(bursts, key=lambda x: x[2])[0] * BURST_BIN


def gap_features(t, size, up, speech_end, reply_start):
    w = (t > speech_end) & (t < reply_start)
    f = {}
    for d, is_up in (("up", True), ("dn", False)):
        sel = w & (up == is_up)
        f[f"gap_{d}_pkts"] = float(sel.sum())
        f[f"gap_{d}_bytes"] = float(size[sel].sum())
        f[f"gap_{d}_large"] = float((sel & (size > preprocess.CTRL_MIN)).sum())
        f[f"gap_{d}_size_max"] = float(size[sel].max()) if sel.any() else 0.0
    dur = max(reply_start - speech_end, 1e-3)
    f["gap_dn_bps"] = f["gap_dn_bytes"] / dur
    f["gap_up_bps"] = f["gap_up_bytes"] / dur
    return f


def idle_features(t, size, up, speech_start, reply_end):
    """Silent stretches: before you speak and after Claude finishes."""
    first_frame = t[up & (size > preprocess.ACK_MAX)].min() if (up & (size > preprocess.ACK_MAX)).any() else 0
    windows = {"pre": (first_frame + preprocess.SKIP_START, speech_start - IDLE_GUARD),
               "post": (reply_end + IDLE_GUARD, float(t.max()) if len(t) else reply_end)}
    f = {}
    for name, (a, b) in windows.items():
        if b - a < 0.5:
            a, b = a, a + 0.5
        rates, w = _rates(f"idle_{name}", t, size, up, a, b)
        f.update(rates)
        upf = w & up & (size > preprocess.ACK_MAX) & (size < preprocess.CTRL_MIN)
        f[f"idle_{name}_up_frame_mean"] = float(size[upf].mean()) if upf.any() else 0.0
        dn = w & ~up
        f[f"idle_{name}_dn_size_mean"] = float(size[dn].mean()) if dn.any() else 0.0
        f[f"idle_{name}_dn_size_mode"] = float(statistics.mode(size[dn].tolist())) if dn.any() else 0.0
    return f


def capture_features(packets, edges, baseline=None):
    """Feature dict for one capture. edges: speech_start/end, reply_start/end (s).

    baseline: median base_ latency for the capture's sitting (None -> lat_norm_s = lat_s).
    """
    t, size, up = as_arrays(packets)
    ss, se = edges["speech_start"], edges["speech_end"]
    a, b = edges["reply_start"], edges["reply_end"]
    lat = a - se
    main = main_answer_start(t, size, up, a, b)
    f = {"lat_s": lat, "lat_norm_s": lat - (baseline if baseline is not None else 0.0),
         "lat_main_s": main - se}
    f.update(reply_features(t, size, up, a, b))
    f.update(gap_features(t, size, up, se, a))
    f.update(idle_features(t, size, up, ss, b))
    return f


def feature_names(f, groups=MAIN_GROUPS):
    return [k for k in f if k.startswith(tuple(groups))]


# ---------- loading ----------

def read_index():
    path = preprocess.OUT_DIR / "index.csv"
    if not path.exists():
        raise SystemExit("No processed data yet. Run:  python preprocess.py")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def edges_of(row):
    return {k: float(row[k]) for k in ("speech_start", "speech_end", "reply_start", "reply_end")}


def sitting_baselines(index_rows):
    """{sitting: median first-reply latency of its base_ captures}."""
    out = {}
    for s in {r["sitting"] for r in index_rows}:
        lat = [float(r["latency"]) for r in index_rows
               if r["sitting"] == s and r["class"] == "base" and r["status"].startswith("ok")]
        if lat:
            out[s] = statistics.median(lat)
    return out


def load_dataset(skip_flagged=True):
    """One dict per usable capture: index columns + 'features'."""
    index = read_index()
    base = sitting_baselines(index)
    rows = []
    for r in index:
        if not r["status"].startswith("ok"):
            if skip_flagged:
                print(f"  skipping {r['file']}: {r['status']}")
            continue
        packets = preprocess.read_packets_csv(preprocess.OUT_DIR / (Path(r["file"]).stem + ".csv"))
        rows.append({**r, "features": capture_features(packets, edges_of(r), base.get(r["sitting"]))})
    return rows, base


def from_capture(path, baseline=None):
    """(packets, edges or None, status, features or None) for a raw .pcapng or processed .csv."""
    path = Path(path)
    if path.suffix == ".csv":
        packets = preprocess.read_packets_csv(path)
    else:
        packets, _, _ = preprocess.load_flow(path)
    if not packets:
        return [], None, "no port-443 traffic", None
    det = preprocess.detect(packets)
    if det["status"] != "ok":
        return packets, None, det["status"], None
    edges = {k: det[k] for k in ("speech_start", "speech_end", "reply_start", "reply_end")}
    return packets, edges, "ok", capture_features(packets, edges, baseline)
