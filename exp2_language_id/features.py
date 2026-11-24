"""Packet-metadata features for one question/answer turn.

Shared by train_model.py, predict_language.py and plot_packets.py. A turn is a
list of (t, size, direction) packets plus where Claude's reply starts and ends,
as written by preprocess.py. Metadata only: size, direction, timing.

Features come from Claude's reply only (prefix r_). The question window is kept
for plotting, but its features were dropped: they carried a recording artifact
(idle keep-alive packet sizes differ between sessions) and no language signal.
"""
import csv
from pathlib import Path

import numpy as np

import preprocess

QUESTION_WINDOW = 8.0     # seconds before a reply that hold the spoken question
ACK_MAX = 80              # packets this small carry no audio (TCP ACKs, keep-alives)
BIN = 0.5

# Histogram edges for audio-carrying packet sizes (bytes on the wire).
DOWN_EDGES = [0, 160, 170, 175, 178, 181, 185, 190, 200, 250, 100000]
UP_EDGES = [0, 110, 118, 122, 126, 130, 134, 138, 150, 200, 100000]


def as_arrays(packets):
    """(t, size, is_up) numpy arrays from a list of (t, size, direction)."""
    if not packets:
        return np.zeros(0), np.zeros(0, int), np.zeros(0, bool)
    t, size, d = zip(*packets)
    return np.array(t, float), np.array(size, int), np.array(d) == "up"


def read_piece(path):
    """Packets of one processed/questions CSV."""
    with open(path, newline="") as f:
        return [(float(r["t"]), int(r["size"]), r["direction"]) for r in csv.DictReader(f)]


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


def _window(name, t, size, up, start, end):
    dur = max(end - start, BIN)
    w = (t >= start) & (t < end)
    f = {f"{name}_dur": end - start}
    for d, is_up, edges in (("up", True, UP_EDGES), ("dn", False, DOWN_EDGES)):
        sel = w & (up == is_up)
        audio = sel & (size > ACK_MAX)
        f[f"{name}_{d}_bps"] = float(size[sel].sum()) / dur
        f[f"{name}_{d}_pps"] = float(sel.sum()) / dur
        f[f"{name}_{d}_audio_frac"] = float(audio.sum()) / sel.sum() if sel.sum() else 0.0
        f.update(_stats(f"{name}_{d}_size", size[audio]))
        f.update(_hist(f"{name}_{d}_size", size[audio], edges))
        f.update(_iat(f"{name}_{d}", t[audio]))
    return f


def turn_features(packets, reply_start, reply_end):
    """Feature dict for one turn, from Claude's reply."""
    t, size, up = as_arrays(packets)
    f = _window("r", t, size, up, reply_start, reply_end)
    # How bursty the reply is: per-bin download bytes across the reply.
    w = (t >= reply_start) & (t < reply_end) & ~up
    if w.any():
        bins = np.bincount(((t[w] - reply_start) / BIN).astype(int), weights=size[w]).astype(float)
        f["r_dn_bin_cv"] = float(np.std(bins) / np.mean(bins)) if np.mean(bins) else 0.0
        f["r_dn_bin_max"] = float(bins.max()) / BIN
    else:
        f["r_dn_bin_cv"] = f["r_dn_bin_max"] = 0.0
    return f


def turns_from_capture(path):
    """Turns of a raw capture (.pcapng/.pcap) or a processed question CSV.

    Returns (packets, turns) where turns is a list of
    {"turn", "q_start", "r_start", "r_end", "features"}.
    """
    path = Path(path)
    packets = read_piece(path) if path.suffix == ".csv" else preprocess.load_flow(path)
    replies = preprocess.find_replies(packets)
    turns, prev_end = [], 0.0
    for i, (a, b) in enumerate(replies, 1):
        turns.append({"turn": i, "q_start": max(prev_end, a - QUESTION_WINDOW), "r_start": a,
                      "r_end": b, "features": turn_features(packets, a, b)})
        prev_end = b
    return packets, turns
