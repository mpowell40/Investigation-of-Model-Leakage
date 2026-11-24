"""Experiment 1 features: one row per turn piece, from packet metadata only.

Reads processed/index.csv + processed/*.npz, writes results/features.csv.

Feature groups (column prefix):
  reply_*   Claude's reply: duration, bytes/s and packets/s per direction,
            sizes and size histogram of audio-sized downstream packets,
            inter-arrival mean/CV, burstiness, early-reply volume, fragments.
  lat_*     latency from end of the user's speech to the first reply audio.
  pre_*     Experiment-specific. While you speak, the server sends a train of
            small non-audio download messages (~150-270 B, a few bytes bigger
            every ~0.2 s, plus a few ~300-450 B ones) before the reply audio.
            They look like live status/transcript updates (we don't read
            payloads, so that's a guess). Their count and volume track how long
            you spoke; the size of the largest is loosely related to sentence length.
  speech_*  Experiment-specific. The goal is to tell which sentence was SAID,
            and the user's own audio frames are the most direct trace of it:
            the client sends 50 frames/s with sizes that follow speech energy,
            so frame count over the threshold ~ how long you spoke, and the
            frame-size contour ~ the rhythm (syllables, pauses) of the sentence.
            Reply features only tell us what Claude said back, which is a
            noisier proxy for the question.

Audio-sized means > 100 B: 86 B packets are bare TCP ACKs (IPv6), not audio.

    python features.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
RESULTS = ROOT / "results"

AUDIO_MIN = 100
SIZE_BINS = [100, 150, 175, 200, 225, 250, 300, 600, 1500]   # downstream audio sizes
UP_BINS = [100, 140, 160, 170, 180, 190, 200, 1500]           # upstream speech frame sizes
CONTOUR = 5        # speech frame-size contour: mean size in N equal slices
BURST_BIN = 0.25   # s, for burstiness of the reply


def _stats(x, prefix):
    if len(x) == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "std", "med", "p10", "p90")}
    return {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_std": float(np.std(x)),
            f"{prefix}_med": float(np.median(x)), f"{prefix}_p10": float(np.percentile(x, 10)),
            f"{prefix}_p90": float(np.percentile(x, 90))}


def _hist(x, bins, prefix):
    h, _ = np.histogram(x, bins=bins)
    tot = max(h.sum(), 1)
    return {f"{prefix}_{a}_{b}": float(n / tot) for a, b, n in zip(bins[:-1], bins[1:], h)}


def _iat(t, prefix):
    if len(t) < 3:
        return {f"{prefix}_iat_mean": 0.0, f"{prefix}_iat_cv": 0.0}
    d = np.diff(np.sort(t))
    return {f"{prefix}_iat_mean": float(d.mean()), f"{prefix}_iat_cv": float(d.std() / max(d.mean(), 1e-9))}


def _window(a, t0, t1):
    return a[(a[:, 0] >= t0) & (a[:, 0] < t1)] if len(a) else a


def turn_features(up, dn, replies, speech):
    """Feature dict for one turn. Times are seconds from the capture start."""
    f = {}
    r0, r1 = float(replies[0][0]), float(replies[-1][1])
    dur = max(r1 - r0, 1e-3)
    ru, rd = _window(up, r0, r1), _window(dn, r0, r1)
    rda = rd[rd[:, 1] > AUDIO_MIN] if len(rd) else rd

    # ---- Claude's reply
    f["reply_dur"] = dur
    f["reply_fragments"] = len(replies)
    gaps = [b[0] - a[1] for a, b in zip(replies[:-1], replies[1:])]
    f["reply_max_gap"] = float(max(gaps)) if gaps else 0.0
    f["reply_dn_Bps"] = float(rd[:, 1].sum() / dur) if len(rd) else 0.0
    f["reply_dn_pps"] = len(rd) / dur
    f["reply_up_Bps"] = float(ru[:, 1].sum() / dur) if len(ru) else 0.0
    f["reply_up_pps"] = len(ru) / dur
    f["reply_dn_audio_kB"] = float(rda[:, 1].sum() / 1e3) if len(rda) else 0.0
    f["reply_dn_audio_pkts"] = len(rda)
    f["reply_dn_audio_frac"] = len(rda) / max(len(rd), 1)
    f.update(_stats(rda[:, 1] if len(rda) else [], "reply_dn_size"))
    f.update(_hist(rda[:, 1] if len(rda) else [], SIZE_BINS, "reply_dn_hist"))
    f.update(_iat(rda[:, 0] if len(rda) else [], "reply_dn"))
    nb = max(int(dur / BURST_BIN), 1)
    if len(rda):
        b = np.bincount(np.minimum(((rda[:, 0] - r0) / BURST_BIN).astype(int), nb - 1),
                        weights=rda[:, 1], minlength=nb)
        f["reply_burst_cv"] = float(b.std() / max(b.mean(), 1e-9))
        f["reply_burst_peak_ratio"] = float(b.max() / max(b.mean(), 1e-9))
        f["reply_burst_low_frac"] = float((b < 0.5 * np.median(b)).mean())
        f["reply_early_kB"] = float(rda[rda[:, 0] < r0 + 1.5, 1].sum() / 1e3)
        f["reply_late_kB"] = float(rda[rda[:, 0] >= r1 - 1.5, 1].sum() / 1e3)
    else:
        f.update(reply_burst_cv=0.0, reply_burst_peak_ratio=0.0, reply_burst_low_frac=0.0,
                 reply_early_kB=0.0, reply_late_kB=0.0)

    # ---- latency
    s0, s1 = float(speech[0]), float(speech[1])
    have_speech = not (np.isnan(s0) or np.isnan(s1))
    f["lat_reply"] = r0 - s1 if have_speech else np.nan
    first = rda[0, 0] if len(rda) else r0
    f["lat_first_audio"] = first - s1 if have_speech else np.nan

    # ---- server messages while the user speaks (experiment-specific)
    if have_speech:
        pre = _window(dn, s0 - 0.2, r0)
        pre = pre[pre[:, 1] > AUDIO_MIN] if len(pre) else pre
        ps = pre[:, 1] if len(pre) else np.array([])
        f["pre_dn_msgs"] = len(ps)
        f["pre_dn_kB"] = float(ps.sum() / 1e3)
        f["pre_dn_max"] = float(ps.max()) if len(ps) else 0.0
        f["pre_dn_big_msgs"] = int((ps > 300).sum())
        f["pre_dn_rate"] = len(ps) / max(r0 - s0 + 0.2, 1e-3)

    # ---- the user's speech (experiment-specific)
    if have_speech:
        sd = max(s1 - s0, 1e-3)
        su = _window(up, s0, s1)
        sua = su[su[:, 1] > AUDIO_MIN] if len(su) else su
        sizes = sua[:, 1] if len(sua) else np.array([])
        f["speech_dur"] = sd
        f["speech_up_Bps"] = float(su[:, 1].sum() / sd) if len(su) else 0.0
        f["speech_up_pps"] = len(su) / sd
        f["speech_up_kB"] = float(sizes.sum() / 1e3)
        f["speech_frames"] = len(sizes)
        # bytes above a silent frame: ~ total speech energy
        base = 118.0
        f["speech_excess_kB"] = float(np.clip(sizes - base, 0, None).sum() / 1e3)
        f["speech_loud_frames"] = int((sizes >= 170).sum())
        f.update(_stats(sizes, "speech_size"))
        f.update(_hist(sizes, UP_BINS, "speech_hist"))
        f.update(_iat(sua[:, 0] if len(sua) else [], "speech_up"))
        # frame-size contour over 5 equal slices, and number of dips (pauses)
        if len(sua) >= CONTOUR:
            edges = np.linspace(s0, s1, CONTOUR + 1)
            for i in range(CONTOUR):
                m = sizes[(sua[:, 0] >= edges[i]) & (sua[:, 0] < edges[i + 1])]
                f[f"speech_contour_{i + 1}"] = float(m.mean()) if len(m) else base
            k = 5   # 0.1 s at 50 frames/s
            sm = np.convolve(sizes, np.ones(k) / k, mode="valid")
            low = sm < 160
            f["speech_dips"] = int(np.sum(np.diff(low.astype(int)) == 1))
            f["speech_low_frac"] = float(low.mean())
        else:
            for i in range(CONTOUR):
                f[f"speech_contour_{i + 1}"] = base
            f["speech_dips"], f["speech_low_frac"] = 0, 0.0
    return f


def load_piece(piece):
    z = np.load(PROCESSED / f"{piece}.npz")
    return z["up"], z["dn"], z["replies"], z["speech"]


def main():
    idx_path = PROCESSED / "index.csv"
    if not idx_path.exists():
        raise SystemExit("No processed/index.csv. Run: python preprocess.py")
    with open(idx_path, newline="") as f:
        index = list(csv.DictReader(f))
    keep = ["piece", "source_file", "set", "rep", "sitting", "split", "turn", "label", "status"]
    rows = []
    for r in index:
        up, dn, replies, speech = load_piece(r["piece"])
        feats = turn_features(up, dn, replies, speech)
        rows.append({**{k: r[k] for k in keep}, **feats})
    cols = keep + sorted({k for r in rows for k in r if k not in keep})
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "features.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) and not np.isnan(v) else ("" if isinstance(v, float) else v))
                        for k, v in r.items()})
    n_feat = len(cols) - len(keep)
    print(f"{len(rows)} turns x {n_feat} features -> {out}")


if __name__ == "__main__":
    sys.exit(main())
