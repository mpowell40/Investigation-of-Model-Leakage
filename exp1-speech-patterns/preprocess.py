"""Experiment 1 preprocessing: voice flow -> one piece per spoken turn.

Metadata only (time, size, direction); payloads are never read.

For every capture in captures/:
  1. Keep only the voice flow: the busiest port-443 flow, checked against the
     Anthropic ranges from Exp 3.
  2. Detect Claude's replies: a smoothed 0.5 s downstream packet rate
     (~25-50 pkt/s of ACKs in silence, ~100-115 pkt/s while Claude talks),
     thresholded at the midpoint of its 10th and 90th percentiles.
  3. Detect the user's speech. Upstream the client sends 50 frames/s whether you
     talk or not, so the upstream packet *rate* is flat. The same adaptive rule is
     applied to the smoothed mean upstream audio-frame size instead
     (~118-132 B in silence, ~180 B while speaking), with the percentiles taken
     over the time Claude isn't replying.
  4. Pair them: an utterance (speech bursts with no reply in between) plus all
     reply fragments up to the next utterance is one turn. If the number of turns
     doesn't match the phrase order in the manifest, the file is flagged and its
     turns are left unlabelled rather than guessed.

Writes processed/<file>_tNN.npz, processed/index.csv (one row per turn) and
processed/files.csv (one row per capture, with detection status).

    python preprocess.py              # all captures
    python preprocess.py FILE.pcapng  # one file (prints the detection)
"""
import argparse
import collections
import csv
import datetime as dt
import ipaddress
import json
import re
import sys
from pathlib import Path

import numpy as np

from pcaplib import read_packets

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
PROCESSED = ROOT / "processed"
MANIFEST = ROOT / "manifest_exp1.csv"

ANTHROPIC_NETS = [ipaddress.ip_network("2607:6bc0::/48"),
                  ipaddress.ip_network("160.79.104.0/23")]

# 86 B is a bare TCP ACK over IPv6 (14 Ethernet + 40 IPv6 + 32 TCP with
# timestamps); about 40% of packets in each direction. Anything above 100 B
# carries audio.
AUDIO_MIN = 100

BIN = 0.05            # s, resolution of the rate/size series
SMOOTH = 0.5          # s, smoothing window (as in the protocol)
MIN_REPLY = 0.5       # s, shorter downstream bursts are ignored
MIN_SPEECH = 0.3      # s, shorter upstream bursts are ignored
SPEECH_MERGE = 0.6    # s, pauses shorter than this stay inside one speech burst
UTTER_GAP = 1.5       # s, bursts further apart than this are not one sentence
NOISE_MERGE = 1.5     # s, noise files: reply fragments closer than this merge
PAD = 1.0             # s, kept before the utterance in each piece

# Held-out test set (from the protocol notes): never trained or tuned on.
HELDOUT = {"batchA": set(range(17, 21)), "noise": {3}}

FILE_RE = re.compile(r"^phrase_(?P<set>batchA|batchB|noise)_s(?P<spk>\d+)_r(?P<rep>\d\d)\.pcapng$")
SITTING_GAP_H = 2.0   # captures more than 2 h apart start a new sitting


# --------------------------------------------------------------------------- manifest
def parse_name(name):
    m = FILE_RE.match(Path(name).name)
    if not m:
        return None
    return {"set": m["set"], "speaker": f"S{int(m['spk'])}", "rep": int(m["rep"])}


def load_manifest(path=MANIFEST):
    """file name -> dict(set, rep, order[list], when, sitting, split)."""
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        when = dt.datetime.strptime(r["date_time"].strip(), "%b %d, %Y, %I:%M %p")
        out[r["file"]] = {
            "set": r["class"], "rep": int(r["rep"]), "speaker": r["speaker"],
            "order": [p.strip() for p in r["prompt_id"].split(",") if p.strip()],
            "when": when, "reply_words": r.get("reply_words", ""),
        }
    # Sittings: consecutive captures less than SITTING_GAP_H apart.
    names = sorted(out, key=lambda n: out[n]["when"])
    sit, prev, label = 0, None, ""
    for n in names:
        w = out[n]["when"]
        if prev is None or (w - prev).total_seconds() > SITTING_GAP_H * 3600:
            sit += 1
            label = f"sit{sit}_{w:%b%d}"          # named after the sitting's first capture
        out[n]["sitting"] = label
        prev = w
    for n, m in out.items():
        m["split"] = "heldout" if m["rep"] in HELDOUT.get(m["set"], ()) else "train"
    return out


# --------------------------------------------------------------------------- flow
def voice_flow(path):
    """Return (up, dn, info): up/dn are (n,2) arrays of [t, size], t from flow start."""
    pkts = read_packets(path)
    bytes_by_flow = collections.Counter()
    for p in pkts:
        if p.proto is None:
            continue
        if p.dport == 443:
            bytes_by_flow[(p.proto, p.dst, p.sport)] += p.length
        elif p.sport == 443:
            bytes_by_flow[(p.proto, p.src, p.dport)] += p.length
    if not bytes_by_flow:
        raise ValueError("no port-443 traffic")
    ranked = bytes_by_flow.most_common(2)
    (proto, server, lport), top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    up = np.array([(p.ts, p.length) for p in pkts
                   if p.proto == proto and p.dst == server and p.sport == lport and p.dport == 443], float)
    dn = np.array([(p.ts, p.length) for p in pkts
                   if p.proto == proto and p.src == server and p.dport == lport and p.sport == 443], float)
    t0 = min(up[0, 0], dn[0, 0])
    up[:, 0] -= t0
    dn[:, 0] -= t0
    info = {"proto": proto, "server": server, "local_port": lport,
            "anthropic": any(ipaddress.ip_address(server) in n for n in ANTHROPIC_NETS),
            "flow_share": top / sum(bytes_by_flow.values()),
            "dominance": top / max(second, 1), "t0": t0,
            "duration": max(up[-1, 0], dn[-1, 0])}
    return up, dn, info


# --------------------------------------------------------------------------- detection
def _smooth(x):
    k = int(round(SMOOTH / BIN))
    return np.convolve(x, np.ones(k) / k, mode="same")


def _runs(mask, min_len, merge_gap=0.0):
    """Contiguous True runs as [(start_s, end_s)], merging gaps < merge_gap."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    runs = [[a * BIN, b * BIN] for a, b in zip(idx[::2], idx[1::2])]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged if r[1] - r[0] >= min_len]


def _midpoint(x):
    return (np.percentile(x, 10) + np.percentile(x, 90)) / 2


def series(up, dn, duration):
    nb = int(duration / BIN) + 1
    dn_rate = _smooth(np.bincount((dn[:, 0] / BIN).astype(int), minlength=nb)[:nb] / BIN)
    a = up[up[:, 1] > AUDIO_MIN]
    ib = (a[:, 0] / BIN).astype(int)
    s = np.bincount(ib, weights=a[:, 1], minlength=nb)[:nb]
    c = np.bincount(ib, minlength=nb)[:nb]
    size = np.full(nb, np.nan)
    size[c > 0] = s[c > 0] / c[c > 0]
    # forward-fill empty bins, then smooth
    last = np.nanmedian(size)
    for i in range(nb):
        if np.isnan(size[i]):
            size[i] = last
        else:
            last = size[i]
    return dn_rate, _smooth(size)


def detect(up, dn, duration):
    dn_rate, up_size = series(up, dn, duration)
    r_thr = _midpoint(dn_rate)
    replies = _runs(dn_rate > r_thr, MIN_REPLY)
    # The user is silent while Claude talks (protocol), so reply time only adds
    # silence frames and would pull the 90th percentile down into the silence
    # levels (~118 and ~132 B). Take the percentiles over non-reply time.
    quiet = np.ones(len(up_size), bool)
    for a, b in replies:
        quiet[int(a / BIN):int(b / BIN) + 1] = False
    s_thr = _midpoint(up_size[quiet] if quiet.sum() > 20 else up_size)
    speech = _runs(up_size > s_thr, MIN_SPEECH, SPEECH_MERGE)
    return {"dn_rate": dn_rate, "up_size": up_size, "reply_thr": r_thr,
            "speech_thr": s_thr, "replies": replies, "speech": speech}


def pair_turns(speech, replies):
    """Group speech bursts and reply fragments into turns.

    An utterance is the speech bursts just before a reply, joined while the
    pauses between them are under UTTER_GAP; earlier bursts with no reply (e.g.
    the voice-mode start-up blip at t=0) are dropped as strays. A turn is an
    utterance plus all reply fragments after it and before the next utterance.
    Returns (turns, orphans, silent, strays): orphans are reply fragments before
    any speech, silent are utterances at the end that never got a reply.
    """
    events = sorted([(a, b, "s") for a, b in speech] + [(a, b, "r") for a, b in replies])
    turns, orphans, silent, strays, cur = [], [], [], [], None
    for a, b, kind in events:
        if kind == "s":
            if cur is not None and cur["replies"]:
                turns.append(cur)
                cur = None
            if cur is not None and a - cur["speech"][1] > UTTER_GAP:
                strays.append(tuple(cur["speech"]))
                cur = None
            if cur is None:
                cur = {"speech": [a, b], "replies": []}
            else:
                cur["speech"][1] = b            # same utterance, keep extending
        else:
            if cur is None:
                orphans.append((a, b))
            else:
                cur["replies"].append((a, b))
    if cur is not None:
        (turns if cur["replies"] else silent).append(cur)
    return turns, orphans, silent, strays


def noise_turns(replies):
    """Noise files: speech detection is swamped by background noise; group replies only."""
    merged = []
    for a, b in replies:
        if merged and a - merged[-1]["replies"][-1][1] < NOISE_MERGE:
            merged[-1]["replies"].append((a, b))
        else:
            merged.append({"speech": [np.nan, np.nan], "replies": [(a, b)]})
    return merged


# --------------------------------------------------------------------------- per file
def process_file(path, meta=None, write=True):
    path = Path(path)
    up, dn, info = voice_flow(path)
    det = detect(up, dn, info["duration"])
    meta = meta or {}
    fset = meta.get("set") or (parse_name(path.name) or {}).get("set", "")
    order = meta.get("order", [])
    notes = []
    if not info["anthropic"]:
        notes.append(f"busiest 443 flow is {info['server']}, not Anthropic")
    if fset == "noise":
        turns, orphans, silent, strays = noise_turns(det["replies"]), [], [], []
        status = "noise"
        labels = [""] * len(turns)
        notes.append("noise file: replies only, not labelled per turn")
    else:
        turns, orphans, silent, strays = pair_turns(det["speech"], det["replies"])
        late = [s for s in strays if s[0] > 2.0]
        if late:
            notes.append(f"{len(late)} stray speech burst(s) with no reply at " +
                         ", ".join(f"{a:.1f}s" for a, _ in late))
        if orphans:
            notes.append(f"{len(orphans)} reply fragment(s) before any speech")
        if silent:
            notes.append(f"{len(silent)} utterance(s) with no reply")
        if order and len(turns) == len(order) and info["anthropic"]:
            status, labels = "ok", list(order)
        elif not order:
            status, labels = "unlabelled", [""] * len(turns)
            notes.append("not in manifest")
        else:
            status, labels = "flagged", [""] * len(turns)
            notes.append(f"found {len(turns)} turns, manifest lists {len(order)}")

    rows = []
    for k, (turn, label) in enumerate(zip(turns, labels), start=1):
        s0, s1 = turn["speech"]
        r0, r1 = turn["replies"][0][0], turn["replies"][-1][1]
        start = (s0 if not np.isnan(s0) else r0) - PAD
        nxt = turns[k]["speech"][0] if k < len(turns) else np.nan
        nxt = turns[k]["replies"][0][0] if k < len(turns) and np.isnan(nxt) else nxt
        end = min(r1 + PAD, nxt) if not np.isnan(nxt) else r1 + PAD
        start = max(start, 0.0)
        piece = f"{path.stem}_t{k:02d}"
        row = {
            "piece": piece, "source_file": path.name, "set": fset,
            "rep": meta.get("rep", ""), "sitting": meta.get("sitting", ""),
            "split": meta.get("split", ""), "turn": k, "label": label,
            "status": status, "speech_start": round(s0, 3), "speech_end": round(s1, 3),
            "speech_dur": round(s1 - s0, 3), "reply_start": round(r0, 3),
            "reply_end": round(r1, 3), "reply_dur": round(r1 - r0, 3),
            "latency": round(r0 - s1, 3), "n_reply_fragments": len(turn["replies"]),
            "piece_start": round(start, 3), "piece_end": round(end, 3),
        }
        rows.append(row)
        if write:
            PROCESSED.mkdir(exist_ok=True)
            su = up[(up[:, 0] >= start) & (up[:, 0] < end)]
            sd = dn[(dn[:, 0] >= start) & (dn[:, 0] < end)]
            np.savez_compressed(PROCESSED / f"{piece}.npz", up=su, dn=sd,
                                replies=np.array(turn["replies"]),
                                speech=np.array([s0, s1]),
                                meta=np.array([json.dumps(row, default=float)]))
    file_row = {
        "file": path.name, "set": fset, "rep": meta.get("rep", ""),
        "sitting": meta.get("sitting", ""), "split": meta.get("split", ""),
        "status": status, "expected_turns": len(order), "turns": len(turns),
        "speech_bursts": len(det["speech"]), "reply_fragments": len(det["replies"]),
        "reply_thr_pps": round(det["reply_thr"], 1), "speech_thr_B": round(det["speech_thr"], 1),
        "server": info["server"], "anthropic": info["anthropic"],
        "flow_share": round(info["flow_share"], 3), "duration_s": round(info["duration"], 1),
        "notes": "; ".join(notes),
    }
    return rows, file_row, (up, dn, info, det, turns)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="captures (default: captures/*.pcapng)")
    args = ap.parse_args()
    manifest = load_manifest()
    files = [Path(f) for f in args.files] or sorted(CAPTURES.glob("*.pcapng"))
    single = bool(args.files)
    if not single and PROCESSED.exists():
        for old in PROCESSED.glob("*.npz"):
            old.unlink()
    all_rows, file_rows = [], []
    for f in files:
        rows, frow, _ = process_file(f, manifest.get(f.name), write=not single)
        all_rows += rows
        file_rows.append(frow)
        mark = {"ok": " ", "noise": "~", "flagged": "!", "unlabelled": "?"}[frow["status"]]
        print(f"{mark} {f.name:30s} {frow['status']:8s} turns {frow['turns']}/{frow['expected_turns']}  "
              f"speech {frow['speech_bursts']:2d}  reply frags {frow['reply_fragments']:2d}  "
              f"thr {frow['reply_thr_pps']:.0f} pkt/s, {frow['speech_thr_B']:.0f} B  {frow['notes']}")
        if single:
            for r in rows:
                print(f"    t{r['turn']:02d} {r['label'] or '-':4s} speech {r['speech_start']:7.2f}-{r['speech_end']:7.2f}"
                      f"  reply {r['reply_start']:7.2f}-{r['reply_end']:7.2f} ({r['reply_dur']:.1f} s, "
                      f"{r['n_reply_fragments']} frag)  latency {r['latency']:.2f} s")
    if single:
        return
    PROCESSED.mkdir(exist_ok=True)
    write_csv(PROCESSED / "index.csv", all_rows)
    write_csv(PROCESSED / "files.csv", file_rows)
    c = collections.Counter(r["status"] for r in file_rows)
    print(f"\n{len(file_rows)} files: " + ", ".join(f"{n} {s}" for s, n in sorted(c.items())))
    print(f"{len(all_rows)} turn pieces -> {PROCESSED / 'index.csv'}")
    if c.get("flagged"):
        print("Flagged files are kept out of training. Check them with plot_packets.py FILE.")


if __name__ == "__main__":
    sys.exit(main())
