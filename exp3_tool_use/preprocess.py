"""Find your speech and Claude's reply in every Experiment 3 capture, ready for training.

    python preprocess.py                  captures/ + manifest_exp3.csv -> processed/
    python preprocess.py --data DIR ...   other folders of tool_*/base_* .pcapng files

Each capture holds one prompt and one reply: ~3 s of silence, you speak, Claude
answers, ~3 s of silence. Only the voice session is kept: the busiest port-443
flow to Anthropic's address ranges (first_check.is_claude), where busiest means
most packets (the voice socket sends ~50 audio frames/s each way). If no flow
goes to those ranges, the busiest port-443 flow is used and the capture is noted.
Why not plain "busiest port-443 flow by bytes": in 12 of the 80 sitting 1-3
captures that is a background Microsoft download, and in one more it is a
QUIC web connection to Anthropic rather than the TCP voice socket.

Detection uses an adaptive threshold on per-0.5 s bins: the midpoint of the
10th and 90th percentiles of the smoothed series, over the live part of the
session.
  reply   download packet rate (idle ~26 pkt/s, Claude speaking ~200 pkt/s).
          The first above-threshold run after your speech, merged with later runs
          less than MERGE_GAP apart (a search reply is preamble, stall, answer).
  speech  upload packet *rate* is flat (one audio frame every 20 ms whether you
          talk or not), so the same rule runs on the mean upload audio-frame size
          instead (~140 B silent, ~170 B speaking), over the bins before the reply.
Both edges are then refined from 0.5 s bins to single packets:
  speech_end   last audio frame where the 0.3 s moving average of frame size is
               still above the speech threshold, before the reply starts
  reply_start  first download packet after speech_end that opens a dense stretch
               (>= REPLY_DENSE_BYTES in the next 0.2 s): the first reply packet
A capture is flagged (status != ok, reply/speech columns blank) when detection
fails, instead of guessing.

Output, metadata only (time, size, direction; never payload):
  processed/<capture>.csv   t (s since the flow's first packet), size (B), direction (up/down)
  processed/index.csv       one row per capture: labels, sitting, detected edges, counts
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import common
import first_check
import pcaplib

ROOT = common.ROOT
OUT_DIR = ROOT / "processed"

BIN = 0.5                 # seconds per rate bin
ACK_MAX = 100             # upload packets this small are TCP ACKs, not audio frames
CTRL_MIN = 1000           # upload packets this big are control messages, not audio frames
SKIP_START = 1.0          # first second of audio frames is a start-up artifact
MERGE_GAP = 2.0           # reply runs closer than this are one reply
MIN_SPEECH_CONTRAST = 8   # B: speech bins must be this much bigger than silence bins
SMOOTH = 0.3              # s, moving average for the fine speech edge
CLUSTER_GAP = 0.15        # s, above-threshold frames closer than this are one cluster
MIN_CLUSTER = 0.35        # s, shorter clusters are blips, not speech
WORD_GAP = 1.0            # s, clusters closer than this are the same utterance
REPLY_DENSE_BYTES = 2000  # download bytes in 0.2 s that mark reply audio (idle ~400)
FINE_WINDOW = 0.2

INDEX_COLS = ["file", "class", "label", "search_observed", "prompt_id", "rep", "speaker",
              "sitting", "date", "status", "flow", "server", "duration",
              "speech_thr", "reply_thr", "speech_start", "speech_end",
              "reply_start", "reply_end", "reply_runs", "reply_dur", "latency",
              "packets", "up_packets", "down_packets", "up_bytes", "down_bytes"]


def load_flow(path):
    """Voice-session packets as [(t, size, direction)] from 0, plus (server, note)."""
    pkts = [p for p in pcaplib.read_packets(path) if p.proto and 443 in (p.sport, p.dport)]
    if not pkts:
        return [], None, "no port-443 traffic"
    flows = defaultdict(int)
    for p in pkts:
        client = (p.src, p.sport) if p.dport == 443 else (p.dst, p.dport)
        server = (p.dst, p.dport) if p.dport == 443 else (p.src, p.sport)
        flows[(p.proto, client, server)] += 1       # busiest = most packets (see docstring)
    claude = {k: v for k, v in flows.items() if first_check.is_claude(k[2][0])}
    note = "" if claude else "no flow to Anthropic; used busiest port-443 flow"
    proto, client, server = max(claude or flows, key=(claude or flows).get)
    keep = sorted((p for p in pkts if p.proto == proto and
                   {(p.src, p.sport), (p.dst, p.dport)} == {client, server}), key=lambda p: p.ts)
    t0 = keep[0].ts
    return ([(p.ts - t0, p.length, "up" if (p.src, p.sport) == client else "down") for p in keep],
            f"{proto} {server[0]}", note)


def _percentile(values, q):
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def adaptive_threshold(values):
    return (_percentile(values, 10) + _percentile(values, 90)) / 2


def _smooth(x):
    return [(x[i] + x[i - 1]) / 2 if i else x[i] for i in range(len(x))]


def _runs(flags):
    """[(first, last+1)] index runs where flags is True."""
    runs, start = [], None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    return runs


def detect(packets):
    """Speech and reply edges of one capture. Returns a dict; status != 'ok' on failure."""
    frames = [(t, s) for t, s, d in packets if d == "up" and ACK_MAX < s < CTRL_MIN]
    if len(frames) < 100:
        return {"status": "too few upload audio frames"}
    live0, live1 = frames[0][0] + SKIP_START, frames[-1][0]
    n = int(live1 / BIN) + 1
    b0 = int(live0 / BIN) + 1          # first full live bin

    # Reply: download packet rate.
    rate = [0.0] * n
    for t, _, d in packets:
        if d == "down" and t < n * BIN:
            rate[int(t / BIN)] += 1 / BIN
    rate = _smooth(rate)
    live = rate[b0:n]
    if len(live) < 8:
        return {"status": "session too short"}
    reply_thr = adaptive_threshold(live)
    runs = [(a + b0, b + b0) for a, b in _runs(v > reply_thr for v in live)]
    if not runs:
        return {"status": "no reply burst", "reply_thr": reply_thr}
    first_run = runs[0]

    # Speech: mean upload frame size per bin, before the reply.
    size_sum, size_n = [0.0] * n, [0] * n
    for t, s in frames:
        if t >= live0:
            size_sum[int(t / BIN)] += s
            size_n[int(t / BIN)] += 1
    mean_size = _smooth([size_sum[i] / size_n[i] if size_n[i] else 0.0 for i in range(n)])
    pre = [mean_size[i] for i in range(b0, first_run[0]) if size_n[i]]
    if len(pre) < 4:
        return {"status": "reply starts before any silence/speech", "reply_thr": reply_thr}
    lo, hi = _percentile(pre, 10), _percentile(pre, 90)
    speech_thr = (lo + hi) / 2
    out = {"reply_thr": round(reply_thr, 1), "speech_thr": round(speech_thr, 1)}
    if hi - lo < MIN_SPEECH_CONTRAST:
        return {**out, "status": f"no speech contrast before reply ({hi - lo:.0f} B)"}
    speech_bins = [i for i in range(b0, first_run[0]) if size_n[i] and mean_size[i] > speech_thr]
    if not speech_bins:
        return {**out, "status": "no speech before reply"}

    # Fine reply start: first download packet after the user's speech bins that
    # opens a dense stretch of download bytes.
    coarse_reply = first_run[0] * BIN
    down = [(t, s) for t, s, d in packets if d == "down"]
    reply_start = None
    k = 0
    for t, s in down:
        if t < max(coarse_reply - 2 * BIN, speech_bins[0] * BIN):
            continue
        if t > first_run[1] * BIN:
            break
        while k < len(down) and down[k][0] < t:
            k += 1
        dense = sum(s2 for t2, s2 in down[k:k + 200] if t2 < t + FINE_WINDOW)
        if dense >= REPLY_DENSE_BYTES:
            reply_start = t
            break
    if reply_start is None:
        return {**out, "status": "could not place the first reply packet"}

    # Fine speech edges: centred SMOOTH-s moving average of frame sizes, above the
    # speech threshold, grouped into clusters. Clusters shorter than MIN_CLUSTER are
    # blips (start-up frames, the reply's onset reaching the mic) and are ignored.
    ts = [t for t, _ in frames]
    ss = [s for _, s in frames]
    above, j, total = [], 0, 0.0
    for i in range(len(frames)):
        total += ss[i]
        while ts[j] < ts[i] - SMOOTH:
            total -= ss[j]
            j += 1
        tc = ts[i] - SMOOTH / 2                      # centre of the trailing window
        if live0 <= tc < reply_start and total / (i - j + 1) > speech_thr:
            above.append(tc)
    clusters = []
    for t in above:
        if clusters and t - clusters[-1][1] <= CLUSTER_GAP:
            clusters[-1][1] = t
        else:
            clusters.append([t, t])
    clusters = [c for c in clusters if c[1] - c[0] >= MIN_CLUSTER]
    if not clusters:
        return {**out, "status": "no speech before reply"}
    speech_end = clusters[-1][1]
    speech_start = clusters[-1][0]
    for a, b in reversed(clusters[:-1]):              # pauses between words
        if speech_start - b > WORD_GAP:
            break
        speech_start = a
    if reply_start - speech_end < 0.05:
        return {**out, "status": "speech runs into the reply (overlap?)"}

    # Reply end: merge runs after the first one when the gap is short.
    merged = [list(first_run)]
    for a, b in runs[1:]:
        if (a - merged[-1][1]) * BIN < MERGE_GAP:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    n_runs = sum(1 for a, b in runs if first_run[0] <= a < merged[0][1])
    reply_end = merged[0][1] * BIN
    # Refine: last dense download packet inside the final bin.
    tail = [t for t, s in down if merged[0][1] * BIN - BIN <= t < reply_end and s > ACK_MAX]
    if tail:
        reply_end = max(tail) + 1e-3
    # Later bursts more than MERGE_GAP after the reply (e.g. end-of-session tones) are ignored.
    return {**out, "status": "ok", "speech_start": speech_start, "speech_end": speech_end,
            "reply_start": reply_start, "reply_end": reply_end, "reply_runs": n_runs}


def write_packets(out, packets):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "size", "direction"])
        for t, size, d in packets:
            w.writerow([f"{t:.6f}", size, d])


def read_packets_csv(path):
    with open(path, newline="") as f:
        return [(float(r["t"]), int(r["size"]), r["direction"]) for r in csv.DictReader(f)]


def sitting_of(manifest_row):
    m = re.match(r"\s*sitting\s+(\d+)", manifest_row.get("notes", ""))
    return int(m[1]) if m else None


def label_of(cls, search_observed):
    """'search' / 'nosearch' from what the transcript showed, '' when not applicable."""
    if cls == "base":
        return ""
    return {"Y": "search", "N": "nosearch"}.get(search_observed.strip().upper(), "")


def process(path, mrow):
    meta = common.parse_name(path.name)
    packets, flow, note = load_flow(path)
    row = {"file": path.name, "class": meta["class"], "prompt_id": meta["prompt_id"],
           "rep": meta["rep"], "speaker": f"S{meta['speaker']}",
           "search_observed": (mrow or {}).get("search_observed", ""),
           "sitting": sitting_of(mrow) if mrow else "",
           "date": (mrow or {}).get("date_time", "")[:10]}
    row["label"] = label_of(row["class"], row["search_observed"])
    if not packets:
        row["status"] = note
        return row, []
    det = detect(packets)
    up = [s for _, s, d in packets if d == "up"]
    down = [s for _, s, d in packets if d == "down"]
    row.update({"flow": "fallback" if note else "anthropic", "server": flow,
                "duration": round(packets[-1][0], 2), "packets": len(packets),
                "up_packets": len(up), "down_packets": len(down),
                "up_bytes": sum(up), "down_bytes": sum(down),
                "speech_thr": det.get("speech_thr", ""), "reply_thr": det.get("reply_thr", "")})
    status = det["status"]
    if note and status == "ok":
        status = "ok (" + note + ")"
    row["status"] = status
    if det["status"] == "ok":
        row.update({k: round(det[k], 3) for k in ("speech_start", "speech_end",
                                                  "reply_start", "reply_end")})
        row["reply_runs"] = det["reply_runs"]
        row["reply_dur"] = round(det["reply_end"] - det["reply_start"], 3)
        row["latency"] = round(det["reply_start"] - det["speech_end"], 3)
    return row, packets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "captures")],
                    help="folders with tool_*/base_* .pcapng files")
    args = ap.parse_args()

    cfg = common.load_config(require=False)
    manifest = {r["file"]: r for r in common.read_manifest(common.manifest_path(cfg))}
    paths = []
    for d in args.data:
        for p in sorted(Path(d).glob("*.pcap*")):
            if common.parse_name(p.name) is None:
                print(f"  skipping {p.name}: name doesn't follow common.FILE_RE")
                continue
            paths.append(p)
    if not paths:
        raise SystemExit("No tool_*/base_* captures found.")

    print(f"Processing {len(paths)} capture(s):")
    rows = []
    for p in paths:
        mrow = manifest.get(p.name)
        if mrow is None:
            print(f"  {p.name}: not in the manifest, no label or sitting")
        row, packets = process(p, mrow)
        if packets:
            write_packets(OUT_DIR / (p.stem + ".csv"), packets)
        rows.append(row)
        if row["status"].startswith("ok"):
            extra = "" if row["status"] == "ok" else f"   [{row['status'][4:-1]}]"
            print(f"  {p.name:<34} speech {row['speech_start']:6.2f}-{row['speech_end']:6.2f}  "
                  f"reply {row['reply_start']:6.2f}-{row['reply_end']:6.2f}  "
                  f"latency {row['latency']:5.2f} s{extra}")
        else:
            print(f"  {p.name:<34} FLAGGED: {row['status']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"].startswith("ok")]
    print(f"\n{len(ok)}/{len(rows)} captures usable; {len(rows) - len(ok)} flagged")
    fallback = [r["file"] for r in rows if r.get("flow") == "fallback"]
    if fallback:
        print(f"  no Anthropic flow, busiest port-443 used: {', '.join(fallback)}")

    # Observed label vs. the class in the file name.
    tool = [r for r in rows if r["class"] in ("search", "static", "off")]
    expect = {"search": "Y", "static": "N", "off": "N"}
    diff = [r for r in tool if r["search_observed"].upper() != expect[r["class"]]]
    print(f"\nObserved label vs file-name class: {len(diff)}/{len(tool)} differ")
    for r in diff:
        print(f"  {r['file']}: class {r['class']}, search_observed {r['search_observed'] or '(blank)'}")
    counts = defaultdict(int)
    for r in rows:
        counts[(r["sitting"], r["class"])] += 1
    print("Captures per sitting:", ", ".join(
        f"s{s} {c} {n}" for (s, c), n in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))))
    print(f"Index: {OUT_DIR / 'index.csv'}")


if __name__ == "__main__":
    main()
