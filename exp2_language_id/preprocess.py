"""Split every capture into one packet file per question, ready for training.

    python preprocess.py                    sessions/ and captures/ -> processed/
    python preprocess.py --data DIR ...     other folders of lang_*.pcapng files

A playlist session (lang_<lang>_s<N>_pALL_r<NN>.pcapng) holds all 10 questions,
asked in order L01..L10. Claude's replies show up as bursts of download packets,
so the session is cut after each reply into 10 contiguous pieces (no packets
dropped). A session is only split when exactly 10 replies are found; otherwise
it's skipped with a message, since the question IDs couldn't be trusted.
A single-question capture (..._pL07_r01.pcapng) is kept whole as one piece.

Output, metadata only (time, size, direction; never payload):
  processed/questions/<lang>/lang_<lang>_s<N>_p<LNN>_r<NN>.csv
      t          seconds since the start of this piece
      size       bytes on the wire
      direction  up (your mic -> Claude) or down (Claude -> you)
  processed/index.csv   one row per piece: labels, where it came from, reply timing
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exp3_tool_use"))
import pcaplib  # noqa: E402

from questions import LANGS, QUESTIONS  # noqa: E402

OUT_DIR = ROOT / "processed"
BIN = 0.5            # seconds per packet-rate bin
MIN_REPLY = 2.0      # shorter download bursts are noise
MERGE_GAP = 2.0      # bursts closer than this are one reply
TAIL = 1.0           # seconds kept after a reply ends before the next piece starts

NAME_RE = re.compile(r"^lang_(?P<lang>[a-z]{2})_s(?P<spk>\d+)_p(?P<pid>L\d\d|ALL)_r(?P<rep>\d\d)\.pcapng?$")
INDEX_COLS = ["file", "lang", "speaker", "prompt_id", "rep", "source", "src_start", "src_end",
              "duration", "reply_start", "reply_end", "reply_dur", "packets",
              "up_packets", "down_packets", "up_bytes", "down_bytes"]


def load_flow(path):
    """Packets of the busiest port-443 conversation (the voice session).

    Returns a list of (t, size, direction) with t starting at 0.
    """
    pkts = [p for p in pcaplib.read_packets(path) if p.proto and 443 in (p.sport, p.dport)]
    if not pkts:
        return []
    flows = defaultdict(int)
    for p in pkts:
        client = (p.src, p.sport) if p.dport == 443 else (p.dst, p.dport)
        server = (p.dst, p.dport) if p.dport == 443 else (p.src, p.sport)
        flows[(p.proto, client, server)] += p.length
    proto, client, server = max(flows, key=flows.get)
    keep = sorted((p for p in pkts if p.proto == proto and
                   {(p.src, p.sport), (p.dst, p.dport)} == {client, server}), key=lambda p: p.ts)
    t0 = keep[0].ts
    return [(p.ts - t0, p.length, "up" if (p.src, p.sport) == client else "down") for p in keep]


def _percentile(values, q):
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def find_replies(packets):
    """(start, end) seconds of each Claude reply.

    Download packet rate, in BIN-wide bins smoothed over 1 s, sits at ~25-45
    pkt/s when idle and ~100+ while Claude speaks. Anything above the midpoint
    of the quiet (10th pct) and busy (90th pct) levels counts as a reply.
    """
    if not packets:
        return []
    n = int(packets[-1][0] / BIN) + 1
    rate = [0.0] * n
    for t, _, d in packets:
        if d == "down":
            rate[int(t / BIN)] += 1 / BIN
    smooth = [(rate[i] + rate[i - 1]) / 2 if i else rate[i] for i in range(n)]
    thr = (_percentile(smooth, 10) + _percentile(smooth, 90)) / 2

    runs, start = [], None
    for i, v in enumerate(smooth + [0.0]):
        if v > thr and start is None:
            start = i
        elif v <= thr and start is not None:
            if runs and (start - runs[-1][1]) * BIN < MERGE_GAP:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((start, i))
            start = None
    return [(a * BIN, b * BIN) for a, b in runs if (b - a) * BIN >= MIN_REPLY]


def write_piece(out, packets):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "size", "direction"])
        for t, size, d in packets:
            w.writerow([f"{t:.6f}", size, d])


def piece_row(name, meta, pid, source, start, end, reply, packets):
    up = [s for _, s, d in packets if d == "up"]
    down = [s for _, s, d in packets if d == "down"]
    return {
        "file": name, "lang": meta["lang"], "speaker": f"S{meta['spk']}", "prompt_id": pid,
        "rep": int(meta["rep"]), "source": source,
        "src_start": round(start, 2), "src_end": round(end, 2), "duration": round(end - start, 2),
        "reply_start": round(reply[0] - start, 2) if reply else "",
        "reply_end": round(reply[1] - start, 2) if reply else "",
        "reply_dur": round(reply[1] - reply[0], 2) if reply else "",
        "packets": len(packets), "up_packets": len(up), "down_packets": len(down),
        "up_bytes": sum(up), "down_bytes": sum(down),
    }


def process(path):
    """Write the pieces for one capture. Returns index rows (empty if skipped)."""
    meta = NAME_RE.match(path.name).groupdict()
    packets = load_flow(path)
    if not packets:
        print(f"  {path.name}: SKIPPED, no port-443 traffic")
        return []
    replies = find_replies(packets)
    end_of_capture = packets[-1][0] + 1e-6

    if meta["pid"] == "ALL":
        if len(replies) != len(QUESTIONS):
            print(f"  {path.name}: SKIPPED, found {len(replies)} replies, expected {len(QUESTIONS)}. "
                  f"Check the capture (e.g. a question Claude didn't answer).")
            return []
        cuts = [0.0]
        for i, (_, r_end) in enumerate(replies):
            nxt = replies[i + 1][0] if i + 1 < len(replies) else end_of_capture
            cuts.append(min(r_end + TAIL, nxt))
        cuts[-1] = end_of_capture
        spans = [(pid, cuts[i], cuts[i + 1], replies[i]) for i, pid in enumerate(QUESTIONS)]
    else:
        if len(replies) != 1:
            print(f"  {path.name}: note, found {len(replies)} replies (expected 1); kept whole")
        spans = [(meta["pid"], 0.0, end_of_capture, replies[0] if len(replies) == 1 else None)]

    rows = []
    for pid, start, end, reply in spans:
        piece = [(t - start, s, d) for t, s, d in packets if start <= t < end]
        name = f"lang_{meta['lang']}_s{meta['spk']}_p{pid}_r{meta['rep']}.csv"
        write_piece(OUT_DIR / "questions" / meta["lang"] / name, piece)
        rows.append(piece_row(name, meta, pid, path.name, start, end, reply, piece))
    lengths = " ".join(f"{r['reply_dur']}" for r in rows if r["reply_dur"] != "")
    print(f"  {path.name}: {len(rows)} piece(s); reply lengths (s): {lengths}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "sessions"), str(ROOT / "captures")],
                    help="folders with lang_*.pcapng files")
    args = ap.parse_args()

    paths = []
    for d in args.data:
        if Path(d).is_dir():
            for p in sorted(Path(d).glob("lang_*.pcap*")):
                m = NAME_RE.match(p.name)
                if not m or m["lang"] not in LANGS:
                    print(f"  skipping {p.name}: name isn't lang_<{'|'.join(LANGS)}>_s<N>_p<ALL|LNN>_r<NN>")
                    continue
                paths.append(p)
    if not paths:
        raise SystemExit("No lang_*.pcapng captures found.")

    print(f"Processing {len(paths)} capture(s):")
    rows = []
    for p in paths:
        rows += process(p)

    names = [r["file"] for r in rows]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        print(f"\nWARNING: two captures produced the same piece name (later one wins): {sorted(dupes)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} question pieces written to {OUT_DIR / 'questions'}")
    for lang in LANGS:
        n = sum(r["lang"] == lang for r in rows)
        if n:
            print(f"  {lang}: {n}")
    print(f"Index: {OUT_DIR / 'index.csv'}")


if __name__ == "__main__":
    main()
