"""Experiment 3 analysis: does Claude's packet timing reveal a web search?

    python analyze.py                  analyze sittings 1-2 (sitting 3 is held out)
    python analyze.py --freeze         same, then lock in the best rule (results/frozen_rule.json)
    python analyze.py --include-test   score the frozen rule ONCE on sitting 3 (the honest number)

Uses packet metadata only (time, size, direction) on Claude's voice connection:
the TCP WebSocket to Anthropic that carries the audio both ways.

For each capture it finds:
  speech      when you were talking: upstream audio frames are ~145 bytes during
              silence and grow to ~180+ while you speak (variable-bitrate audio)
  reply       Claude's spoken answer: downstream bursts far above the idle rate
and measures:
  latency     end of your speech -> first reply audio
  early_kb    reply audio received in the first 1.5 s. Static answers stream at
              full speed; a search reply is a short spoken preamble ("let me
              check...") and then a stall while results come back, so less arrives
  dip_s       longest stall (under 1.5 KB per 0.1 s) inside the first 6 s of the reply
  answer_lat  end of your speech -> start of the largest burst (the actual answer)
  bursts      how many separate reply bursts at 0.25 s resolution
  reply_kb    total reply audio, as a check that reply lengths are comparable

Writes results/exp3_features.csv and results/exp3_report.html, and prints a summary.
"""
import argparse
import csv
import html
import json
import statistics
from collections import Counter
from pathlib import Path

import common
import first_check
import pcaplib
import plan_sitting

BIN = 0.25            # seconds per downstream bin
REPLY_BIN_BYTES = 4000  # downstream bytes per bin that counts as reply audio (idle is ~500)
MERGE_GAP = 0.5       # reply bins closer than this belong to the same burst
ACK_MAX = 100         # upstream packets this small are TCP ACKs, not audio
CTRL_MIN = 1000       # upstream packets this big are control messages, not audio
SMOOTH = 0.3          # seconds of audio frames averaged when detecting speech
SPEECH_DELTA = 18     # bytes above the silence level that count as speech
FINE = 0.1            # seconds per bin when looking for stalls inside a reply
STALL_BYTES = 1500    # a 0.1 s bin below this is a stall in reply audio
EARLY_S = 1.5         # window for early_kb
DIP_WINDOW = 6.0      # look for stalls in the first 6 s of the reply


def voice_flow(pkts):
    """(server, client port) of the largest TCP flow to Claude, or None."""
    sizes = Counter()
    for p in pkts:
        if p.proto != "TCP" or 443 not in (p.sport, p.dport):
            continue
        up = p.dport == 443
        srv = p.dst if up else p.src
        if first_check.is_claude(srv):
            sizes[(srv, p.sport if up else p.dport)] += p.length
    return sizes.most_common(1)[0][0] if sizes else None


def find_speech(frames, start):
    """Speech start/end from audio frame sizes. frames = [(t, size)], sorted."""
    pre = [s for t, s in frames if start + 0.5 <= t <= start + 2.0]
    if len(pre) < 20:
        return None, None, None
    silence = statistics.median(pre)
    thresh = silence + SPEECH_DELTA
    times = [t for t, _ in frames]
    sizes = [s for _, s in frames]
    above = []
    j = 0
    total = 0.0
    for i in range(len(frames)):          # trailing moving average over SMOOTH seconds
        total += sizes[i]
        while times[j] < times[i] - SMOOTH:
            total -= sizes[j]
            j += 1
        if times[i] - start > 1.0 and total / (i - j + 1) > thresh:
            above.append(times[i])
    if not above:
        return None, None, silence
    return above, thresh, silence


def bursts_from(down, t_from):
    """Group downstream reply-audio bins after t_from into bursts [(start, end, bytes)]."""
    bins = Counter()
    for t, size in down:
        if t >= t_from:
            bins[int(t / BIN)] += size
    hot = sorted(b for b, v in bins.items() if v >= REPLY_BIN_BYTES)
    bursts = []
    for b in hot:
        s, e = b * BIN, (b + 1) * BIN
        if bursts and s - bursts[-1][1] < MERGE_GAP:
            bursts[-1][1] = e
            bursts[-1][2] += bins[b]
        else:
            bursts.append([s, e, bins[b]])
    return bursts


def analyze_file(path):
    pkts = pcaplib.read_packets(path)
    if not pkts:
        return {"error": "empty file"}
    flow = voice_flow(pkts)
    if flow is None:
        return {"error": "no Claude voice connection"}
    srv, port = flow
    t0 = pkts[0].ts
    up = [(p.ts - t0, p.length) for p in pkts
          if p.proto == "TCP" and p.dst == srv and p.sport == port]
    down = [(p.ts - t0, p.length) for p in pkts
            if p.proto == "TCP" and p.src == srv and p.dport == port]
    frames = [(t, s) for t, s in up if ACK_MAX < s < CTRL_MIN]
    if len(frames) < 50:
        return {"error": "too few audio frames"}
    start = frames[0][0]

    # First reply burst must come after you start speaking, so look for speech first,
    # then cut speech off at the first reply burst (later "speech" is reply echo / noise).
    above, thresh, silence = find_speech(frames, start)
    if not above:
        return {"error": "no speech detected", "silence": silence}
    speech_start = above[0]
    bursts = bursts_from(down, speech_start)
    if not bursts:
        return {"error": "no reply audio found", "speech_start": speech_start}
    first = bursts[0][0]
    in_speech = [t for t in above if t < first]
    if not in_speech:
        return {"error": "reply started before speech was detected"}
    speech_end = in_speech[-1]

    largest = max(bursts, key=lambda b: b[2])
    gaps = [b2[0] - b1[1] for b1, b2 in zip(bursts, bursts[1:])]

    # Fine-grained look at the start of the reply.
    fine = Counter()
    for t, s in down:
        if first <= t < first + DIP_WINDOW:
            fine[int((t - first) / FINE)] += s
    early = sum(v for k, v in fine.items() if k < EARLY_S / FINE)
    dip = run = 0
    for i in range(int(DIP_WINDOW / FINE)):
        if fine.get(i, 0) < STALL_BYTES:
            run += 1
        else:
            dip = max(dip, run)          # only count stalls that audio resumes after
            run = 0
    # Downstream timeline for plotting (KB per bin, relative to speech end).
    tl = Counter()
    for t, s in down:
        tl[round((int(t / BIN) * BIN) - speech_end, 2)] += s
    up_tl = Counter()
    for t, s in frames:
        up_tl[round((int(t / BIN) * BIN) - speech_end, 2)] += s
    return {
        "error": "",
        "speech_start": round(speech_start, 2),
        "speech_end": round(speech_end, 2),
        "speech_s": round(speech_end - speech_start, 2),
        "latency_s": round(first - speech_end, 2),
        "answer_latency_s": round(largest[0] - speech_end, 2),
        "early_kb": round(early / 1024, 1),
        "dip_s": round(dip * FINE, 1),
        "bursts": len(bursts),
        "max_gap_s": round(max(gaps), 2) if gaps else 0.0,
        "reply_s": round(bursts[-1][1] - first, 2),
        "reply_kb": round(sum(b[2] for b in bursts) / 1024, 1),
        "silence_frame_b": round(silence, 1),
        "_down": sorted(tl.items()),
        "_up": sorted(up_tl.items()),
    }


# ---------- summary + report ----------

def summarize(rows, key):
    out = {}
    for cls in ("search", "static", "off", "base"):
        vals = [r[key] for r in rows if r["class"] == cls and r["error"] == "" and r[key] != ""]
        if vals:
            out[cls] = (len(vals), statistics.median(vals), min(vals), max(vals))
    return out


def best_threshold(rows, key, below=False):
    """Best 1-feature rule 'searched if key > t' (or '< t' when below=True).

    Returns (t, accuracy, n, true positives, positives, false positives, negatives)."""
    pts = [(r[key], r["search_observed"] == "Y") for r in rows
           if r["error"] == "" and r["class"] in ("search", "static", "off")]
    if not pts:
        return None
    vals = sorted({v for v, _ in pts})
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or vals
    best = None
    for t in cands:
        pred = [(v < t) if below else (v > t) for v, _ in pts]
        acc = sum(p == y for p, (_, y) in zip(pred, pts)) / len(pts)
        if best is None or acc > best[1]:
            tp = sum(p and y for p, (_, y) in zip(pred, pts))
            fp = sum(p and not y for p, (_, y) in zip(pred, pts))
            pos = sum(y for _, y in pts)
            best = (round(t, 2), acc, len(pts), tp, pos, fp, len(pts) - pos)
    return best


RULES = [  # (feature, below?, label)
    ("early_kb", True, "reply audio in first 1.5 s (KB)"),
    ("dip_s", False, "longest stall in reply (s)"),
    ("answer_latency_s", False, "speech end -> main answer (s)"),
    ("latency_s", False, "speech end -> first audio (s)"),
]


REPORT = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 3 Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e3e3e3;--search:#c2410c;--static:#2563eb;--base:#6b7280;--card:#f7f7f5}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#9a9a9a;--line:#333;--card:#1f1f1f;--search:#fb923c;--static:#60a5fa;--base:#9ca3af}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:1100px;margin:auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}p.m{color:var(--muted);margin:0 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border-radius:10px;padding:12px 14px}.card b{font-size:22px;display:block}
.box{position:relative;height:340px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:5px 6px;text-align:right}th:first-child,td:first-child{text-align:left}
.wrap{overflow-x:auto}.s{color:var(--search)}.t{color:var(--static)}
</style></head><body>
<h1>Experiment 3: web search side channel</h1>
<p class="m">__SUBTITLE__</p>
<div class="cards">__CARDS__</div>
<h2>How each reply starts</h2>
<p class="m">Each dot is one capture. Static answers stream at full speed from the first second
(right, near zero stall). A search reply opens with a short spoken preamble, then stalls while
results come back, so less audio arrives early and there is a visible pause (left, higher).</p>
<div class="box"><canvas id="lat"></canvas></div>
<h2>Downstream traffic, all captures overlaid</h2>
<p class="m">Bytes per 0.25 s from Claude, aligned so 0 s is the moment you stopped speaking.</p>
<div class="box"><canvas id="tl"></canvas></div>
<h2>Per-capture features</h2>
<div class="wrap"><table>__TABLE__</table></div>
<script>
const D=__DATA__;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const col={search:css('--search'),static:css('--static'),base:css('--base'),off:css('--base')};
const grid={color:css('--line')},tick={color:css('--muted')};
const order=['static','search','off'].filter(c=>D.rows.some(r=>r.class===c));
new Chart(document.getElementById('lat'),{type:'scatter',data:{datasets:order.map(c=>({label:c,
 data:D.rows.filter(r=>r.class===c&&!r.error).map(r=>({x:r.early_kb,y:r.dip_s+(Math.random()-.5)*.03,f:r.file,d:r.dip_s})),
 backgroundColor:col[c],pointRadius:5}))},options:{maintainAspectRatio:false,
 scales:{x:{title:{display:true,text:'reply audio in first 1.5 s (KB)',color:css('--muted')},ticks:tick,grid},
 y:{title:{display:true,text:'longest stall in reply (s)',color:css('--muted')},ticks:tick,grid,beginAtZero:true}},
 plugins:{legend:{labels:{color:css('--fg')}},tooltip:{callbacks:{label:c=>c.raw.f+': '+c.raw.x+' KB, stall '+c.raw.d+' s'}}}}});
new Chart(document.getElementById('tl'),{type:'line',data:{datasets:D.rows.filter(r=>!r.error&&r.class!=='base').map(r=>({
 label:r.class,data:r._down.filter(p=>p[0]>=-2&&p[0]<=14).map(p=>({x:p[0],y:p[1]/1024})),borderColor:col[r.class]+'99',
 borderWidth:1.2,pointRadius:0,tension:.2}))},options:{maintainAspectRatio:false,parsing:true,
 scales:{x:{type:'linear',min:-2,max:14,title:{display:true,text:'seconds after you stopped speaking',color:css('--muted')},ticks:tick,grid},
 y:{title:{display:true,text:'KB per 0.25 s',color:css('--muted')},ticks:tick,grid}},
 plugins:{legend:{display:false},tooltip:{enabled:false}}}});
</script></body></html>"""


def write_report(rows, path, subtitle, cards):
    cols = ["file", "class", "search_observed", "latency_s", "early_kb", "dip_s",
            "answer_latency_s", "bursts", "reply_s", "reply_kb", "error"]
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for r in sorted(rows, key=lambda r: (r["class"], r["file"])):
        cls = {"search": "s", "static": "t"}.get(r["class"], "")
        body += f'<tr class="{cls}">' + "".join(
            f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
    data = {"rows": [{k: r.get(k) for k in
                      ("file", "class", "early_kb", "dip_s", "error", "_down")} for r in rows]}
    out = (REPORT.replace("__SUBTITLE__", html.escape(subtitle))
           .replace("__CARDS__", cards)
           .replace("__TABLE__", head + body)
           .replace("__DATA__", json.dumps(data)))
    path.write_text(out, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true",
                    help="save the best rule from sittings 1-2 to results/frozen_rule.json")
    ap.add_argument("--include-test", action="store_true",
                    help="also analyze sitting 3 and score the frozen rule on it")
    args = ap.parse_args()
    rule_path = common.ROOT / "results" / "frozen_rule.json"
    if args.include_test and not rule_path.exists():
        raise SystemExit("Freeze a rule first:  python analyze.py --freeze  (uses sittings 1-2 only)")
    if args.include_test and args.freeze:
        raise SystemExit("Freeze on sittings 1-2 first, then score with --include-test in a separate run.")

    cfg = common.load_config(require=False)
    cap_dir = common.capture_dir(cfg)
    manifest = {r["file"]: r for r in common.read_manifest(common.manifest_path(cfg))}

    held_out = set()
    for spk in {r["speaker"].lstrip("S") for r in manifest.values()}:
        p = plan_sitting.plan_path(3, spk)
        if p.exists():
            held_out |= {it["file"] for it in plan_sitting.read_plan(p)}
    held_out |= {f for f, r in manifest.items() if "sitting 3" in r.get("notes", "")}

    rows, skipped = [], 0
    for name, m in sorted(manifest.items()):
        if name in held_out and not args.include_test:
            skipped += 1
            continue
        path = cap_dir / name
        if not path.exists():
            continue
        feats = analyze_file(path)
        rows.append({"file": name, "class": m["class"], "prompt_id": m["prompt_id"],
                     "rep": m["rep"], "search_observed": m["search_observed"],
                     "reply_words": m["reply_words"], "test": name in held_out, **feats})

    out_dir = common.ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    fields = ["file", "class", "prompt_id", "rep", "search_observed", "reply_words",
              "speech_start", "speech_end", "speech_s", "latency_s", "early_kb", "dip_s",
              "answer_latency_s", "bursts", "max_gap_s", "reply_s", "reply_kb",
              "silence_frame_b", "error"]
    with open(out_dir / "exp3_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    all_ok = [r for r in rows if r["error"] == ""]
    ok = [r for r in all_ok if not r["test"]]          # rules are only ever tuned on these
    test_ok = [r for r in all_ok if r["test"]]
    print(f"Analyzed {len(rows)} captures ({len(all_ok)} usable)"
          + (f"; {skipped} held-out sitting 3 files skipped" if skipped else ""))
    for r in rows:
        if r["error"]:
            print(f"  could not measure {r['file']}: {r['error']}")

    print("\nMedian (min-max) by class:")
    labels = {"early_kb": "audio in first 1.5 s (KB)", "dip_s": "longest stall (s)",
              "answer_latency_s": "speech end -> answer (s)",
              "latency_s": "speech end -> 1st audio (s)", "reply_kb": "total reply audio (KB)"}
    for key, label in labels.items():
        parts = [f"{c} {med:g} ({lo:g}-{hi:g})" for c, (n, med, lo, hi) in summarize(ok, key).items()]
        print(f"  {label:<28} " + "   ".join(parts))

    print("\nBest one-feature rule per feature (labels = search_observed):")
    cards = ""
    for key, below, label in RULES:
        res = best_threshold(ok, key, below)
        if not res:
            continue
        t, acc, n, tp, pos, fp, neg = res
        op = "<" if below else ">"
        print(f"  searched if {label} {op} {t:<6g} accuracy {acc:.0%}  "
              f"(caught {tp}/{pos} searches, {fp}/{neg} false alarms)")
        cards += (f'<div class="card">searched if {html.escape(label)} {html.escape(op)} {t:g}'
                  f'<b>{acc:.0%}</b>caught {tp}/{pos} searches, {fp}/{neg} false alarms</div>')
    print("  Thresholds are tuned on this same data (sittings 1-2), so these are optimistic.")

    if args.freeze:
        key, below, label = RULES[0]
        t, acc, n, tp, pos, fp, neg = best_threshold(ok, key, below)
        rule_path.parent.mkdir(exist_ok=True)
        rule_path.write_text(json.dumps({"feature": key, "below": below, "label": label,
                                         "threshold": t, "train_accuracy": acc,
                                         "train_captures": n}, indent=2), encoding="utf-8")
        print(f"\nFROZEN: searched if {label} {'<' if below else '>'} {t}  "
              f"(train accuracy {acc:.0%} on {n}). Saved {rule_path.name}.")
        print("Now record sitting 3, then run:  python analyze.py --include-test")

    if args.include_test:
        rule = json.loads(rule_path.read_text(encoding="utf-8"))
        pts = [r for r in test_ok if r["class"] in ("search", "static", "off")]
        key, t = rule["feature"], rule["threshold"]
        pred = [(r[key] < t) if rule["below"] else (r[key] > t) for r in pts]
        truth = [r["search_observed"] == "Y" for r in pts]
        tp = sum(p and y for p, y in zip(pred, truth))
        fp = sum(p and not y for p, y in zip(pred, truth))
        pos, neg = sum(truth), len(truth) - sum(truth)
        acc = sum(p == y for p, y in zip(pred, truth)) / len(pts) if pts else 0
        op = "<" if rule["below"] else ">"
        print(f"\nHELD-OUT TEST (sitting 3, rule frozen before it was looked at):")
        print(f"  searched if {rule['label']} {op} {t}:  accuracy {acc:.0%} on {len(pts)} captures "
              f"(caught {tp}/{pos} searches, {fp}/{neg} false alarms)")
        cards = (f'<div class="card"><i>Held-out test, sitting 3</i><br>searched if '
                 f'{html.escape(rule["label"])} {html.escape(op)} {t}<b>{acc:.0%}</b>'
                 f'caught {tp}/{pos} searches, {fp}/{neg} false alarms</div>') + cards
        ok = all_ok      # the charts and table show every sitting

    subtitle = (f"{len(ok)} captures from sittings "
                + ("1-3" if args.include_test else "1-2 (sitting 3 held out)")
                + ". Packet metadata only: timing, size and direction on Claude's voice connection.")
    write_report(rows, out_dir / "exp3_report.html", subtitle, cards)
    print(f"\nWrote {out_dir / 'exp3_features.csv'}")
    print(f"Wrote {out_dir / 'exp3_report.html'} (open it in a browser)")


if __name__ == "__main__":
    main()
