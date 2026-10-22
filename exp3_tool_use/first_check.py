"""First-file check: the protocol's 30-second sanity check on a capture.

    python first_check.py captures/base_s1_pB03_r01.pcapng

Mirrors the two Wireshark checks in the protocol:
  1. Statistics > Conversations: is there a large port-443 flow, and is it on
     the same protocol (TCP or UDP) as in the pilot?
  2. Statistics > I/O Graphs split by direction: upstream rises while you
     speak, downstream bursts while Claude replies.
"""
import argparse
import sys
from collections import Counter, defaultdict

import common
import pcaplib


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def summarize(path):
    pkts = [p for p in pcaplib.read_packets(path) if p.proto]
    flows = defaultdict(lambda: {"bytes": 0, "pkts": 0, "first": None, "last": None})
    for p in pkts:
        if 443 not in (p.sport, p.dport):
            continue
        # Client side is the end that isn't port 443.
        client = (p.src, p.sport) if p.dport == 443 else (p.dst, p.dport)
        server = (p.dst, p.dport) if p.dport == 443 else (p.src, p.sport)
        f = flows[(p.proto, client, server)]
        f["bytes"] += p.length
        f["pkts"] += 1
        f["first"] = p.ts if f["first"] is None else f["first"]
        f["last"] = p.ts
    if not flows:
        return {"ok": False, "packets": len(pkts), "flows": 0}

    key, top = max(flows.items(), key=lambda kv: kv[1]["bytes"])
    proto, client, server = key
    t0 = min(p.ts for p in pkts)
    up, down = Counter(), Counter()
    for p in pkts:
        if p.proto != proto:
            continue
        if (p.src, p.sport) == client and (p.dst, p.dport) == server:
            up[int(p.ts - t0)] += p.length
        elif (p.src, p.sport) == server and (p.dst, p.dport) == client:
            down[int(p.ts - t0)] += p.length
    total = sum(f["bytes"] for f in flows.values())
    return {
        "ok": True, "packets": len(pkts), "flows": len(flows),
        "proto": proto, "client": client, "server": server,
        "top_bytes": top["bytes"], "top_pkts": top["pkts"],
        "top_share": top["bytes"] / total if total else 0,
        "duration": (top["last"] - top["first"]) if top["pkts"] > 1 else 0,
        "up": up, "down": down,
        "seconds": int(max(p.ts for p in pkts) - t0) + 1,
        "others": sorted(((k, v["bytes"]) for k, v in flows.items() if k != key),
                         key=lambda kv: -kv[1])[:4],
    }


def bar(value, peak, width=24):
    n = 0 if not peak else round(width * value / peak)
    return ("#" * n).ljust(width)


def print_report(path, pilot_protocol=""):
    s = summarize(path)
    print(f"\nFirst-file check: {path}")
    if not s["ok"]:
        print("  FAIL: no port-443 traffic in this file. Wrong interface? Pick the active Wi-Fi adapter.")
        return False

    print(f"\n1) Conversations ({s['flows']} port-443 flows)")
    print(f"   Largest: {s['proto']}  {s['client'][0]}:{s['client'][1]} <-> "
          f"{s['server'][0]}:{s['server'][1]}")
    print(f"   {human(s['top_bytes'])}, {s['top_pkts']} packets, {s['duration']:.1f} s, "
          f"{s['top_share']:.0%} of all port-443 bytes")
    for (proto, c, srv), b in s["others"]:
        print(f"   other: {proto} {srv[0]}:{srv[1]}  {human(b)}")

    ok = True
    pilot = (pilot_protocol or "").upper()
    if pilot in ("TCP", "UDP"):
        if s["proto"] == pilot:
            print(f"   OK: same protocol as the pilot ({pilot}).")
        else:
            print(f"   PROBLEM: pilot used {pilot}, this flow is {s['proto']}. Stop and tell the team.")
            ok = False
    else:
        print("   (No pilot protocol saved, so compare TCP/UDP with the pilot yourself.)")

    print("\n2) Bytes per second on that flow, split by direction")
    print("    t   upstream (you -> Claude)          downstream (Claude -> you)")
    up_peak, down_peak = max(s["up"].values(), default=0), max(s["down"].values(), default=0)
    for t in range(s["seconds"]):
        u, d = s["up"].get(t, 0), s["down"].get(t, 0)
        print(f"  {t:>3}  {bar(u, up_peak)} {human(u):>9}  {bar(d, down_peak)} {human(d):>9}")
    print("\n  Look for: upstream rising while you spoke (after the 3 s of silence),")
    print("  then a downstream burst while Claude replied. If not, stop and tell the team.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    args = ap.parse_args()
    cfg = common.load_config(require=False)
    sys.exit(0 if print_report(args.file, cfg.get("pilot_protocol", "")) else 1)


if __name__ == "__main__":
    main()
