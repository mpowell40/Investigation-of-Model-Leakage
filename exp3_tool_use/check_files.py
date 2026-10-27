"""Nightly check for Experiment 3 (the protocol's manifest-owner check).

    python check_files.py

  - every capture file has a manifest row, and every row has a file
  - file names match the pattern exactly, and agree with their manifest row
  - files under 50 KB or 10 s are flagged for re-recording
  - progress against the 112-capture target, and how often search appeared
"""
import argparse
import sys
from collections import Counter

import common
import pcaplib
import plan_sitting

TARGET = ({"search": 32, "static": 32, "off": 32, "base": 16} if plan_sitting.SET_B
          else {"search": 32, "static": 32, "base": 16})


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = common.load_config()
    cap_dir = common.capture_dir(cfg)
    rows = common.read_manifest(common.manifest_path(cfg))
    files = {p.name: p for p in cap_dir.glob("*.pcapng")}
    problems = []

    by_file = Counter(r["file"] for r in rows)
    for name, n in by_file.items():
        if n > 1:
            problems.append(f"{name}: {n} manifest rows (should be 1)")
    for name in sorted(set(files) - set(by_file)):
        problems.append(f"{name}: file has no manifest row")
    for name in sorted(set(by_file) - set(files)):
        problems.append(f"{name}: manifest row has no file")

    for r in rows:
        name = r["file"]
        parsed = common.parse_name(name)
        if parsed is None:
            problems.append(f"{name}: name doesn't match the pattern")
            continue
        expect = {
            "experiment": parsed["experiment"], "class": parsed["class"],
            "prompt_id": parsed["prompt_id"], "speaker": f"S{parsed['speaker']}",
            "rep": str(parsed["rep"]),
        }
        for col, val in expect.items():
            if r.get(col, "").strip() != val:
                problems.append(f"{name}: manifest {col}={r.get(col)!r}, name says {val!r}")
        want_setting = "off" if parsed["class"] == "off" else "on"
        if r.get("search_setting") != want_setting:
            problems.append(f"{name}: search_setting should be {want_setting}")
        if r.get("search_observed") not in ("Y", "N"):
            problems.append(f"{name}: search_observed must be Y or N")
        if not r.get("reply_words", "").isdigit():
            problems.append(f"{name}: reply_words should be a number")

    for name, p in sorted(files.items()):
        if common.parse_name(name) is None and name not in by_file:
            problems.append(f"{name}: name doesn't match the pattern")
        size = p.stat().st_size
        try:
            _, dur = pcaplib.capture_stats(p)
        except Exception:
            dur = 0.0
        if size < common.MIN_BYTES or dur < common.MIN_SECONDS:
            problems.append(f"{name}: {size / 1024:.0f} KB, {dur:.0f} s; probably broken, re-record")

    print(f"Experiment 3 nightly check: {len(files)} files, {len(rows)} manifest rows\n")
    counts = Counter(common.parse_name(n)["class"] for n in files if common.parse_name(n))
    print("Progress:")
    for cls, target in TARGET.items():
        print(f"  {cls:<7} {counts.get(cls, 0):>3} / {target}")
    print(f"  total   {sum(counts.values()):>3} / {sum(TARGET.values())}")

    obs = Counter((r.get("class"), r.get("search_observed")) for r in rows)
    print("\nSearch indicator seen (labels follow what happened, not what was expected):")
    for cls in ("search", "static", "off", "base"):
        y, n = obs.get((cls, "Y"), 0), obs.get((cls, "N"), 0)
        if y + n:
            print(f"  {cls:<7} Y={y:<3} N={n:<3} ({y / (y + n):.0%} searched)")

    held_out = []
    p3 = plan_sitting.plan_path(3, cfg["speaker"])
    if p3.exists():
        held_out = [it["file"] for it in plan_sitting.read_plan(p3) if it["status"] == "done"]
    print(f"\nHeld-out test set (sitting 3): {len(held_out)} files. Don't train or tune on these.")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("No problems found.")


if __name__ == "__main__":
    main()
