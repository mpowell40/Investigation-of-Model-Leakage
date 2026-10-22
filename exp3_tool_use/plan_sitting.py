"""Make the shuffled run order for one sitting of Experiment 3.

    python plan_sitting.py --sitting 1

Writes plans/sitting<N>_s<speaker>.csv. Each rep is shuffled separately, as the
protocol asks. An existing plan is never re-shuffled unless you pass --force,
so a sitting interrupted halfway resumes in the same order.
"""
import argparse
import csv
import random

import common
from prompts import BASELINE, PAIRS, prompt_text

# Sitting -> blocks, straight from the protocol's "Sittings by experiment" table.
#   base = latency baseline (B01-B04), A = matched pairs, search ON,
#   B = search OFF control (T01-T08 current-info only).
SCHEDULE = {
    1: [("base", 1), ("A", 1), ("A", 2)],
    2: [("base", 2), ("A", 3), ("B", 1), ("B", 2)],
    3: [("base", 3), ("A", 4), ("B", 3), ("B", 4), ("base", 4)],
}
EXPECTED = {1: 36, 2: 36, 3: 40}          # 112 in total
PLAN_COLS = ["seq", "file", "class", "prompt_id", "rep",
             "search_setting", "prompt", "status"]


def plan_path(sitting, speaker):
    return common.PLANS_DIR / f"sitting{sitting}_s{speaker}.csv"


def build(sitting, speaker, rng):
    items = []
    for block, rep in SCHEDULE[sitting]:
        if block == "base":
            group = [("base", p) for p in BASELINE]
        elif block == "A":
            group = [(c, p) for p in PAIRS for c in ("search", "static")]
        else:
            group = [("off", p) for p in PAIRS]
        rng.shuffle(group)
        for cls, pid in group:
            items.append({
                "file": common.make_name(cls, speaker, pid, rep),
                "class": cls,
                "prompt_id": pid,
                "rep": rep,
                "search_setting": "off" if cls == "off" else "on",
                "prompt": prompt_text(cls, pid),
                "status": "pending",
            })
    for i, it in enumerate(items, 1):
        it["seq"] = i
    assert len(items) == EXPECTED[sitting]
    return items


def read_plan(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_plan(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLS)
        w.writeheader()
        for it in items:
            w.writerow({c: it[c] for c in PLAN_COLS})


def ensure_plan(sitting, speaker, force=False, seed=None):
    path = plan_path(sitting, speaker)
    if path.exists() and not force:
        return path, read_plan(path), False
    items = build(sitting, speaker, random.Random(seed))
    write_plan(path, items)
    return path, read_plan(path), True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sitting", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--force", action="store_true",
                    help="re-shuffle even if a plan exists (only before you start recording)")
    ap.add_argument("--seed", type=int, help="fixed shuffle seed (testing only)")
    args = ap.parse_args()

    cfg = common.load_config()
    path, items, created = ensure_plan(args.sitting, cfg["speaker"], args.force, args.seed)
    print(("Created " if created else "Using existing ") + str(path))
    for it in items:
        print(f"{it['seq']:>3}  {it['search_setting']:<3}  {it['file']:<34} {it['prompt']}")


if __name__ == "__main__":
    main()
