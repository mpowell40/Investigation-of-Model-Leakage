"""Experiment 1 prediction: guess the sentence said in each turn of a capture.

    python predict.py captures/phrase_batchA_s1_r17.pcapng
    python predict.py processed/phrase_batchA_s1_r17_t03.npz
    python predict.py FILE --model models/exp1_setA.joblib

Model choice (unless --model): Set B names -> exp1_setB, otherwise exp1_setA.
Turns without detectable speech (noise files) use the reply-only Set A model.
If the file follows the naming pattern and is in manifest_exp1.csv, each guess
is checked against the phrase order logged for it.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment

import preprocess as P
from features import turn_features

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def load(name_or_path):
    p = Path(name_or_path)
    if not p.exists():
        p = MODELS / f"{name_or_path}.joblib"
    if not p.exists():
        raise SystemExit(f"Model {name_or_path} not found. Run: python train_model.py")
    return joblib.load(p), p.stem


def turns_from_pcap(path, manifest):
    meta = manifest.get(path.name)
    rows, frow, (up, dn, info, det, turns) = P.process_file(path, meta, write=False)
    out = []
    for r, t in zip(rows, turns):
        out.append({"name": f"t{r['turn']:02d}", "label": r["label"],
                    "feats": turn_features(up, dn, t["replies"], t["speech"]), "row": r})
    return out, frow, meta


def turns_from_piece(path, manifest):
    z = np.load(path)
    row = json.loads(str(z["meta"][0]))
    meta = manifest.get(row["source_file"])
    feats = turn_features(z["up"], z["dn"], z["replies"], z["speech"])
    label = row.get("label", "")
    if not label and meta and row.get("status") == "ok":
        label = meta["order"][int(row["turn"]) - 1]
    return [{"name": f"t{int(row['turn']):02d}", "label": label, "feats": feats, "row": row}], \
        {"status": row.get("status", ""), "notes": "", "file": row["source_file"]}, meta


def predict_one(bundle, feats):
    x = np.array([[feats.get(c, np.nan) for c in bundle["features"]]], float)
    p = bundle["model"].predict_proba(x)[0]
    return dict(zip(map(str, bundle["model"].classes_), p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help=".pcapng capture or processed/*.npz piece")
    ap.add_argument("--model", help="model name in models/ or path to a .joblib")
    args = ap.parse_args()
    path = Path(args.file)
    manifest = P.load_manifest()
    if path.suffix == ".npz":
        turns, frow, meta = turns_from_piece(path, manifest)
        source = frow["file"]
    else:
        turns, frow, meta = turns_from_pcap(path, manifest)
        source = path.name
    named = P.parse_name(source)
    if args.model:
        main_model, main_name = load(args.model)
    elif named and named["set"] == "batchB":
        main_model, main_name = load("exp1_setB")
    else:
        main_model, main_name = load("exp1_setA")
    reply_model = reply_name = None

    print(f"{source}: {len(turns)} turn(s), detection {frow['status']}"
          + (f" ({frow['notes']})" if frow.get("notes") else ""))
    if source in main_model.get("trained_on", []):
        print(f"  NOTE: {source} was in the training data of {main_name}; this is not a fair test.")
    have_truth = bool(named and meta and any(t["label"] for t in turns))
    if named and not meta:
        print("  file name follows the pattern but isn't in manifest_exp1.csv: no labels to check")
    elif named and meta and not have_truth:
        print("  labels not checked: detection status is", frow["status"])

    probs, rows = [], []
    for t in turns:
        no_speech = np.isnan(t["feats"].get("lat_reply", np.nan))
        if no_speech and reply_model is None:
            reply_model, reply_name = load("exp1_setA_reply")
        b, bn = (reply_model, reply_name) if no_speech else (main_model, main_name)
        pr = predict_one(b, t["feats"])
        guess = max(pr, key=pr.get)
        second = sorted(pr.items(), key=lambda kv: -kv[1])[1]
        mark = ""
        if have_truth and t["label"]:
            mark = "  right" if guess == t["label"] else f"  WRONG (was {t['label']})"
        print(f"  {t['name']}  guess {guess}  conf {pr[guess]:.2f}  (next {second[0]} {second[1]:.2f})  "
              f"[{bn}]{mark}")
        probs.append(pr)
        rows.append((t, guess))

    # Overall
    classes = sorted(probs[0]) if probs else []
    if len(turns) > 1:
        if (frow["status"] != "noise" and len(turns) <= len(classes)
                and all(set(p) == set(classes) for p in probs)):
            cost = -np.log(np.array([[p[c] for c in classes] for p in probs]) + 1e-9)
            ri, ci = linear_sum_assignment(cost)
            order = [classes[c] for _, c in sorted(zip(ri, ci))]
            print(f"  best order if each phrase is said once: {','.join(order)}")
            if have_truth:
                truth = [t["label"] for t, _ in rows]
                print(f"  logged order:                           {','.join(truth)}"
                      f"   ({sum(a == b for a, b in zip(order, truth))}/{len(truth)} right)")
        conf = np.mean([max(p.values()) for p in probs])
        if have_truth:
            n = sum(g == t["label"] for t, g in rows)
            print(f"  overall: {n}/{len(rows)} turns right ({n / len(rows):.0%}), mean confidence {conf:.2f}")
        else:
            print(f"  overall: mean confidence {conf:.2f}")


if __name__ == "__main__":
    sys.exit(main())
