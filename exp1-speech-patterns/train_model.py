"""Experiment 1 training: which sentence was said, from one turn's traffic.

Task A: P02-P07 (6 classes) on Set A. Leave-one-file-out cross-validation on
        reps 1-16 (every file is its own chat, so no chat lands on both sides),
        then ONE test of the final model on the held-out reps 17-20 (Oct 27).
Task B: D01/D03/D05 (3 classes) on Set B. Leave-one-file-out over its 6 files.
        All six were recorded on Oct 25, so there is no held-out day.
Noise:  robustness check only. Speech can't be found under café noise, so a
        reply-only Set A model is run on the noise turns and its guesses are
        printed, not scored (the 10 s windows don't map to one phrase).

Also reported: accuracy when each phrase may be used once per file (the
protocol says each sentence is said exactly once per batch), and an ablation
of feature groups (speech vs reply vs latency).

    python train_model.py             # both tasks + noise check
    python train_model.py --task A
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
META = ["piece", "source_file", "set", "rep", "sitting", "split", "turn", "label", "status"]
TASKS = {"A": "batchA", "B": "batchB"}


def make_model():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def load_features():
    path = RESULTS / "features.csv"
    if not path.exists():
        raise SystemExit("No results/features.csv. Run: python features.py")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def feature_names(rows, groups=None):
    names = [c for c in rows[0] if c not in META]
    if groups:
        names = [c for c in names if c.split("_")[0] in groups]
    return names


def matrix(rows, names):
    return np.array([[float(r[c]) if r[c] != "" else np.nan for c in names] for r in rows])


def lofo(rows, names):
    """Leave-one-file-out probabilities, in row order."""
    X, y = matrix(rows, names), np.array([r["label"] for r in rows])
    files = np.array([r["source_file"] for r in rows])
    classes = sorted(set(map(str, y)))
    proba = np.zeros((len(rows), len(classes)))
    for f in sorted(set(files)):
        test = files == f
        m = make_model().fit(X[~test], y[~test])
        p = m.predict_proba(X[test])
        proba[np.ix_(test, [classes.index(c) for c in m.classes_])] = p
    return proba, classes


def assign(rows, proba, classes):
    """Per file, give each phrase to at most one turn (max total log-probability)."""
    out = [None] * len(rows)
    by_file = defaultdict(list)
    for i, r in enumerate(rows):
        by_file[r["source_file"]].append(i)
    for idx in by_file.values():
        cost = -np.log(proba[idx] + 1e-9)
        ri, ci = linear_sum_assignment(cost)
        for a, b in zip(ri, ci):
            out[idx[a]] = classes[b]
    return out


def report(title, rows, proba, classes, show_sessions=True):
    y = [r["label"] for r in rows]
    pred = [str(classes[i]) for i in proba.argmax(1)]
    acc = np.mean([a == b for a, b in zip(y, pred)])
    maj = max(Counter(y).values()) / len(y)
    print(f"\n=== {title}")
    print(f"turns {len(y)}, files {len({r['source_file'] for r in rows})}, classes {len(classes)}")
    print(f"accuracy {acc:.1%}   (chance {1 / len(classes):.1%}, majority {maj:.1%})")
    asg = assign(rows, proba, classes)
    print(f"accuracy with one-phrase-per-file assignment {np.mean([a == b for a, b in zip(y, asg)]):.1%}")
    cm = confusion_matrix(y, pred, labels=classes)
    print("confusion (rows = true, cols = predicted)")
    print("        " + " ".join(f"{c:>5s}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"  {c:5s} " + " ".join(f"{n:5d}" for n in row))
    print("per class   precision  recall")
    for i, c in enumerate(classes):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
        print(f"  {c:5s}     {tp / max(tp + fp, 1):6.0%}  {tp / max(tp + fn, 1):6.0%}")
    if show_sessions:
        for key in ("sitting", "source_file"):
            g = defaultdict(list)
            for r, p in zip(rows, pred):
                g[r[key]].append(r["label"] == p)
            print(f"per {'file' if key == 'source_file' else key}:  " + "  ".join(
                f"{k.replace('phrase_', '').replace('.pcapng', '')} {np.mean(v):.0%}" for k, v in sorted(g.items())))
    return pred, asg, acc


def top_features(model, names, n=12):
    imp = sorted(zip(model.feature_importances_, names), reverse=True)[:n]
    print("top features (RF importance)")
    for v, k in imp:
        print(f"  {v:.3f}  {k}")


def prediction_rows(rows, proba, classes, pred, asg, evaluation, model_name):
    out = []
    for r, p, a, pr in zip(rows, pred, asg, proba):
        d = {k: r[k] for k in ("piece", "source_file", "set", "sitting", "split", "turn", "label")}
        d.update(eval=evaluation, model=model_name, pred=p, confidence=round(float(pr.max()), 3),
                 correct=(p == r["label"]) if r["label"] else "", assigned_pred=a or "")
        d.update({f"p_{c}": round(float(v), 3) for c, v in zip(classes, pr)})
        out.append(d)
    return out


def fit_and_save(rows, names, name, task, note):
    X, y = matrix(rows, names), np.array([r["label"] for r in rows])
    m = make_model().fit(X, y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": m, "features": names, "classes": list(m.classes_), "task": task,
                 "trained_on": sorted({r["source_file"] for r in rows}), "note": note},
                MODELS / f"{name}.joblib")
    return m


def ablation(rows):
    print("ablation (leave-one-file-out accuracy by feature group)")
    for groups in (["speech"], ["pre"], ["reply"], ["lat"], ["reply", "lat"], ["speech", "pre"],
                   ["speech", "pre", "reply", "lat"]):
        names = feature_names(rows, groups)
        proba, classes = lofo(rows, names)
        acc = np.mean([classes[i] == r["label"] for i, r in zip(proba.argmax(1), rows)])
        print(f"  {'+'.join(groups):18s} {len(names):3d} features  {acc:.1%}")


def task_a(all_rows, out_rows, do_ablation):
    rows = [r for r in all_rows if r["set"] == "batchA" and r["status"] == "ok"]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "heldout"]
    skipped = [r for r in all_rows if r["set"] == "batchA" and r["status"] != "ok"]
    if skipped:
        print(f"(skipping {len(skipped)} turns from flagged Set A files)")
    names = feature_names(train)
    proba, classes = lofo(train, names)
    pred, asg, _ = report("Task A: P02-P07, leave-one-file-out on reps 1-16", train, proba, classes)
    out_rows += prediction_rows(train, proba, classes, pred, asg, "cv", "exp1_setA")
    m = fit_and_save(train, names, "exp1_setA", "A", "Set A reps 1-16, all features")
    top_features(m, names)
    if do_ablation:
        ablation(train)

    # Held-out sitting: scored once, with the frozen model.
    if test:
        p = np.zeros((len(test), len(classes)))
        p[:, [classes.index(c) for c in m.classes_]] = m.predict_proba(matrix(test, names))
        pred, asg, _ = report("Task A HELD-OUT: reps 17-20 (Oct 27), model frozen on reps 1-16",
                              test, p, classes)
        out_rows += prediction_rows(test, p, classes, pred, asg, "heldout", "exp1_setA")

    # Reply-only model for the noise check (no speech under café noise).
    rnames = feature_names(train, ["reply"])
    mr = fit_and_save(train, rnames, "exp1_setA_reply", "A",
                      "Set A reps 1-16, reply features only (for files without detectable speech)")
    noise = [r for r in all_rows if r["set"] == "noise"]
    if noise:
        p = np.zeros((len(noise), len(classes)))
        p[:, [classes.index(c) for c in mr.classes_]] = mr.predict_proba(matrix(noise, rnames))
        pred = [str(classes[i]) for i in p.argmax(1)]
        print("\n=== Noise robustness check (reply-only Set A model, not scored)")
        rp, rc = lofo(train, rnames)
        racc = np.mean([rc[i] == r["label"] for i, r in zip(rp.argmax(1), train)])
        print(f"reply-only model, quiet leave-one-file-out accuracy for reference: {racc:.1%}")
        quiet_dur = np.array([float(r["reply_dur"]) for r in train])
        for f in sorted({r["source_file"] for r in noise}):
            idx = [i for i, r in enumerate(noise) if r["source_file"] == f]
            guesses = "  ".join(f"t{noise[i]['turn']}:{pred[i]}({p[i].max():.2f},{float(noise[i]['reply_dur']):.1f}s)"
                                for i in idx)
            print(f"  {f.replace('.pcapng', ''):24s} [{noise[idx[0]]['sitting']}] {guesses}")
        nd = np.array([float(r["reply_dur"]) for r in noise])
        print(f"reply duration: quiet {np.median(quiet_dur):.1f} s median, noise {np.median(nd):.1f} s median")
        print(f"guess counts: {dict(Counter(map(str, pred)))}  (reading order was P02..P07, cut at 10 s windows)")
        asg = [""] * len(noise)
        out_rows += prediction_rows(noise, p, classes, pred, asg, "noise", "exp1_setA_reply")


def task_b(all_rows, out_rows, do_ablation):
    rows = [r for r in all_rows if r["set"] == "batchB" and r["status"] == "ok"]
    if not rows:
        return
    names = feature_names(rows)
    proba, classes = lofo(rows, names)
    pred, asg, _ = report("Task B: D01/D03/D05, leave-one-file-out (all Oct 25, no held-out day)",
                          rows, proba, classes, show_sessions=False)
    out_rows += prediction_rows(rows, proba, classes, pred, asg, "cv", "exp1_setB")
    m = fit_and_save(rows, names, "exp1_setB", "B", "Set B, all 6 files (Oct 25)")
    top_features(m, names, n=8)
    if do_ablation:
        ablation(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["A", "B", "all"], default="all")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    rows = load_features()
    out_rows = []
    if args.task in ("A", "all"):
        task_a(rows, out_rows, not args.no_ablation)
    if args.task in ("B", "all"):
        task_b(rows, out_rows, not args.no_ablation)
    RESULTS.mkdir(exist_ok=True)
    cols = list(dict.fromkeys(k for r in out_rows for k in r))
    with open(RESULTS / "cv_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {RESULTS / 'cv_predictions.csv'} and models/*.joblib")


if __name__ == "__main__":
    sys.exit(main())
