"""Train a random forest that tells English, Spanish and Hindi apart from packet metadata.

    python preprocess.py      first: split the sessions into per-question pieces
    python train_model.py

Each question/answer piece in processed/ is one example. Accuracy is measured
with leave-one-session-out cross-validation: train on every session but one,
test on the one left out, and rotate, so pieces from the same recording never
sit on both sides. The final model is then trained on everything and saved to
models/lang_rf.joblib for predict_language.py.
"""
import csv
from collections import Counter

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import features
from preprocess import OUT_DIR, ROOT
from questions import LANG_NAME, LANGS

MODEL_PATH = ROOT / "models" / "lang_rf.joblib"
RESULTS_DIR = ROOT / "results"
META_COLS = ["file", "lang", "session", "prompt_id"]


def make_model():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def load_dataset():
    index = OUT_DIR / "index.csv"
    if not index.exists():
        raise SystemExit("No processed data yet. Run:  python preprocess.py")
    rows = []
    with open(index, newline="") as f:
        for r in csv.DictReader(f):
            if r["reply_start"] == "":
                print(f"  skipping {r['file']}: no reply found in it")
                continue
            packets = features.read_piece(OUT_DIR / "questions" / r["lang"] / r["file"])
            row = {"file": r["file"], "lang": r["lang"],
                   "session": f"{r['lang']}_{r['speaker'].lower()}_r{int(r['rep']):02d}",
                   "prompt_id": r["prompt_id"]}
            row.update(features.turn_features(packets, float(r["reply_start"]), float(r["reply_end"])))
            rows.append(row)
    if not rows:
        raise SystemExit("processed/index.csv lists no usable pieces.")
    return rows


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    print("Reading processed pieces...")
    rows = load_dataset()
    names = [k for k in rows[0] if k not in META_COLS]
    X = np.array([[r[n] for n in names] for r in rows])
    y = np.array([r["lang"] for r in rows])
    groups = np.array([r["session"] for r in rows])
    sessions = sorted(set(groups))
    labels = [l for l in LANGS if l in set(y)]
    print(f"{len(rows)} pieces {dict(Counter(y.tolist()))} from {len(sessions)} sessions, "
          f"{len(names)} features")
    write_csv(RESULTS_DIR / "features.csv", rows, META_COLS + names)
    if len(labels) < 2 or len(sessions) < 2:
        raise SystemExit("Need at least two languages and two sessions to evaluate.")

    # Leave one session out.
    pred = np.array([""] * len(y), dtype=object)
    proba = np.zeros((len(y), len(labels)))
    print("\nLeave-one-session-out cross-validation:")
    for s in sessions:
        test = groups == s
        m = make_model().fit(X[~test], y[~test])
        pred[test] = m.predict(X[test])
        p = m.predict_proba(X[test])
        for j, lab in enumerate(m.classes_):
            proba[test, labels.index(lab)] = p[:, j]
        print(f"  test on {s:<10} {np.sum(pred[test] == y[test]):>3}/{test.sum():<3} correct")

    acc = float(np.mean(pred == y))
    print(f"\nOverall piece accuracy: {acc:.1%}  ({np.sum(pred == y)}/{len(y)})   "
          f"chance = {1 / len(labels):.0%}")
    cm = confusion_matrix(y, pred.astype(str), labels=labels)
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("          " + "".join(f"{l:>6}" for l in labels))
    for lab, line in zip(labels, cm):
        print(f"  {lab:<6}  " + "".join(f"{v:>6}" for v in line))
    print("\nPer language:")
    for lab, line in zip(labels, cm):
        print(f"  {LANG_NAME[lab]:<8} {line[labels.index(lab)]}/{line.sum()}")

    # Whole-session vote: average the per-piece probabilities.
    votes = sum(labels[int(np.argmax(proba[groups == s].mean(axis=0)))] == y[groups == s][0]
                for s in sessions)
    print(f"\nWhole-session vote accuracy: {votes}/{len(sessions)}")

    print("\nPer-question accuracy:")
    for pid in sorted({r["prompt_id"] for r in rows}):
        sel = np.array([r["prompt_id"] == pid for r in rows])
        print(f"  {pid}: {np.sum(pred[sel] == y[sel])}/{sel.sum()}")

    for r, p, pr in zip(rows, pred, proba):
        r["predicted"] = p
        r["confidence"] = round(float(pr.max()), 3)
        for lab, v in zip(labels, pr):
            r[f"prob_{lab}"] = round(float(v), 3)
    write_csv(RESULTS_DIR / "cv_predictions.csv", rows,
              ["file", "session", "prompt_id", "lang", "predicted", "confidence"]
              + [f"prob_{l}" for l in labels])

    # Final model on all data.
    final = make_model().fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": names, "labels": labels,
                 "sklearn": sklearn.__version__, "cv_accuracy": acc, "sessions": sessions},
                MODEL_PATH)

    print("\nMost useful features (final model):")
    for i in np.argsort(final.feature_importances_)[::-1][:12]:
        print(f"  {names[i]:<24} {final.feature_importances_[i]:.3f}")
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Cross-validation predictions: {RESULTS_DIR / 'cv_predictions.csv'}")


if __name__ == "__main__":
    main()
