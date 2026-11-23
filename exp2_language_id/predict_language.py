"""Guess the language (English, Spanish or Hindi) of a capture with the trained model.

    python predict_language.py some_session.pcapng
    python predict_language.py processed/questions/hi/lang_hi_s1_pL04_r03.csv
    python predict_language.py FILE [FILE ...]

Takes a raw capture (.pcapng/.pcap: one question or a whole playlist session)
or a processed per-question .csv. The file name doesn't matter. It prints a
guess for every question/answer turn it finds, then an overall guess from the
average of the turn probabilities. If the name follows the lang_<xx>_...
pattern, it also says whether the guess was right.
"""
import argparse
import re
from pathlib import Path

import joblib
import numpy as np

import features
from questions import LANG_NAME
from train_model import MODEL_PATH


def predict_file(bundle, path):
    _, turns = features.turns_from_capture(path)
    print(f"\n{path}")
    if not turns:
        print("  No question/answer turns found. Is this a voice-mode capture?")
        return None
    model, labels = bundle["model"], list(bundle["model"].classes_)
    X = np.array([[t["features"][n] for n in bundle["features"]] for t in turns])
    proba = model.predict_proba(X)
    print(f"  {'turn':>4}  {'reply at':>13}  {'guess':<8} confidence")
    for t, p in zip(turns, proba):
        k = int(np.argmax(p))
        span = f"{t['r_start']:.0f}-{t['r_end']:.0f} s"
        print(f"  {t['turn']:>4}  {span:>13}  {LANG_NAME[labels[k]]:<8} {p[k]:.0%}")
    mean = proba.mean(axis=0)
    k = int(np.argmax(mean))
    others = ", ".join(f"{LANG_NAME[l]} {v:.0%}" for l, v in zip(labels, mean) if l != labels[k])
    print(f"  Overall: {LANG_NAME[labels[k]]} ({mean[k]:.0%} average; {others}) over {len(turns)} turn(s)")
    m = re.match(r"^lang_([a-z]{2})_", Path(path).name)
    if m and m[1] in LANG_NAME:
        print(f"  File name says {LANG_NAME[m[1]]}: {'correct' if m[1] == labels[k] else 'WRONG'}")
    return labels[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help=".pcapng / .pcap captures or processed .csv pieces")
    args = ap.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model yet. Run:  python train_model.py")
    bundle = joblib.load(MODEL_PATH)
    print(f"Model: {MODEL_PATH.name}, trained on {len(bundle['sessions'])} sessions, "
          f"cross-validated accuracy {bundle['cv_accuracy']:.0%}")
    print("Files the model was trained on will score well; test on new recordings for a fair check.")
    for path in args.files:
        predict_file(bundle, path)


if __name__ == "__main__":
    main()
