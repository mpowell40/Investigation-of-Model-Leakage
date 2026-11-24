"""Guess whether Claude ran a web search in a capture, with the trained model.

    python predict_tool.py captures/tool_search_s1_pT03_r04.pcapng
    python predict_tool.py processed/tool_static_s1_pT05_r04.csv
    python predict_tool.py FILE [FILE ...]
    python predict_tool.py FILE --comparison search_vs_off

Takes a raw capture (.pcapng/.pcap, one prompt and one reply) or a processed
.csv. The file name doesn't matter. Prints "searched" / "didn't search" with the
model's confidence. If the name follows the tool_<class>_s<N>_p<TNN>_r<NN>
pattern, it also says whether the guess was right: against the manifest's
search_observed when the file is in it, else against the class in the name.

The saved model was trained on sittings 1-2 only, so sitting 3 files are a fair
test; sittings 1-2 files were seen in training and will look better than they are.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

import common
import features
from preprocess import sitting_of
from train_model import MODEL_PATH, TRAIN_SITTINGS


def truth_for(name, manifest):
    """('search'/'nosearch', source) from the manifest or file name, or (None, None)."""
    meta = common.parse_name(Path(name).stem + ".pcapng")
    if meta is None or meta["class"] == "base":
        return None, None, meta
    row = manifest.get(Path(name).stem + ".pcapng")
    if row and row.get("search_observed", "").upper() in ("Y", "N"):
        return ("search" if row["search_observed"].upper() == "Y" else "nosearch",
                "manifest search_observed", meta)
    return ("search" if meta["class"] == "search" else "nosearch", "file-name class", meta)


def predict_file(bundle, comp, path, manifest):
    entry = bundle["models"][comp]
    row = manifest.get(Path(path).stem + ".pcapng")
    sitting = str(sitting_of(row)) if row else None
    baselines = bundle["baselines"]
    if sitting in baselines:
        base, base_src = baselines[sitting], f"sitting {sitting} base captures"
    else:
        vals = [baselines[s] for s in sorted(baselines) if s in TRAIN_SITTINGS]
        base = float(np.median(vals)) if vals else None
        base_src = "median of training sittings (sitting unknown)"
    print(f"\n{path}")
    packets, edges, status, f = features.from_capture(path, base)
    if f is None:
        print(f"  Could not measure it: {status}. Is this a single-prompt voice-mode capture?")
        return None
    X = np.array([[f[n] for n in entry["features"]]])
    p = entry["model"].predict_proba(X)[0][list(entry["model"].classes_).index("search")]
    guess = "search" if p >= 0.5 else "nosearch"
    print(f"  speech {edges['speech_start']:.2f}-{edges['speech_end']:.2f} s, "
          f"reply {edges['reply_start']:.2f}-{edges['reply_end']:.2f} s, "
          f"latency {f['lat_s']:.2f} s ({f['lat_norm_s']:+.2f} vs baseline from {base_src})")
    word = "searched" if guess == "search" else "didn't search"
    print(f"  Guess: {word}  (P(search) = {p:.0%}, confidence {max(p, 1 - p):.0%})")
    truth, src, meta = truth_for(path, manifest)
    if truth:
        ok = "correct" if truth == guess else "WRONG"
        print(f"  Truth ({src}): {'searched' if truth == 'search' else 'did not search'}: {ok}")
        if sitting in TRAIN_SITTINGS:
            print("  (this file was in the training data)")
    elif meta and meta["class"] == "base":
        print("  (base_ latency capture: not a search/static prompt, no ground truth)")
    return guess


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help=".pcapng / .pcap captures or processed .csv files")
    ap.add_argument("--comparison", help="which saved model to use (default: search_vs_static)")
    args = ap.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model yet. Run:  python train_model.py")
    bundle = joblib.load(MODEL_PATH)
    comp = args.comparison or bundle["default"]
    if comp not in bundle["models"]:
        raise SystemExit(f"No '{comp}' model saved. Available: {', '.join(bundle['models'])}")
    e = bundle["models"][comp]
    ho = (f"held-out sitting 3 accuracy {e['heldout_accuracy']:.0%} on {e['heldout_n']}"
          if e["heldout_accuracy"] is not None else "no held-out score")
    print(f"Model: {comp}, trained on {bundle['trained_on']}; "
          f"CV accuracy {e['cv_accuracy']:.0%} on {e['cv_n']}, {ho}")
    cfg = common.load_config(require=False)
    manifest = {r["file"]: r for r in common.read_manifest(common.manifest_path(cfg))}
    for path in args.files:
        predict_file(bundle, comp, path, manifest)


if __name__ == "__main__":
    main()
