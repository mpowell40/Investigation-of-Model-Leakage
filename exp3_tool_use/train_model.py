"""Train a random forest that tells from packet metadata whether Claude ran a web search.

    python preprocess.py      first: find speech and reply in every capture
    python train_model.py

Labels are the manifest's search_observed (Y = searched), not the file-name class.
Two comparisons, reported separately:
  search_vs_static   current-info vs static prompts, both with search ON
  search_vs_off      the same current-info prompts with search ON vs OFF
                     (skipped while there are no tool_off_ captures)

Evaluation, per comparison:
  1. Cross-validation on sittings 1-2 only, GroupKFold grouped by prompt_id
     (T01..T08): every fold tests on prompts the model never saw, so it can't
     learn the topic. This is where comparisons between feature sets are made.
  2. Main test: train on sittings 1-2, test ONCE on sitting 3 (held out; a
     different day). Nothing is tuned on it.
Every accuracy is printed with its test count and a 95% Wilson interval.

Checks for the pitfalls from Experiment 2:
  reply length   reply duration alone, and the model without length features
  idle artifact  a model on silence-window (idle_) features only: if it works, red flag
  batch effect   can the sitting be predicted from the features?

Writes results/cv_predictions.csv, results/ml_features.csv and models/tool_rf.joblib
(the search_vs_static model trained on sittings 1-2, i.e. the one scored on sitting 3).
"""
import csv
import math
from collections import Counter

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

import features
from preprocess import ROOT

MODEL_PATH = ROOT / "models" / "tool_rf.joblib"
RESULTS_DIR = ROOT / "results"
TRAIN_SITTINGS = {"1", "2"}
TEST_SITTING = "3"
COMPARISONS = {
    "search_vs_static": ("search", "static"),
    "search_vs_off": ("search", "off"),
}
LABELS = ["nosearch", "search"]
NICE = {"search": "searched", "nosearch": "no search"}


def make_model():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0, n_jobs=-1)


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def acc_str(k, n):
    if n == 0:
        return "n/a (0 test examples)"
    lo, hi = wilson(k, n)
    return f"{k / n:6.1%}  ({k}/{n}, 95% CI {lo:.0%}-{hi:.0%})"


def matrix(X_rows, names):
    return np.array([[r["features"][n] for n in names] for r in X_rows], float)


def feature_sets(all_names):
    main = [n for n in all_names if n.startswith(features.MAIN_GROUPS)]
    return [
        ("all (latency + reply + gap)", main),
        ("reply duration only", ["r_dur"]),
        ("latency only (normalised)", ["lat_norm_s"]),
        ("first-packet + main-answer latency", ["lat_norm_s", "lat_main_s"]),
        ("all minus reply length", [n for n in main if n not in features.LENGTH_FEATURES]),
        ("idle windows only (red-flag check)", [n for n in all_names if n.startswith("idle_")]),
    ]


def cv_predict(rows, names, y):
    """Out-of-fold predictions on rows (GroupKFold by prompt_id)."""
    groups = np.array([r["prompt_id"] for r in rows])
    n_groups = len(set(groups))
    pred = np.empty(len(rows), dtype=object)
    proba = np.zeros(len(rows))
    if n_groups < 2 or len(set(y)) < 2:
        return None, None
    X = matrix(rows, names)
    for tr, te in GroupKFold(n_splits=min(8, n_groups)).split(X, y, groups):
        if len(set(y[tr])) < 2:
            pred[te] = y[tr][0]
            proba[te] = float(y[tr][0] == "search")
            continue
        m = make_model().fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        proba[te] = m.predict_proba(X[te])[:, list(m.classes_).index("search")]
    return pred, proba


def print_confusion(y, pred):
    print("      confusion (rows = true, columns = predicted):")
    print("                 " + "".join(f"{NICE[l]:>11}" for l in LABELS))
    for t in LABELS:
        print(f"        {NICE[t]:<9}" + "".join(f"{int(np.sum((y == t) & (pred == p))):>11}"
                                         for p in LABELS))


def print_per_prompt(rows, y, pred):
    out = []
    for pid in sorted({r["prompt_id"] for r in rows}):
        sel = np.array([r["prompt_id"] == pid for r in rows])
        out.append(f"{pid} {int(np.sum(pred[sel] == y[sel]))}/{int(sel.sum())}")
    print("      per prompt: " + "  ".join(out))


def chance(y):
    c = Counter(y.tolist()).most_common(1)[0]
    return f"chance (always '{NICE[c[0]]}') {c[1] / len(y):.0%} of {len(y)}"


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_comparison(name, classes, rows, all_names, pred_rows):
    sel = [r for r in rows if r["class"] in classes and r["label"]]
    have = Counter(r["class"] for r in sel)
    print(f"\n{'=' * 78}\n{name}: {classes[0]} vs {classes[1]}\n{'=' * 78}")
    missing = [c for c in classes if not have.get(c)]
    if missing:
        print(f"  SKIPPED: no {', '.join(missing)} captures. "
              + ("Record Set B (tool_off_*) to run this comparison." if "off" in missing else ""))
        return None
    train = [r for r in sel if r["sitting"] in TRAIN_SITTINGS]
    test = [r for r in sel if r["sitting"] == TEST_SITTING]
    y_tr = np.array([r["label"] for r in train])
    y_te = np.array([r["label"] for r in test])
    print(f"  train (sittings 1-2): {len(train)} captures {dict(Counter(y_tr.tolist()))}; "
          f"{len({r['prompt_id'] for r in train})} prompt groups")
    print(f"  test  (sitting 3):    {len(test)} captures {dict(Counter(y_te.tolist()))}")
    if len(set(y_tr)) < 2:
        print("  SKIPPED: training data has only one label.")
        return None

    print("\n  Feature set comparison (CV = GroupKFold by prompt on sittings 1-2; "
          "held-out = train 1-2, test 3):")
    print(f"    {'features':<38} {'n':>3}  {'CV accuracy':<34} held-out accuracy")
    sets = feature_sets(all_names)
    summary = {}
    for label, names in sets:
        pred, _ = cv_predict(train, names, y_tr)
        k_cv = int(np.sum(pred == y_tr))
        if test:
            m = make_model().fit(matrix(train, names), y_tr)
            p_te = m.predict(matrix(test, names))
            ho = acc_str(int(np.sum(p_te == y_te)), len(y_te))
        else:
            ho = "n/a (no sitting 3)"
        summary[label] = (k_cv, len(y_tr))
        print(f"    {label:<38} {len(names):>3}  {acc_str(k_cv, len(y_tr)):<34} {ho}")

    # Detailed report for the main feature set.
    label, names = sets[0]
    print(f"\n  Main model: {label}, {len(names)} features")
    pred, proba = cv_predict(train, names, y_tr)
    print(f"    CV on sittings 1-2:   {acc_str(int(np.sum(pred == y_tr)), len(y_tr))}   {chance(y_tr)}")
    print_confusion(y_tr, pred)
    print_per_prompt(train, y_tr, pred)
    for r, p, pr in zip(train, pred, proba):
        pred_rows.append({"comparison": name, "eval": "cv_sittings_1-2", "file": r["file"],
                          "class": r["class"], "prompt_id": r["prompt_id"], "sitting": r["sitting"],
                          "true": r["label"], "predicted": p, "prob_search": round(float(pr), 3)})

    model = make_model().fit(matrix(train, names), y_tr)
    if test:
        p_te = model.predict(matrix(test, names))
        pr_te = model.predict_proba(matrix(test, names))[:, list(model.classes_).index("search")]
        print(f"\n    HELD-OUT sitting 3:   {acc_str(int(np.sum(p_te == y_te)), len(y_te))}   {chance(y_te)}")
        print_confusion(y_te, p_te)
        print_per_prompt(test, y_te, p_te)
        for r, p, pr in zip(test, p_te, pr_te):
            pred_rows.append({"comparison": name, "eval": "heldout_sitting_3", "file": r["file"],
                              "class": r["class"], "prompt_id": r["prompt_id"], "sitting": r["sitting"],
                              "true": r["label"], "predicted": p, "prob_search": round(float(pr), 3)})
            if p != r["label"]:
                print(f"      wrong: {r['file']}  P(search) = {pr:.2f}")
        ho_acc = float(np.mean(p_te == y_te))
    else:
        ho_acc = None

    print("\n    Most useful features (model trained on sittings 1-2):")
    for i in np.argsort(model.feature_importances_)[::-1][:12]:
        f = names[i]
        med = "  ".join(f"{NICE[l]} {np.median([r['features'][f] for r, yy in zip(train, y_tr) if yy == l]):.3g}"
                        for l in LABELS)
        print(f"      {f:<22} {model.feature_importances_[i]:.3f}   median: {med}")
    return {"model": model, "features": names, "cv_accuracy": float(np.mean(pred == y_tr)),
            "cv_n": len(y_tr), "heldout_accuracy": ho_acc, "heldout_n": len(test),
            "classes": classes}


def batch_check(rows, all_names):
    """Can the sitting be predicted from the features? (Experiment 2's batch effect.)"""
    print(f"\n{'=' * 78}\nBatch-effect check: predict the SITTING from the features\n{'=' * 78}")
    tool = [r for r in rows if r["class"] in ("search", "static", "off")]
    # String labels like "1" trip sklearn's class_weight handling, so prefix them.
    y_s = np.array([f"s{r['sitting']}" for r in tool])
    y_day = np.array(["sitting 3" if r["sitting"] == TEST_SITTING else "sittings 1-2" for r in tool])
    groups = np.array([r["prompt_id"] for r in tool])
    print(f"  {len(tool)} search/static captures, sittings {dict(sorted(Counter(y_s.tolist()).items()))}. "
          "GroupKFold by prompt.")
    print("  (Diagnostic only: uses sitting 3's features but not its search labels; "
          "nothing below changes the model.)")
    main = [n for n in all_names if n.startswith(features.MAIN_GROUPS)]
    idle = [n for n in all_names if n.startswith("idle_")]
    for target, y in (("sitting 1/2/3", y_s), ("day: sitting 3 vs 1-2", y_day)):
        maj = Counter(y.tolist()).most_common(1)[0][1] / len(y)
        for label, names in (("main features", main), ("idle features", idle),
                             ("latency only", ["lat_s"])):
            X = matrix(tool, names)
            pred = np.empty(len(y), dtype=object)
            for tr, te in GroupKFold(n_splits=8).split(X, y, groups):
                pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
            print(f"    {target:<22} from {label:<14} {acc_str(int(np.sum(pred == y)), len(y))}"
                  f"   chance {maj:.0%}")
    m = make_model().fit(matrix(tool, main), y_s)
    top = np.argsort(m.feature_importances_)[::-1][:6]
    print("  Features that tell sittings apart, median by sitting:")
    for i in top:
        f = main[i]
        meds = "  ".join(f"s{s} {np.median([r['features'][f] for r in tool if r['sitting'] == s]):.4g}"
                         for s in sorted({r['sitting'] for r in tool}))
        print(f"    {f:<22} {meds}")
    base = [r for r in rows if r["class"] == "base"]
    meds = "  ".join(f"s{s} {np.median([r['features']['lat_s'] for r in base if r['sitting'] == s]):.3f}"
                     for s in sorted({r['sitting'] for r in base}))
    print(f"  Base-prompt first-reply latency, median by sitting: {meds}")


def main():
    print("Reading processed captures...")
    rows, baselines = features.load_dataset()
    all_names = features.feature_names(rows[0]["features"], features.GROUPS)
    print(f"{len(rows)} usable captures, {len(all_names)} features "
          f"({sum(n.startswith(features.MAIN_GROUPS) for n in all_names)} in the main model, "
          f"{sum(n.startswith('idle_') for n in all_names)} idle diagnostics)")
    print("Base-latency baseline per sitting (s): "
          + ", ".join(f"sitting {s}: {v:.3f}" for s, v in sorted(baselines.items())))

    tool = [r for r in rows if r["class"] in ("search", "static", "off")]
    expect = {"search": "Y", "static": "N", "off": "N"}
    diff = [r for r in tool if r["search_observed"].upper() != expect[r["class"]]]
    print(f"Observed label differs from file-name class in {len(diff)}/{len(tool)} captures"
          + (": " + ", ".join(r["file"] for r in diff) if diff else ""))

    write_csv(RESULTS_DIR / "ml_features.csv",
              [{**{k: r[k] for k in ("file", "class", "label", "prompt_id", "rep", "sitting")},
                **r["features"]} for r in rows],
              ["file", "class", "label", "prompt_id", "rep", "sitting"] + all_names)

    pred_rows, models = [], {}
    for name, classes in COMPARISONS.items():
        res = run_comparison(name, classes, rows, all_names, pred_rows)
        if res:
            models[name] = res

    batch_check(rows, all_names)

    write_csv(RESULTS_DIR / "cv_predictions.csv", pred_rows,
              ["comparison", "eval", "file", "class", "prompt_id", "sitting", "true",
               "predicted", "prob_search"])
    if models:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": models, "baselines": baselines, "sklearn": sklearn.__version__,
                     "trained_on": "sittings 1-2", "default": next(iter(models))}, MODEL_PATH)
        print(f"\nSaved model(s) {', '.join(models)} to {MODEL_PATH}")
    print(f"Predictions: {RESULTS_DIR / 'cv_predictions.csv'}")
    print(f"Features:    {RESULTS_DIR / 'ml_features.csv'}")


if __name__ == "__main__":
    main()
