"""
diagnose_flex.py

Why does 'Flex / PCB / Solder / Wiring' score F1 = 0.00 for BOTH the rules
baseline and the Random Forest? Three questions, answered from data:

1. Where do the Flex test rows actually get predicted, and which raw
   root_cause_category strings do they carry?  (Is the class itself a mixed
   bag, or is the model just ignoring it?)
2. Does ANY single FT feature separate Flex from the rest?  One-vs-rest ROC
   AUC per feature (0.5 = no separation, 1.0 / 0.0 = perfect). Features are
   ranked on TRAIN; the TEST AUC is printed beside it so a feature that only
   looks good by chance on train is visible as such.
3. How are Flex units spread across the calibration-sequence zones
   (chain_depth)?  If they sit in every zone, the sequence signal can't
   isolate them.

Read-only diagnostic: nothing here feeds back into training or into the
baseline rules, so test data is only LOOKED AT, never fitted to.

Usage
-----
    python src/diagnose_flex.py \
        --train-features data/processed/train_features.csv \
        --test-features  data/processed/test_features.csv \
        --test-units     data/processed/test_units.csv \
        --model          models/defect_classifier.joblib
"""

import argparse
import csv
from collections import Counter, defaultdict

import warnings

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FLEX = "Flex / PCB / Solder / Wiring"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--test-features", required=True)
    ap.add_argument("--test-units", required=True,
                    help="test_units.csv -- same row order as test_features.csv; "
                         "used only to show the raw root_cause_category text")
    ap.add_argument("--model", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    train = pd.read_csv(args.train_features)
    test = pd.read_csv(args.test_features)
    with open(args.test_units, encoding="latin-1") as f:
        units = list(csv.DictReader(f))
    if len(units) != len(test):
        raise SystemExit(f"Row mismatch: {len(units)} test_units rows vs {len(test)} "
                         f"test_features rows -- they must come from the same split run.")

    feat_cols = [c for c in train.columns if c not in ("PO", "model_target")]
    if not (train["model_target"] == FLEX).any():
        raise SystemExit(f"No '{FLEX}' rows in {args.train_features} -- nothing to diagnose "
                         f"(check you passed the real train_features.csv).")
    model = joblib.load(args.model)
    pred = model.predict(test[feat_cols])

    # ---- 1. Where do Flex test rows go? -------------------------------------
    is_flex_test = (test["model_target"] == FLEX).to_numpy()
    print(f"=== 1. Flex test rows: {is_flex_test.sum()} of {len(test)} ===")
    print("Predicted as:")
    for cls, n in Counter(pred[is_flex_test]).most_common():
        print(f"  {n:4d}  {cls}")
    print("\nRaw root_cause_category of Flex test rows (count, mostly predicted-as):")
    by_cat = defaultdict(list)
    for i in range(len(test)):
        if is_flex_test[i]:
            by_cat[(units[i].get("root_cause_category") or "").strip()].append(pred[i])
    for cat, preds in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        top_pred, top_n = Counter(preds).most_common(1)[0]
        print(f"  {len(preds):4d}  {cat}   -> {top_pred} ({top_n}/{len(preds)})")
    flex_predicted = (pred == FLEX).sum()
    print(f"\nRows the model predicted as Flex at all: {flex_predicted}")

    # ---- 2. Does any single feature separate Flex? ---------------------------
    y_tr = (train["model_target"] == FLEX).astype(int)
    y_te = (test["model_target"] == FLEX).astype(int)
    rows = []
    for c in feat_cols:
        if train[c].nunique() < 2:
            continue
        try:
            auc_tr = roc_auc_score(y_tr, train[c])
            auc_te = roc_auc_score(y_te, test[c]) if test[c].nunique() > 1 and y_te.nunique() > 1 else float("nan")
        except ValueError:
            continue
        rows.append((c, auc_tr, auc_te))
    rows.sort(key=lambda r: -abs(r[1] - 0.5))
    print(f"\n=== 2. Single-feature separation of Flex vs rest (ROC AUC; 0.5 = none) ===")
    print(f"{'feature':<34}{'AUC train':>10}{'AUC test':>10}")
    for c, a_tr, a_te in rows[:args.top]:
        print(f"{c:<34}{a_tr:>10.3f}{a_te:>10.3f}")
    best = abs(rows[0][1] - 0.5) + 0.5 if rows else 0.5
    verdict = ("no single feature separates Flex well (best |AUC| < 0.65)"
               if best < 0.65 else "at least one feature shows real separation -- worth modelling")
    print(f"\nBest single-feature separation on train: {best:.3f} -> {verdict}")

    # ---- 3. Flex across calibration-sequence zones ---------------------------
    print("\n=== 3. Flex share by chain_depth (train) ===")
    if "chain_depth" in train.columns:
        g = train.groupby("chain_depth")["model_target"].agg(
            n="size", flex=lambda s: (s == FLEX).sum())
        g["flex_%"] = (100 * g["flex"] / g["n"]).round(1)
        print(g.to_string())
    else:
        print("(no chain_depth column -- rerun features.py with the reach-indicator version)")


if __name__ == "__main__":
    main()
