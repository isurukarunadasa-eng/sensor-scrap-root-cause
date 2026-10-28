"""
sensitivity_merged_class.py

SECONDARY (sensitivity) analysis -- not the headline result.

Why: Flex / PCB / Solder / Wiring scores F1 = 0.00 for both methods. The flex
PCB physically carries the ASIC, and the engineers' own -7022 entry lists both
as causes, so on FT data alone the two may simply not be separable. This asks:
"if we stop requiring the two to be told apart, how well do both methods do?"

It merges 'ASIC / EEPROM' and 'Flex / PCB / Solder / Wiring' into one class,
'ASIC / Flex interface', and reports macro F1 three ways on the SAME held-out
test set:

  A. Baseline (rules) predictions, relabelled -- no retraining needed.
  B. The already-trained 7-class model's predictions, relabelled -- no retraining.
  C. A Random Forest RETRAINED on the merged labels (same grid + GroupKFold(PO)
     as model.py, train split only).

Reading the numbers: merging classes changes what 'macro F1' averages over
(6 classes instead of 7), so the merged scores are NOT comparable to the
headline 0.496 / 0.349 as a leaderboard. Compare baseline vs ML within this
analysis, and compare B vs C to see whether retraining on the merged label
helps beyond simply relabelling.

Usage
-----
    python src/sensitivity_merged_class.py \
        --train data/processed/train_features.csv \
        --test  data/processed/test_features.csv \
        --baseline-preds reports/baseline_predictions.csv \
        --model models/defect_classifier.joblib \
        --out reports/sensitivity_merged_class.txt
"""

import argparse
import csv
import os
import sys
import warnings

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_features, train_model  # noqa: E402

warnings.filterwarnings("ignore")

ASIC = "ASIC / EEPROM"
FLEX = "Flex / PCB / Solder / Wiring"
MERGED = "ASIC / Flex interface"


def merge(labels):
    return pd.Series(labels).replace({ASIC: MERGED, FLEX: MERGED}).to_numpy()


def section(title, y_true, y_pred, lines):
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = (y_true == y_pred).mean()
    lines.append(f"=== {title} ===")
    lines.append(classification_report(y_true, y_pred, zero_division=0))
    lines.append(f"Macro F1: {f1:.3f}   Accuracy: {acc:.3f}\n")
    return f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--baseline-preds", required=True,
                    help="reports/baseline_predictions.csv (same row order as test_features.csv)")
    ap.add_argument("--model", default=None, help="trained 7-class model (joblib); optional")
    ap.add_argument("--out", default="reports/sensitivity_merged_class.txt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X_tr, y_tr, g_tr, _ = load_features(args.train)
    X_te, y_te, g_te, _ = load_features(args.test)
    y_true = merge(y_te)

    with open(args.baseline_preds, encoding="utf-8") as f:
        brows = list(csv.DictReader(f))
    if len(brows) != len(X_te):
        raise SystemExit(f"Row mismatch: {len(brows)} baseline rows vs {len(X_te)} test rows -- "
                         f"both must come from the same split run.")
    if [r["model_target"] for r in brows] != list(y_te):
        raise SystemExit("baseline_predictions.csv rows are not in the same order / labels as "
                         "test_features.csv -- rerun rules_baseline.py on the current test_units.csv.")

    lines = [f"Merged class: '{ASIC}' + '{FLEX}' -> '{MERGED}'",
             f"Train {len(X_tr)} rows / {g_tr.nunique()} POs, test {len(X_te)} rows / {g_te.nunique()} POs\n",
             f"Test class counts (merged): {pd.Series(y_true).value_counts().to_dict()}\n"]

    res = {}
    res["A baseline"] = section("A. Baseline (rules), relabelled", y_true,
                                merge([r["baseline_prediction"] for r in brows]), lines)

    if args.model and os.path.exists(args.model):
        m = joblib.load(args.model)
        res["B 7-class RF, relabelled"] = section(
            "B. Existing 7-class Random Forest, predictions relabelled", y_true,
            merge(m.predict(X_te)), lines)
    else:
        lines.append("(B skipped: no --model file found)\n")

    y_tr_m = pd.Series(merge(y_tr), index=y_tr.index)
    print(f"Retraining on merged labels: {y_tr_m.value_counts().to_dict()}")
    m2 = train_model(X_tr, y_tr_m, g_tr, args.seed)
    res["C RF retrained on merged"] = section("C. Random Forest retrained on merged labels",
                                              y_true, m2.predict(X_te), lines)

    lines.append("Summary (macro F1, merged 6-class view):")
    for k, v in res.items():
        lines.append(f"  {k:<28} {v:.3f}")
    lines.append("\nNote: not comparable as a leaderboard to the 7-class headline numbers "
                 "(different number of classes). Secondary analysis only.")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nWritten -> {args.out}")


if __name__ == "__main__":
    main()
