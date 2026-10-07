"""
score_baseline.py

Computes the same per-class precision/recall/F1 + macro F1 report for
rules_baseline.py's predictions that src/model.py already computes for the
ML model -- so the baseline-vs-improved comparison is apples-to-apples
(macro F1, not plain accuracy, since accuracy is misleading under this
dataset's class imbalance: see model.py's own docstring).

Usage
-----
    python src/score_baseline.py --in reports/baseline_predictions.csv
"""

import argparse
import csv

from sklearn.metrics import classification_report, f1_score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                     help="reports/baseline_predictions.csv, as written by rules_baseline.py "
                          "(needs model_target and baseline_prediction columns)")
    ap.add_argument("--out", default="reports/baseline_classification_report.txt")
    args = ap.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows or "model_target" not in rows[0] or "baseline_prediction" not in rows[0]:
        raise SystemExit(f"{args.in_path} needs 'model_target' and 'baseline_prediction' columns "
                          f"-- re-run rules_baseline.py on a file that has model_target (e.g. "
                          f"test_units.csv), not one without ground truth.")

    y_true = [r["model_target"] for r in rows]
    y_pred = [r["baseline_prediction"] for r in rows]

    report = classification_report(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("=== Baseline classification report (test set) ===")
    print(report)
    print(f"Macro F1 (unweighted across classes): {macro_f1:.3f}")
    print("(Compare this number directly against the improved method's macro F1 in "
          "reports/figures/classification_report.txt -- that's the real baseline-vs-"
          "improved comparison, not the plain accuracy numbers.)")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
        f.write(f"\nMacro F1: {macro_f1:.3f}\n")
    print(f"\nWritten -> {args.out}")


if __name__ == "__main__":
    main()
