"""
predict_root_cause.py

The operational pipeline: FA uploads a Final Tester report -> this script finds
the FAILED sensors in it and predicts the root-cause defect family for each.

No pass/fail model is involved. FA can already see which sensors passed; a unit
counts as failed when the tester flagged it (any H / L / ! cell, which includes
every ASIC / EEPROM error code). Unflagged units are not predicted on; the
count is printed so nothing disappears silently.

For each failed sensor it reports
  - the improved method's prediction (Random Forest) with its confidence and the
    next-most-likely families,
  - the top FT parameters that drove that call (SHAP),
  - the baseline's engineering-rules prediction, its evidence tier and reason,
  - whether the two agree, and a REVIEW flag when confidence is low or the two
    methods disagree.

Parity with training: the feature vector is built by calling the SAME
features.build_feature_matrix used for training, with the report's failed units
standing in for the "test" file. Medians are therefore fit on the training data
only, and the __missing / chain_depth features are computed by the same code.
--selfcheck proves this on your own data before you trust any prediction.

Usage
-----
    # 1) one-off parity check (should print PARITY OK)
    python src/predict_root_cause.py --selfcheck \
        --train data/processed/train_units.csv \
        --test-units data/processed/test_units.csv \
        --test-features data/processed/test_features.csv

    # 2) predict for an uploaded report (a file, or a folder of .xlsx files)
    python src/predict_root_cause.py \
        --report path/to/FT_report.xlsx \
        --train data/processed/train_units.csv \
        --model models/defect_classifier.joblib \
        --out reports/predictions.csv
"""

import argparse
import csv
import glob
import os
import sys
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as ftfeatures  # noqa: E402
import parser as ftparser  # noqa: E402
import rules_baseline  # noqa: E402

warnings.filterwarnings("ignore")
LOW_CONFIDENCE = 0.50


def failed_unit_rows(path):
    """One dict per FLAGGED unit in the report: all parameter values (as in
    training rows) plus identifiers. Returns (rows, n_units, n_unflagged)."""
    parsed = ftparser.load_po_file(path)
    n = parsed["n_units"]
    flagged = ftparser.flagged_unit_indices(parsed)
    base = os.path.basename(path)
    po = base.split("_")[0]
    rows = []
    for i in flagged:
        fp = [p for p, fl in parsed["flags"].items() if i < len(fl) and fl[i]]
        row = {"PO": po, "model_target": "?", "source_file": base, "unit_index": i + 1,
               "flagged_params": ";".join(fp)}
        for name, vals in parsed["id_fields"].items():
            row[name] = vals[i] if i < len(vals) else None
        for p, vals in parsed["params"].items():
            if p.endswith(" []") or p in ("unique_number []", "id_unique_number []"):
                continue
            row[p] = vals[i] if i < len(vals) else None
        rows.append(row)
    return rows, n, n - len(flagged)


def write_temp_units(rows, columns):
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    return tmp


def shap_top(model, X, classes, pred_idx, feature_names, k=3):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    out = []
    for r in range(len(X)):
        if isinstance(sv, list):
            vals = sv[pred_idx[r]][r]
        else:
            vals = sv[r, :, pred_idx[r]]
        top = np.argsort(-np.abs(vals))[:k]
        out.append("; ".join(f"{feature_names[j]} ({'+' if vals[j] >= 0 else '-'}{abs(vals[j]):.2f})"
                             for j in top))
    return out


def selfcheck(args):
    with open(args.train, encoding="latin-1") as f:
        train_cols = list(csv.DictReader(f).fieldnames)
    X_tr, y_tr, g_tr, X_te, y_te, g_te, names = ftfeatures.build_feature_matrix(
        args.train, args.test_units)
    ref = pd.read_csv(args.test_features)
    ref_cols = [c for c in ref.columns if c not in ("PO", "model_target")]
    if list(names) != ref_cols:
        print("PARITY MISMATCH: feature columns differ.")
        print("  only in rebuilt:", [c for c in names if c not in ref_cols][:10])
        print("  only in saved  :", [c for c in ref_cols if c not in names][:10])
        sys.exit(1)
    A = np.array(X_te, dtype=float)
    B = ref[ref_cols].to_numpy(dtype=float)
    if A.shape != B.shape or not np.allclose(A, B, rtol=1e-9, atol=1e-9, equal_nan=True):
        bad = [] if A.shape != B.shape else [ref_cols[j] for j in range(A.shape[1])
                                           if not np.allclose(A[:, j], B[:, j], equal_nan=True)]
        print(f"PARITY MISMATCH: values differ in {bad[:10]}")
        sys.exit(1)
    print(f"PARITY OK: rebuilt features match {args.test_features} "
          f"({A.shape[0]} rows x {A.shape[1]} columns).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="data/processed/train_units.csv")
    ap.add_argument("--report", help="FT report .xlsx, or a folder of them")
    ap.add_argument("--model", default="models/defect_classifier.joblib")
    ap.add_argument("--out", default="reports/predictions.csv")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--test-units")
    ap.add_argument("--test-features")
    ap.add_argument("--no-shap", action="store_true", help="skip the per-unit explanation")
    args = ap.parse_args()

    if args.selfcheck:
        if not (args.test_units and args.test_features):
            raise SystemExit("--selfcheck needs --test-units and --test-features")
        return selfcheck(args)
    if not args.report:
        raise SystemExit("--report is required (or use --selfcheck)")

    files = (sorted(glob.glob(os.path.join(args.report, "**", "*.xlsx"), recursive=True))
             if os.path.isdir(args.report) else [args.report])
    if not files:
        raise SystemExit(f"No .xlsx files found at {args.report}")

    with open(args.train, encoding="latin-1") as f:
        columns = list(csv.DictReader(f).fieldnames)
    for extra in ("PO", "model_target"):
        if extra not in columns:
            columns.append(extra)

    all_rows, summary = [], []
    for path in files:
        try:
            rows, n, n_ok = failed_unit_rows(path)
        except Exception as e:  # noqa: BLE001
            print(f"  could not read {path}: {e}")
            continue
        summary.append((os.path.basename(path), n, len(rows), n_ok))
        all_rows.extend(rows)

    print(f"{'report':<40}{'units':>7}{'failed':>8}{'not flagged':>13}")
    for name, n, nf, ok in summary:
        print(f"{name:<40}{n:>7}{nf:>8}{ok:>13}")
    if not all_rows:
        print("\nNo flagged (failed) units found -- nothing to predict.")
        return

    tmp = write_temp_units(all_rows, columns)
    try:
        _, _, _, X_new, _, _, names = ftfeatures.build_feature_matrix(args.train, tmp)
    finally:
        os.remove(tmp)

    model = joblib.load(args.model)
    Xdf = pd.DataFrame(X_new, columns=names)
    if hasattr(model, "feature_names_in_"):
        Xdf = Xdf[list(model.feature_names_in_)]
    proba = model.predict_proba(Xdf)
    classes = list(model.classes_)
    pred_idx = proba.argmax(axis=1)
    explain = [""] * len(Xdf) if args.no_shap else shap_top(model, Xdf, classes, pred_idx, list(Xdf.columns))

    out_rows = []
    for k, r in enumerate(all_rows):
        order = np.argsort(-proba[k])
        base_row = {c: ("" if r.get(c) is None else str(r.get(c))) for c in r}
        b_family, b_tier, b_why = rules_baseline.classify_unit(base_row)
        conf = float(proba[k][order[0]])
        agree = (classes[order[0]] == b_family)
        review = conf < LOW_CONFIDENCE or not agree
        out_rows.append({
            "source_file": r["source_file"], "unit_index": r["unit_index"],
            "unique_number": r.get("unique_number", ""),
            "flagged_params": r["flagged_params"],
            "ml_prediction": classes[order[0]], "ml_confidence": round(conf, 3),
            "ml_runner_up": f"{classes[order[1]]} ({proba[k][order[1]]:.2f})",
            "ml_top_features": explain[k],
            "baseline_prediction": b_family, "baseline_tier": b_tier, "baseline_reason": b_why,
            "methods_agree": agree, "needs_review": review,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    df = pd.DataFrame(out_rows)
    print(f"\nPredicted root cause for {len(df)} failed sensor(s) -> {args.out}\n")
    print(df["ml_prediction"].value_counts().to_string())
    print(f"\nMethods agree on {int(df['methods_agree'].sum())}/{len(df)}; "
          f"{int(df['needs_review'].sum())} flagged for human review "
          f"(confidence < {LOW_CONFIDENCE:.2f} or the two methods disagree).")


if __name__ == "__main__":
    main()
