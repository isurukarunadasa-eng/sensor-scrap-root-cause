"""
features.py

Builds the feature matrix for the defect-family classifier from
train_units.csv / test_units.csv.

What is and isn't a feature
----------------------------
labeled_units.csv (and its train/test split) has 4 kinds of columns:

1. Identifiers / metadata - CC_reference, PO, article_no, source_file,
   unique_number, id_unique_number, lot_major, lot_minor. Never features:
   they identify a unit/lot, they don't describe its electrical behaviour.

2. Label-derived text - root_cause_category, root_cause_narrative,
   defect_family, model_target. model_target IS the label (y); the others
   are the human-written text that model_target was derived from, so they
   can't be inputs either -- that would be telling the model the answer.

3. LEAKY derived columns - severity_score, n_flagged_params,
   flagged_params, flagged_params_matched. These look like real signal
   but parser.py computed them by first looking at root_cause_category,
   deciding which FT parameters were plausible for THAT family (via
   CATEGORY_PARAM_RULES), and only then checking which of those matched.
   So flagged_params_matched effectively already encodes which family a
   row was tightened against. Using these as inputs would let the model
   learn the labeling pipeline's own logic instead of the actual
   electrical signature of the defect, which would make test-set accuracy
   meaningless (and wouldn't work at inference time anyway, since a brand
   new unit has no root_cause_category yet -- that's the whole point of
   building this classifier).

4. Real FT measurement columns - everything else (DistUnom [mm],
   HystUnom [%], ILedOn [mA], TestIsolation [0/1], the Calibr* flags,
   etc.). These are the actual per-unit test results and are the ONLY
   columns this module turns into features.

Missing values
---------------
A blank cell means that parameter was skipped ("--") for that unit during
the FT run (not every parameter applies to every article/test program).
These are median-imputed per column. Imputation statistics are fit on the
TRAIN split only and then applied to the test split, so no information
about the test set's own distribution leaks into training.

Usage
-----
As a library (how src/model.py should use it):

    from features import build_feature_matrix
    X_train, y_train, groups_train, X_test, y_test, groups_test, feature_names = \\
        build_feature_matrix("data/processed/train_units.csv",
                              "data/processed/test_units.csv")

As a script, to materialize the matrices for inspection:

    python src/features.py \\
        --train data/processed/train_units.csv \\
        --test data/processed/test_units.csv \\
        --out-dir data/processed
"""

import argparse
import csv
import os

NON_FEATURE_COLUMNS = {
    # identifiers / metadata
    "CC_reference", "PO", "article_no", "source_file",
    "unique_number", "id_unique_number", "lot_major", "lot_minor",
    # label + label-derived text
    "root_cause_category", "root_cause_narrative", "defect_family", "model_target",
    # leaky: derived FROM the label via parser.py's category-matching logic
    "severity_score", "n_flagged_params", "flagged_params", "flagged_params_matched",
}

LABEL_COLUMN = "model_target"
GROUP_COLUMN = "PO"


def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="latin-1") as f:
        return list(csv.DictReader(f))


def _feature_columns(fieldnames: list[str]) -> list[str]:
    cols = [c for c in fieldnames if c not in NON_FEATURE_COLUMNS]
    if not cols:
        raise SystemExit(
            "No feature columns found -- check that the input CSV has the "
            "expected FT parameter columns and hasn't already been filtered."
        )
    return cols


def _to_float(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _median(values: list[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def build_feature_matrix(train_path: str, test_path: str):
    """Returns (X_train, y_train, groups_train, X_test, y_test, groups_test,
    feature_names). X_* are list[list[float]], y_* are list[str],
    groups_* are list[str] (PO, for anyone who wants to double-check split
    integrity or do further group-aware work downstream)."""

    train_rows = _read_csv(train_path)
    test_rows = _read_csv(test_path)

    fieldnames = list(train_rows[0].keys()) if train_rows else []
    feature_names = _feature_columns(fieldnames)

    if LABEL_COLUMN not in fieldnames:
        raise SystemExit(
            f"'{LABEL_COLUMN}' column not found in {train_path}. "
            f"Run consolidate_categories.py --apply-out first."
        )

    # Raw (possibly-missing) numeric values for every row.
    def extract(rows):
        raw = [[_to_float(row.get(c, "")) for c in feature_names] for row in rows]
        y = [row[LABEL_COLUMN] for row in rows]
        groups = [row.get(GROUP_COLUMN, "") for row in rows]
        return raw, y, groups

    train_raw, y_train, groups_train = extract(train_rows)
    test_raw, y_test, groups_test = extract(test_rows)

    # Fit per-column medians on TRAIN only.
    medians = []
    for col_idx in range(len(feature_names)):
        col_values = [row[col_idx] for row in train_raw if row[col_idx] is not None]
        medians.append(_median(col_values))

    def impute(raw):
        return [
            [v if v is not None else medians[i] for i, v in enumerate(row)]
            for row in raw
        ]

    X_train = impute(train_raw)
    X_test = impute(test_raw)

    return X_train, y_train, groups_train, X_test, y_test, groups_test, feature_names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    X_train, y_train, groups_train, X_test, y_test, groups_test, feature_names = \
        build_feature_matrix(args.train, args.test)

    os.makedirs(args.out_dir, exist_ok=True)

    def write(path, X, y, groups):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["PO", "model_target"] + feature_names)
            for g, label, row in zip(groups, y, X):
                writer.writerow([g, label] + row)

    train_out = os.path.join(args.out_dir, "train_features.csv")
    test_out = os.path.join(args.out_dir, "test_features.csv")
    write(train_out, X_train, y_train, groups_train)
    write(test_out, X_test, y_test, groups_test)

    print(f"{len(feature_names)} feature columns (raw FT parameters only -- "
          f"no label-derived columns included):")
    for c in feature_names:
        print(f"  {c}")
    print(f"\nExcluded as identifiers/metadata/label-derived/leaky: "
          f"{sorted(NON_FEATURE_COLUMNS)}")
    print(f"\nTrain: {len(X_train)} rows -> {train_out}")
    print(f"Test:  {len(X_test)} rows -> {test_out}")


if __name__ == "__main__":
    main()
