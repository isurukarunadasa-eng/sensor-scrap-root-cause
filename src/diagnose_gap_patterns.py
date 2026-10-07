"""
diagnose_gap_patterns.py

One-off diagnostic to re-derive rules_baseline.py's TEST_ORDER/GAP_RULES
from the REAL training data, instead of the small 37-row leftover sample
they were provisionally built from.

Why this has to run on train_units.csv only
---------------------------------------------
The baseline's gap-zone rules are a form of "fitting" to data, exactly like
the ML model's parameters are -- they were chosen by looking at which
root_cause_category values go with which reach/no-reach pattern. If we
derive them from the same rows we then score the baseline on, that's
circular (train-on-test leakage) and the accuracy number would be
meaningless, same as it would be for the Random Forest. So this script
only ever looks at train_units.csv. Once GAP_RULES is updated from its
output, rules_baseline.py gets scored on test_units.csv, which this script
never touches.

What it prints
---------------
For each distinct reach/no-reach pattern across the 11 wired flags, the
root_cause_category breakdown of units with that pattern -- the same
report Claude used on the small sample to find the three clean zones
(ASIC-loose / coil-damage / "sequence completed"). Paste the FULL output
back so GAP_RULES can be re-derived from the real, full-size clusters
instead of 37 rows.

Usage
-----
    python src/diagnose_gap_patterns.py --in data/processed/train_units.csv
"""

import argparse
import csv
from collections import Counter, defaultdict

COLS = [
    "IcPreconfig [0/1]", "5V4trim [0/1]", "TemperatureComp [0/1]",
    "CalibrSP1Hys [0/1]", "CalibrSP1 [0/1]", "CalibrSP2Hys [0/1]", "CalibrSP2 [0/1]",
    "CalibrSP3Hys [0/1]", "CalibrSP3 [0/1]", "YellowLed [0/1]", "TestIsolation [0/1]",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--category-col", default="root_cause_category",
                     help="Use 'model_target' instead if you want family-level "
                          "clusters rather than raw category text (default: "
                          "root_cause_category, since that's the more specific "
                          "label to validate gap zones against).")
    args = ap.parse_args()

    with open(args.in_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    missing = [c for c in COLS if c not in fieldnames]
    if missing:
        raise SystemExit(f"Columns not found in {args.in_path}: {missing}\n"
                          f"Columns present: {fieldnames}")
    if args.category_col not in fieldnames:
        raise SystemExit(f"'{args.category_col}' not found in {args.in_path}.")

    by_pattern = defaultdict(Counter)
    pattern_counts = Counter()
    for r in rows:
        pattern = "".join("1" if (r.get(c) or "").strip() else "0" for c in COLS)
        pattern_counts[pattern] += 1
        by_pattern[pattern][(r.get(args.category_col) or "").strip()] += 1

    print(f"{len(rows)} rows from {args.in_path}")
    print(f"Columns, in order (position 0..10): {COLS}\n")
    print(f"{len(pattern_counts)} distinct reach/no-reach patterns found.\n")

    for pattern, n in pattern_counts.most_common():
        # Where does the gap start? (first '0' position, or 'none' if all '1')
        gap_at = pattern.find("0")
        gap_desc = f"gap after reaching index {gap_at - 1} ({COLS[gap_at - 1]})" if gap_at > 0 \
            else ("gap at index 0 (nothing reached)" if gap_at == 0 else "no gap -- full sequence reached")
        print(f"Pattern {pattern}  ({n} rows, {gap_desc})")
        for cat, cnt in by_pattern[pattern].most_common():
            print(f"    {cnt:4d}  {cat}")
        print()


if __name__ == "__main__":
    main()
