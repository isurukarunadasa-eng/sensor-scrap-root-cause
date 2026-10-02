"""
consolidate_categories.py

Collapses the ~110 raw root_cause_category strings in usable_far_rows.csv /
labeled_units.csv (many of them spelling variants of the same defect, or
singleton one-off descriptions) into a small set of defect families a
classifier can actually learn from.

Why
---
With 1,372 labeled rows spread across 110 raw category strings, most
classes have 1-3 examples -- not enough for any model to learn a decision
boundary, and a lot-level train/test split would leave several classes
with zero examples in the test fold.

Families are defined with the SAME failure mechanism grouping already used
in parser.py's CATEGORY_PARAM_RULES (which group categories by which FT
parameters they'd plausibly affect), so the taxonomy here stays consistent
with the logic that tightened the per-unit labels in the first place:

  1. Dimensional / Alignment  - ferrite-cap-shield gap, centering, potting
                                  layer thickness, trimming, pallet fit
  2. Coil                      - coil winding damage, resistance, bridging
  3. ASIC / EEPROM             - ASIC placement/damage, EEPROM fault codes
                                  not already covered by coil
  4. LED                       - LED damage / not lighting
  5. Insulation / Isolation    - insulation tape, shield-to-housing contact
  6. Gluing / Potting Process  - excess/insufficient adhesive, not already
                                  covered by the dimensional family
  7. Flex / PCB / Solder / Wiring - flex PCB damage, solder joints, cable,
                                  bridging, wire detachment
  8. Other                     - doesn't match any family (process/test
                                  equipment issues, "further investigation",
                                  genuinely unclear one-offs)

This script does NOT silently relabel anything. It reads the distinct
category strings + counts from a FAR-derived CSV and writes a review table
(category, count, assigned_family) to category_mapping.csv -- intended to
be eyeballed and corrected before being applied to the real dataset.

Usage
-----
    python src/consolidate_categories.py \
        --in data/processed/labeled_units.csv \
        --out data/processed/category_mapping.csv
"""

import argparse
import csv
import re
from collections import Counter

# Ordered (first match wins), same precedence logic as parser.py's
# CATEGORY_PARAM_RULES, extended to produce a human-readable family name
# instead of a parameter list, and extended with a couple of extra
# patterns (glue/loctite) that parser.py didn't need but this does.
FAMILY_RULES = [
    ("Dimensional / Alignment",
     r"ferrite|gap|cap\b|shield|pallet|centered|trimming|distance|potting layer|misalign"),
    ("Coil",
     r"\bcoil\b"),
    ("ASIC / EEPROM",
     r"asic|eeprom"),
    ("LED",
     r"\bled\b"),
    ("Insulation / Isolation",
     r"insulation|isolation|contact with housing"),
    ("Gluing / Potting Process",
     r"glu|loctite|\bpotting\b"),
    ("Flex / PCB / Solder / Wiring",
     r"flex|pcb|solder|bridge|\bwire\b|cable|connector|pcba"),
]


def classify(category: str) -> str:
    text = category.lower()
    for family, pattern in FAMILY_RULES:
        if re.search(pattern, text):
            return family
    return "Other"


def read_categories(path: str) -> Counter:
    counts = Counter()
    with open(path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        if "root_cause_category" not in reader.fieldnames:
            raise SystemExit(
                f"'root_cause_category' column not found in {path}. "
                f"Columns present: {reader.fieldnames}"
            )
        for row in reader:
            cat = (row.get("root_cause_category") or "").strip()
            if cat:
                counts[cat] += 1
    return counts


MINOR_BUCKET_NAME = "Other / Minor Defects"


def apply_to_csv(in_path: str, out_path: str, min_family_count: int):
    """Write a full copy of in_path with two columns appended:

      defect_family - the full 7-family grouping from classify(), untouched,
                       so no information is ever lost.
      model_target   - the column to actually train on: same as
                       defect_family, EXCEPT any family with fewer than
                       min_family_count rows (in this run of the data) is
                       folded into "Other / Minor Defects".

    Because the min-count check is recomputed from whatever rows are in
    in_path every time this runs, a family that currently has too few
    examples (e.g. LED, Gluing/Potting, Insulation/Isolation today)
    automatically "graduates" into its own model_target class on a later
    run, once enough new FAR/FT data pushes it past the threshold -- no
    code change needed, just rerun this script on the updated
    labeled_units.csv.
    """
    import os

    with open(in_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) + ["defect_family", "model_target"]
        rows = []
        for row in reader:
            family = classify((row.get("root_cause_category") or "").strip())
            row["defect_family"] = family
            rows.append(row)

    family_counts = Counter(r["defect_family"] for r in rows)
    for row in rows:
        family = row["defect_family"]
        if family_counts[family] < min_family_count:
            row["model_target"] = MINOR_BUCKET_NAME
        else:
            row["model_target"] = family

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    target_totals = Counter(r["model_target"] for r in rows)
    print(f"Applied defect_family + model_target (min {min_family_count} rows/class) "
          f"to {len(rows)} rows -> {out_path}")
    for family, n in target_totals.most_common():
        below = " (merged from small families)" if family == MINOR_BUCKET_NAME else ""
        print(f"  {n:4d}  {family}{below}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                     help="CSV with a root_cause_category column "
                          "(usable_far_rows.csv or labeled_units.csv)")
    ap.add_argument("--out", default="data/processed/category_mapping.csv",
                     help="Where to write the review table (category, count, family)")
    ap.add_argument("--apply-out", default=None,
                     help="If given, also write a full copy of --in with "
                          "defect_family and model_target columns appended "
                          "(e.g. data/processed/labeled_units_with_family.csv)")
    ap.add_argument("--min-family-count", type=int, default=20,
                     help="Families with fewer rows than this (recomputed "
                          "from --in every run) are merged into "
                          "'Other / Minor Defects' for model_target. "
                          "Default 20.")
    args = ap.parse_args()

    counts = read_categories(args.in_path)

    rows = []
    family_totals = Counter()
    for cat, n in counts.items():
        family = classify(cat)
        family_totals[family] += n
        rows.append((cat, n, family))

    rows.sort(key=lambda r: (-r[1], r[0]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["root_cause_category", "count", "assigned_family"])
        writer.writerows(rows)

    print(f"{len(counts)} distinct category strings -> {len(family_totals)} families")
    print(f"Written to {args.out}\n")
    print("Family totals (row count, i.e. weighted by how often each category occurs):")
    for family, n in family_totals.most_common():
        print(f"  {n:4d}  {family}")

    other_rows = [r for r in rows if r[2] == "Other"]
    if other_rows:
        print(f"\n'Other' bucket -- {sum(r[1] for r in other_rows)} rows across "
              f"{len(other_rows)} distinct categories -- worth a manual look:")
        for cat, n, _ in other_rows:
            print(f"  {n:3d}  {cat}")

    if args.apply_out:
        print()
        apply_to_csv(args.in_path, args.apply_out, args.min_family_count)


if __name__ == "__main__":
    main()
