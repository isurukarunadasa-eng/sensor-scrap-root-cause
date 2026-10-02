"""
train_test_split.py

Splits labeled_units_with_family.csv into train/test sets at the PO (lot)
level, not the row level.

Why not a plain random row split
---------------------------------
Several labeled units can come from the same PO-linked FT lot (e.g. a FAR
row with qty=3 contributes 3 unit rows, all from the same PO; different
FAR rows can also happen to share a PO). Units from the same lot are
correlated -- same assembly batch, same operator, often very similar FT
signatures. A row-level random split can put some units from a lot in
train and others from the SAME lot in test, which leaks information and
makes test accuracy look better than the model would actually achieve on
a genuinely new, unseen lot.

So every unit belonging to a given PO is kept together, entirely in train
or entirely in test.

How the split is chosen
------------------------
Simple random group split can still land you with a test set that's
badly skewed on model_target (e.g. almost no Coil examples), since POs
vary a lot in both size and which defect they represent. Instead this
uses a greedy balanced allocation:

  1. Group units by PO; compute each PO's class-count vector over
     model_target.
  2. Shuffle POs (seeded, for reproducibility) and walk through them
     once, assigning each PO to whichever of {train, test} is currently
     furthest below its target share for that PO's most-represented
     class -- i.e. greedily keep both splits' class proportions close to
     the requested test_size.
  3. Report the resulting class balance for both splits so you can see
     exactly how close it got (small classes won't split perfectly, by
     nature of being small and lot-grouped).

Usage
-----
    python src/train_test_split.py \
        --in data/processed/labeled_units_with_family.csv \
        --test-size 0.2 \
        --seed 42 \
        --out-dir data/processed
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def greedy_group_split(rows: list[dict], test_size: float, seed: int):
    # Group row indices by PO.
    po_rows = defaultdict(list)
    for i, row in enumerate(rows):
        po = row.get("PO", "").strip()
        po_rows[po].append(i)

    # Each PO's class-count vector.
    po_class_counts = {}
    for po, idxs in po_rows.items():
        po_class_counts[po] = Counter(rows[i]["model_target"] for i in idxs)

    overall_counts = Counter(r["model_target"] for r in rows)
    total = sum(overall_counts.values())
    target_test_counts = {c: n * test_size for c, n in overall_counts.items()}

    pos = list(po_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(pos)

    train_count = Counter()
    test_count = Counter()
    train_pos, test_pos = [], []

    for po in pos:
        counts = po_class_counts[po]
        # How "urgently" does test need more of this PO's classes, vs train?
        # Score = sum over classes in this PO of (remaining room in test
        # towards its target) minus (remaining room in train towards ITS
        # target, scaled by the train/test ratio). Positive => send to test.
        train_target_scale = (1 - test_size) / test_size if test_size > 0 else float("inf")
        score = 0.0
        for cls, n in counts.items():
            test_room = target_test_counts[cls] - test_count[cls]
            train_room = (target_test_counts[cls] * train_target_scale) - train_count[cls]
            score += (test_room - train_room)

        if score > 0:
            test_pos.append(po)
            test_count.update(counts)
        else:
            train_pos.append(po)
            train_count.update(counts)

    train_idx = [i for po in train_pos for i in po_rows[po]]
    test_idx = [i for po in test_pos for i in po_rows[po]]

    return train_idx, test_idx, train_pos, test_pos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    rows, fieldnames = read_rows(args.in_path)
    if "PO" not in fieldnames:
        raise SystemExit(f"'PO' column not found in {args.in_path}. Columns: {fieldnames}")
    if "model_target" not in fieldnames:
        raise SystemExit(
            f"'model_target' column not found in {args.in_path}. "
            f"Run consolidate_categories.py --apply-out on this file first."
        )

    train_idx, test_idx, train_pos, test_pos = greedy_group_split(
        rows, args.test_size, args.seed
    )

    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_units.csv")
    test_path = os.path.join(args.out_dir, "test_units.csv")

    for path, data in [(train_path, train_rows), (test_path, test_rows)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    # Sanity check: no PO appears in both splits.
    overlap = set(train_pos) & set(test_pos)
    assert not overlap, f"Leakage! These POs ended up in both splits: {overlap}"

    print(f"POs: {len(train_pos)} train, {len(test_pos)} test "
          f"({len(test_pos) / (len(train_pos) + len(test_pos)):.1%} of POs in test)")
    print(f"Rows: {len(train_rows)} train, {len(test_rows)} test "
          f"({len(test_rows) / len(rows):.1%} of rows in test)")
    print(f"No PO overlap between train/test: confirmed\n")

    print(f"{'model_target':<30} {'train':>8} {'test':>8} {'test %':>8}")
    all_classes = sorted(set(r["model_target"] for r in rows))
    for cls in all_classes:
        tr = sum(1 for r in train_rows if r["model_target"] == cls)
        te = sum(1 for r in test_rows if r["model_target"] == cls)
        pct = te / (tr + te) * 100 if (tr + te) else 0
        print(f"{cls:<30} {tr:>8} {te:>8} {pct:>7.1f}%")

    print(f"\nWritten: {train_path}, {test_path}")


if __name__ == "__main__":
    main()
