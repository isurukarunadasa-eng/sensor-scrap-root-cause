"""
src/parser.py

Turns the raw Final Tester (FT) .xlsx exports into a tidy, per-unit table,
and (when a matched FAR row is available) attaches the confirmed root-cause
label to the units that actually show an out-of-spec flag.

Background
----------
An FT export sheet is wide and semi-structured: row 1-7 are a header block
(designation, article no, tester name, ...), row 9 is "Measure" with a
column index per tested unit, and every row after that is one test
parameter, one column per unit. A cell is either:
    - a plain number                  -> pass
    - "H<value>" or "L<value>"        -> out of spec, too high / too low
    - "!-<code>"                      -> hard ASIC/EEPROM fault code
    - "--"                            -> test skipped

This script:
  1. Parses one FT file into a tidy long-format table: one row per unit,
     one column per parameter, with flagged cells split into a clean
     numeric value plus a separate flag marker.
  2. Given the matched-rows table produced by match_far_to_ft.py
     (data/processed/usable_far_rows.csv), joins each confirmed FAR row to
     the FT units in its linked PO-named lot file, and keeps only the
     units whose flagged parameter(s) are *physically consistent* with
     that row's root-cause category (see "Tightening" below) -- not just
     any flag in the lot.
  3. Writes three outputs:
       - data/processed/labeled_units.csv   -- per-unit rows with a
         root_cause_category label attached (training data)
       - data/processed/unlabelable_rows.csv -- FAR rows whose matched lot
         had *no* flagged units at all (see note below)
       - data/processed/ambiguous_rows.csv  -- FAR rows whose matched lot
         had flagged units, but none of them matched an expected parameter
         family for that category (flag present, but not the *right kind*
         of flag) -- logged rather than guessed at

Tightening: why "any flag in the lot" isn't enough
----------------------------------------------------
The first version of this script labeled every flagged unit in a matched
lot with that row's root cause, regardless of *which* parameter was
flagged. That's too loose: a lot can have several units flagged for
unrelated reasons (e.g. an EEPROM fault on one unit, an isolation fault on
another) and only one of them is plausibly the confirmed defect. To bond
the label to the right evidence:

  1. When the category text itself names an EEPROM/ASIC fault code (many
     rows already do, e.g. "Eeprom 7015 - Coil damage", "Eeprom 7022 -
     ASIC Loose") -- require the unit's eeprom_integrity flag to carry
     that exact code. This is the tightest, most direct evidence
     available: the analyst effectively already pointed at the parameter.
  2. Otherwise, match the category's keywords (ferrite/gap/cap/shield/
     distance -> sensing-distance parameters; coil/ASIC -> eeprom_integrity;
     LED -> ILedOn; insulation/isolation -> TestIsolation; current/ASIC
     heated -> current/saturation parameters) against a keyword ->
     parameter-family table (CATEGORY_PARAM_RULES below) and keep only
     units flagged on a parameter in the matching family.
  3. A unit is kept only if at least one of its flagged parameters
     satisfies (1) or (2) -- an unrelated flag on the same unit doesn't
     disqualify it, but it also isn't treated as evidence for this label.

Why some rows can't be labeled this way
----------------------------------------
Verified directly on real data (PO 2004884, 11 confirmed "flex damage"
rows): a lot can have zero units with any FT flag, because the confirmed
defect was mechanical/visual (a cracked ferrite, a bent flex cable) and
never produced an electrical symptom the tester could catch. For those
rows, PO/lot linkage narrows things down to the right batch, but doesn't
single out a unit -- there's nothing in the FT data for a classifier to
learn from. Those rows are written to unlabelable_rows.csv rather than
silently dropped or force-labeled, so the gap is visible and can be cited
directly as a risk/limitation.

Usage
-----
    python src/parser.py \
        --matched-rows data/processed/usable_far_rows.csv \
        --po-dir data/raw/po_reports \
        --out-dir data/processed
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import openpyxl

FLAG_RE = re.compile(r"^([HL!])(-?[\d.]+)$")

EEPROM_CODE_RE = re.compile(r"eeprom\s*(-?\d{3,5})", re.IGNORECASE)

# Keyword (matched against the lowercased root_cause_category text) ->
# set of parameter-name substrings that are physically plausible evidence
# for that family of defect. Checked in order; first match wins. This is
# a coarse, defensible mapping, not a certified fault-tree -- documented
# here so it can be reviewed/extended, and every match is traceable back
# to which parameter fired (see "flagged_params_matched" in the output).
CATEGORY_PARAM_RULES = [
    # mechanical/geometry defects -> sensing-distance parameters
    (r"ferrite|gap|cap|shield|pallet|centered|trimming|distance|potting layer",
     ["UDist", "IDist", "DistUnom", "HystUnom"]),
    # coil defects -> no dedicated coil-resistance test exists in this FT
    # format; coil faults surface as EEPROM integrity faults (matches the
    # FAR log's own "Eeprom 7015 - Coil damage" labeling)
    (r"coil",
     ["eeprom_integrity"]),
    # ASIC defects -> EEPROM integrity fault
    (r"asic|eeprom",
     ["eeprom_integrity"]),
    # LED defects -> LED drive current
    (r"led",
     ["ILedOn"]),
    # insulation / isolation / solder-contact-with-housing defects
    (r"insulation|isolation|solder contact with housing",
     ["TestIsolation"]),
    # high-current / overheating / component damage
    (r"high current|heated|component damage|surge",
     ["IccEnclUmax", "UsatUmaxIchmax", "IfuiteUmax", "IcInternalTemperature"],),
    # flex / PCB / solder-bridge defects can disrupt any signal path --
    # broadest family, used as a fallback for this keyword group only
    (r"flex|pcb|solder|bridge|wire|cable",
     ["UDist", "IDist", "ILedOn", "eeprom_integrity"]),
]


def expected_param_families(category: str) -> list[str]:
    cat = (category or "").lower()
    for pattern, families in CATEGORY_PARAM_RULES:
        if re.search(pattern, cat):
            return families
    return []  # no rule matched -> nothing counts as consistent evidence


def matching_flagged_params(category: str, flagged_params: list[str], parsed: dict, unit_idx: int) -> list[str]:
    """
    Of this unit's flagged parameters, return the ones that count as
    consistent evidence for `category` -- either an exact EEPROM code
    match (when the category text names one) or a parameter-family match.
    """
    code_match = EEPROM_CODE_RE.search(category or "")
    if code_match:
        target_code = float(code_match.group(1))
        if not target_code < 0:
            target_code = -target_code  # FAR text gives "7022", FT stores "-7022"
        eeprom_values = parsed["params"].get("eeprom_integrity []")
        if eeprom_values and unit_idx < len(eeprom_values):
            if eeprom_values[unit_idx] == target_code:
                return ["eeprom_integrity []"]
        return []  # category names a specific code but this unit doesn't carry it

    families = expected_param_families(category)
    if not families:
        return []
    return [p for p in flagged_params if any(fam in p for fam in families)]


def parse_ft_sheet(ws) -> dict:
    """
    Parse a single FT worksheet into:
      {
        "article": str | None,
        "units": [unit_index, ...],           # 1-based column order
        "params": {param_name: [values...]},  # numeric, per unit
        "flags":  {param_name: [flag|None...]} # "H"/"L"/"!" per unit, or None
        "id_fields": {"unique_number": [...], "id_unique_number": [...], "lot_major": [...]}
      }
    """
    article = None
    header_row = None
    params = {}
    flags = {}
    n_units = 0

    for r, row in enumerate(ws.iter_rows(min_row=1, max_row=300, values_only=True), start=1):
        if not row or row[0] is None:
            continue
        key = str(row[0])

        if key.startswith("Article no"):
            m = re.search(r"Article no:\s*([0-9-]+)", key)
            if m:
                article = m.group(1)
            continue

        if key == "Measure":
            header_row = r
            n_units = sum(1 for v in row[1:] if v is not None)
            continue

        if header_row is None:
            continue  # still in the header block, not a parameter row yet

        values = []
        unit_flags = []
        for v in row[1 : n_units + 1]:
            if isinstance(v, str):
                m = FLAG_RE.match(v.strip())
                if m:
                    unit_flags.append(m.group(1))
                    try:
                        values.append(float(m.group(2)))
                    except ValueError:
                        values.append(None)
                    continue
                if v.strip() in ("--", ""):
                    unit_flags.append(None)
                    values.append(None)
                    continue
                # unflagged string (shouldn't normally happen) -- keep as-is
                unit_flags.append(None)
                values.append(v)
            else:
                unit_flags.append(None)
                values.append(v)

        params[key] = values
        flags[key] = unit_flags

    id_fields = {}
    for id_name in ("unique_number", "id_unique_number", "lot_major", "lot_minor"):
        param_key = f"{id_name} []"
        if param_key in params:
            id_fields[id_name] = params[param_key]

    return {
        "article": article,
        "n_units": n_units,
        "params": params,
        "flags": flags,
        "id_fields": id_fields,
    }


def flagged_unit_indices(parsed: dict) -> list[int]:
    """Indices (0-based, into the per-unit arrays) of units with >=1 flag."""
    n = parsed["n_units"]
    flagged = []
    for i in range(n):
        if any(flags[i] for flags in parsed["flags"].values() if i < len(flags)):
            flagged.append(i)
    return flagged


def load_po_file(path: str) -> dict:
    """Parse the first sheet of a PO-named FT file (sheet '<PO>_1' has the
    original flag strings; the '_2' sheet is a cleaned duplicate with the
    same values as numbers -- we only need the first)."""
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sheet_name = next((s for s in wb.sheetnames if s.endswith("_1")), wb.sheetnames[0])
    parsed = parse_ft_sheet(wb[sheet_name])
    wb.close()
    return parsed


def article_number_from_filename(path: str) -> str | None:
    base = os.path.basename(path)
    m = re.match(r"^(\d{3})[-_ ]?(\d{3})[-_ ]?(\d{3})", base)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


class ArticleBaselines:
    """
    Per-article, per-parameter mean/stdev computed from the *bulk* FT
    exports (data/raw/ft_exports/), using only that article's own passing
    (unflagged) measurements. This gives a real population to score
    against, rather than treating the tester's H/L string as the only
    signal -- important because the same absolute deviation can be
    routine for one parameter/article and extreme for another.

    Built lazily and cached per article, since a bulk export can hold
    thousands of units and there's no need to reparse it for every FAR
    row that shares the same article.
    """

    def __init__(self, ft_dir: str):
        self.ft_dir = ft_dir
        self._file_index = None
        self._cache: dict[str, dict] = {}

    def _index_files(self):
        if self._file_index is not None:
            return
        self._file_index = defaultdict(list)
        for f in glob.glob(os.path.join(self.ft_dir, "**", "*.xlsx"), recursive=True):
            art = article_number_from_filename(f)
            if art:
                self._file_index[art].append(f)

    def get(self, article: str) -> dict:
        """Returns {param_name: (mean, stdev)} using only unflagged values."""
        if article in self._cache:
            return self._cache[article]

        self._index_files()
        sums = defaultdict(float)
        sumsq = defaultdict(float)
        counts = defaultdict(int)

        for path in self._file_index.get(article, []):
            try:
                wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
                for sheet_name in wb.sheetnames:
                    parsed = parse_ft_sheet(wb[sheet_name])
                    for p, values in parsed["params"].items():
                        if p.endswith(" []"):
                            continue  # ID/config fields, not measurements
                        flags = parsed["flags"].get(p, [])
                        for i, v in enumerate(values):
                            if v is None or not isinstance(v, (int, float)):
                                continue
                            if i < len(flags) and flags[i]:
                                continue  # skip flagged (not "normal") values
                            sums[p] += v
                            sumsq[p] += v * v
                            counts[p] += 1
                wb.close()
            except Exception:
                continue  # a malformed bulk file shouldn't block baseline building

        stats = {}
        for p, n in counts.items():
            if n < 5:
                continue  # too few passing samples for a meaningful baseline
            mean = sums[p] / n
            variance = max(sumsq[p] / n - mean * mean, 0.0)
            stats[p] = (mean, variance ** 0.5)

        self._cache[article] = stats
        return stats


def severity_score(param: str, value, baselines: dict) -> float:
    """|z-score| of `value` for `param` against the article's own baseline.
    Falls back to 0 (no information) when the baseline or value is missing,
    so it never crashes ranking -- it just doesn't add evidence."""
    if value is None or not isinstance(value, (int, float)):
        return 0.0
    stat = baselines.get(param)
    if not stat:
        return 0.0
    mean, std = stat
    if std == 0:
        return 0.0
    return abs((value - mean) / std)


def build_labeled_units(matched_rows_path: str, po_dir: str, ft_dir: str | None = None):
    # index available PO files by PO number (a PO can have >1 tester/lot file)
    po_files = defaultdict(list)
    for f in glob.glob(os.path.join(po_dir, "**", "*.xlsx"), recursive=True):
        po = os.path.basename(f).split("_")[0]
        po_files[po].append(f)

    with open(matched_rows_path, encoding="latin-1") as f:
        far_rows = list(csv.DictReader(f))

    baselines = ArticleBaselines(ft_dir) if ft_dir else None

    labeled_rows = []
    unlabelable_rows = []
    ambiguous_rows = []
    parse_errors = []

    for far_row in far_rows:
        po = far_row["PO"]
        files = po_files.get(po, [])
        if not files:
            continue  # not covered by the PO data on hand yet

        article_baseline = baselines.get(far_row["article_no"]) if baselines else {}

        any_flagged = False
        candidates = []  # units consistent with this category, before qty ranking
        for path in files:
            try:
                parsed = load_po_file(path)
            except Exception as e:
                parse_errors.append({"PO": po, "file": path, "error": str(e)})
                continue

            flagged = flagged_unit_indices(parsed)
            if flagged:
                any_flagged = True

            for i in flagged:
                flagged_params = [
                    p for p, fl in parsed["flags"].items() if i < len(fl) and fl[i]
                ]
                consistent = matching_flagged_params(
                    far_row["root_cause_category"], flagged_params, parsed, i
                )
                if not consistent:
                    continue  # flagged, but not on a parameter this category would explain

                # severity = worst |z-score|, against this article's own
                # baseline, among the parameters that count as evidence
                severity = max(
                    (
                        severity_score(p, parsed["params"][p][i], article_baseline)
                        for p in consistent
                        if p in parsed["params"] and i < len(parsed["params"][p])
                    ),
                    default=0.0,
                )

                unit_row = {
                    "CC_reference": far_row["CC_reference"],
                    "PO": po,
                    "article_no": far_row["article_no"],
                    "source_file": os.path.basename(path),
                    "root_cause_category": far_row["root_cause_category"],
                    "root_cause_narrative": far_row["root_cause_narrative"],
                    "severity_score": round(severity, 3),
                }
                for id_name, values in parsed["id_fields"].items():
                    unit_row[id_name] = values[i] if i < len(values) else None

                unit_row["n_flagged_params"] = len(flagged_params)
                unit_row["flagged_params"] = ";".join(flagged_params)
                unit_row["flagged_params_matched"] = ";".join(consistent)

                # full parameter vector for this unit
                for p, values in parsed["params"].items():
                    if p.endswith(" []") or p in (
                        "unique_number []", "id_unique_number []"
                    ):
                        continue  # ID/config fields, not test features
                    unit_row[p] = values[i] if i < len(values) else None

                candidates.append(unit_row)

        any_consistent_unit = bool(candidates)

        # Keep only the top-`qty` most severe candidates: the FAR row
        # confirms `qty` physical units for this narrative, so once a
        # per-article baseline is available there is no reason to label
        # every consistent-but-mild flag -- the worst-deviating ones are
        # the most defensible match to "this is probably the confirmed
        # unit". Without a baseline (--ft-dir not given), severity can't
        # be computed, so fall back to the looser behaviour: keep every
        # consistent unit rather than narrowing on an arbitrary order.
        if candidates:
            if baselines is not None:
                try:
                    qty = int(float(far_row.get("qty") or 1))
                except (TypeError, ValueError):
                    qty = 1
                qty = max(qty, 1)
                candidates.sort(key=lambda r: r["severity_score"], reverse=True)
                labeled_rows.extend(candidates[:qty])
            else:
                labeled_rows.extend(candidates)

        if not any_flagged:
            unlabelable_rows.append(
                {
                    "CC_reference": far_row["CC_reference"],
                    "PO": po,
                    "article_no": far_row["article_no"],
                    "root_cause_category": far_row["root_cause_category"],
                    "reason": "no unit in the linked PO lot showed any FT flag",
                }
            )
        elif not any_consistent_unit:
            ambiguous_rows.append(
                {
                    "CC_reference": far_row["CC_reference"],
                    "PO": po,
                    "article_no": far_row["article_no"],
                    "root_cause_category": far_row["root_cause_category"],
                    "reason": "lot had flagged units, but none on a parameter "
                              "consistent with this category (see CATEGORY_PARAM_RULES)",
                }
            )

    return labeled_rows, unlabelable_rows, ambiguous_rows, parse_errors


def write_csv(rows: list[dict], path: str):
    if not rows:
        # still write an empty file with no rows so downstream steps don't break
        open(path, "w").close()
        return
    # union of all keys across rows (parameter sets can differ slightly by article)
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matched-rows", required=True, help="usable_far_rows.csv from match_far_to_ft.py")
    ap.add_argument("--po-dir", required=True, help="Directory of PO-linked FT files")
    ap.add_argument(
        "--ft-dir",
        default=None,
        help="Directory of bulk FT export files, used to build per-article "
             "baselines for severity ranking. Without this, every consistent "
             "flagged unit is kept (no qty-based narrowing).",
    )
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    labeled_rows, unlabelable_rows, ambiguous_rows, parse_errors = build_labeled_units(
        args.matched_rows, args.po_dir, args.ft_dir
    )

    write_csv(labeled_rows, os.path.join(args.out_dir, "labeled_units.csv"))
    write_csv(unlabelable_rows, os.path.join(args.out_dir, "unlabelable_rows.csv"))
    write_csv(ambiguous_rows, os.path.join(args.out_dir, "ambiguous_rows.csv"))

    print(f"Labeled per-unit rows written: {len(labeled_rows)}")
    print(f"FAR rows with no FT signal at all (unlabelable): {len(unlabelable_rows)}")
    print(f"FAR rows with flags, but none matching this category (ambiguous): {len(ambiguous_rows)}")
    if parse_errors:
        print(f"Files that failed to parse: {len(parse_errors)}")
        for e in parse_errors[:10]:
            print(f"  - {e['file']}: {e['error']}")
    print(f"Written to {args.out_dir}/labeled_units.csv and {args.out_dir}/unlabelable_rows.csv")


if __name__ == "__main__":
    main()
