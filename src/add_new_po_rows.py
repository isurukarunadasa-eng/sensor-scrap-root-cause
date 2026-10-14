"""
add_new_po_rows.py

Bring the second batch of POs (po.xlsx + FT exports, no FAR rows) into the
existing pipeline WITHOUT touching how the old data was labeled.

parser.py wants "FAR rows" (PO, article_no, root_cause_category, qty, ...).
These POs have no FAR rows, only a one-line reason per PO from the engineers,
so this script writes synthetic rows in the SAME format as usable_far_rows.csv
and parser.py then labels units exactly as before (only units whose flagged
FT parameter is physically consistent with the category are kept).

Two judgement calls, both visible in the output (new_po_far_rows.csv):

1. Category text. The engineers' free text is often misclassified by the
   family regexes (e.g. "Eeprom 7022 - LED damge" -> ASIC / EEPROM, and
   "Contact between shield and housing" -> Dimensional because of the word
   "shield"). So each PO gets a normalised category (PO_LABELS below) that
   follows the sheet / reason the engineers gave. Edit the dict to change
   any label; the original reason is kept in root_cause_narrative.
2. LED. On the tester these units fail as EEPROM code -7022, not as an LED
   current flag (ILedOn), so category "LED damaged" matches no unit. The old
   FAR rows were worded "Eeprom 7022 - LED damge", which classify() files
   under ASIC / EEPROM. Default (--led-mode asic7022) does the same, so old
   and new LED POs are treated identically. --led-mode skip leaves them out.
3. qty. There is no FAR quantity. qty is set to the number of units in the
   PO file that carry a parameter consistent with the category, so every
   such unit is kept (parser.py would otherwise keep only the top qty by
   severity). That is looser than the old FAR-backed rows.

Usage
-----
    python src/add_new_po_rows.py \
        --po-xlsx   data/raw/new_pos/po.xlsx \
        --ft-dir-in data/raw/new_pos/ft \
        --po-dir    data/raw/po_reports \
        --old-far   data/processed/usable_far_rows.csv \
        --out-far   data/processed/usable_far_rows_all.csv
Then rerun parser.py with --matched-rows data/processed/usable_far_rows_all.csv
"""

import argparse
import csv
import glob
import os
import re
import shutil
import sys
from collections import defaultdict

import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser as ftparser  # noqa: E402  (src/parser.py)

LED = "LED damaged"          # replaced by LED_ASIC when --led-mode asic7022 (default)
LED_ASIC = "Eeprom 7022 - LED damaged"
INS = "Insulation damage - contact with housing"
GLUE = "Excess glue amount"
PCBA = "PCBA damage"
FLEX = "Flex bridged with excess solder"
WRONG = "Wrong PCBA assembled - eeprom issue"

# PO -> normalised category. Decided with the user on 2026-10-14.
PO_LABELS = {
    # sheet "LED" (1493499 also appears on the EEPROM sheet: one PO, one label)
    **{po: LED for po in (1674186, 2041783, 2041787, 2041781, 1654556, 2041440, 2038910,
                          2037221, 1483554, 2031118, 1571551, 1589254, 2022273, 1493499)},
    # sheet "Insulation_isolation"
    **{po: INS for po in (1485303, 1678403, 2041337, 2040294, 2038990)},
    # sheet "Flex with variety"
    **{po: PCBA for po in (1568619, 2005010, 2004796, 1488987, 2005460, 1520187, 1527160)},
    # sheet "EEPROM": labelled by the reason text, not the sheet name
    1484082: LED, 1487350: LED,
    1478970: FLEX, 1485484: FLEX,
    1571958: "eeprom fail/flex damage",
    1509966: WRONG, 1565681: WRONG,
    1503449: "ferrite crack",
    1551006: "ASIC loose",
}
# sheet "excess glue" is read from the xlsx (every PO on it gets GLUE)
SKIP_POS = {1568805}          # dropped from the list by the user (no reason)
DROP_FILES = {"1493499_eeprom.xlsx"}   # same PO as 1493499_LED.xlsx -> avoid double count


def ftparser_family(cat):
    from consolidate_categories import classify
    return classify(cat)


def read_po_list(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    out = {}
    for ws in wb:
        for po, reason in ws.iter_rows(values_only=True):
            if not po:
                continue
            po = int(po)
            if po in SKIP_POS:
                continue
            if ws.title.lower().startswith("excess glue"):
                out.setdefault(po, (GLUE, reason))
            else:
                out.setdefault(po, (PO_LABELS.get(po), reason))
    missing = [p for p, (c, _) in out.items() if not c]
    if missing:
        raise SystemExit(f"No label for PO(s) {missing} -- add them to PO_LABELS.")
    return out


def header_info(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[next((s for s in wb.sheetnames if s.endswith("_1")), wb.sheetnames[0])]
    desig = art = ""
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=3, values_only=True), 1):
        v = str(row[0] or "")
        if v.startswith("Designation:"):
            desig = v.split(":", 1)[1].strip()
        if v.startswith("Article no:"):
            m = re.search(r"(\d{3}-\d{3}-\d{3})", v)
            art = m.group(1) if m else ""
    wb.close()
    return desig, art


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--po-xlsx", required=True)
    ap.add_argument("--ft-dir-in", required=True, help="folder with the new <PO>_<tag>.xlsx FT files")
    ap.add_argument("--po-dir", required=True, help="parser.py's PO folder; new files are copied here")
    ap.add_argument("--old-far", required=True, help="existing usable_far_rows.csv")
    ap.add_argument("--out-far", required=True)
    ap.add_argument("--new-only-out", default=None, help="also write just the new rows here")
    ap.add_argument("--led-mode", choices=["asic7022", "skip"], default="asic7022")
    ap.add_argument("--keep-overlap", action="store_true",
                    help="also add POs that already have FAR rows (default: skip them; "
                         "the real FAR row is more authoritative than the one-line reason)")
    args = ap.parse_args()

    pos = read_po_list(args.po_xlsx)
    with open(args.old_far, encoding="latin-1") as f:
        old_by_po = {str(r["PO"]): r["root_cause_category"] for r in csv.DictReader(f)}
    for po in list(pos):
        cat, reason = pos[po]
        if cat == LED:
            if args.led_mode == "skip":
                del pos[po]
            else:
                pos[po] = (LED_ASIC, reason)
    overlap = [po for po in pos if str(po) in old_by_po]
    if overlap and not args.keep_overlap:
        print("Skipping POs that already have FAR rows (old label kept):")
        for po in sorted(overlap):
            print(f"  {po}: old='{old_by_po[str(po)]}'  sheet='{pos[po][0]}'")
            del pos[po]
    files = defaultdict(list)
    for f in glob.glob(os.path.join(args.ft_dir_in, "**", "*.xlsx"), recursive=True):
        if os.path.basename(f) in DROP_FILES:
            continue
        files[int(os.path.basename(f).split("_")[0])].append(f)

    os.makedirs(args.po_dir, exist_ok=True)
    new_rows, report = [], []
    for po, (cat, reason) in sorted(pos.items()):
        if po not in files:
            print(f"WARNING: PO {po} has no FT file -- skipped")
            continue
        n_ok, n_flag, desig, art = 0, 0, "", ""
        for f in files[po]:
            shutil.copy2(f, os.path.join(args.po_dir, os.path.basename(f)))
            d, a = header_info(f)
            desig, art = desig or d, art or a
            parsed = ftparser.load_po_file(f)
            for i in ftparser.flagged_unit_indices(parsed):
                n_flag += 1
                fp = [p for p, fl in parsed["flags"].items() if i < len(fl) and fl[i]]
                if ftparser.matching_flagged_params(cat, fp, parsed, i):
                    n_ok += 1
        new_rows.append({
            "CC_reference": f"PO-{po}", "PO": po, "article_no": art, "series": desig,
            "symptom": "", "root_cause_category": cat,
            "root_cause_narrative": (reason or "").strip(), "qty": max(n_ok, 1),
            "po_already_covered": False,
        })
        report.append((po, cat, n_flag, n_ok))

    cols = ["CC_reference", "PO", "article_no", "series", "symptom", "root_cause_category",
            "root_cause_narrative", "qty", "po_already_covered"]
    with open(args.old_far, encoding="latin-1") as f:
        old = list(csv.DictReader(f))
    with open(args.out_far, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(old); w.writerows(new_rows)
    if args.new_only_out:
        with open(args.new_only_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(new_rows)

    print(f"{len(new_rows)} new PO rows -> {args.out_far}  (old rows kept: {len(old)})")
    print(f"{'PO':>8}  {'flagged':>7}  {'consistent':>10}  category")
    for po, cat, nf, nk in report:
        print(f"{po:>8}  {nf:>7}  {nk:>10}  {cat}")
    zero = [po for po, _, _, nk in report if nk == 0]
    if zero:
        print(f"\nPOs that will yield NO labeled unit (no consistent flag): {zero}")
    from collections import Counter
    fam = Counter()
    for _, cat, _, nk in report:
        fam[ftparser_family(cat)] += nk
    print("\nNew labeled units by family (before the 20-unit rule):")
    for k, v in fam.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
