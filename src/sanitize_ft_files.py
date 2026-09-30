"""
sanitize_ft_files.py

Strips employer-identifying metadata from FT/PO .xlsx export files before
they go into a public git repo.

Why this is needed
-------------------
Excel silently stores a "last saved from" path in every workbook it saves
via OneDrive/SharePoint, inside xl/workbook.xml:

    <x15ac:absPath url="https://contrinex365-my.sharepoint.com/personal/
        dulakshi_bernadge_contrinex_com/Documents/Desktop/ft files/" .../>

That single line embeds both the company name/domain and a specific
colleague's name+email. The same identifying info can also live in
docProps/core.xml (creator / lastModifiedBy) and docProps/app.xml
(Company), and occasionally in docProps/custom.xml. None of this is
visible when you open the file in Excel and look at the cells -- it's
only visible if you unzip the .xlsx (it's a zip of XML parts) or grep it
as text, which is exactly what anyone pulling the file from a public
GitHub repo would be able to do.

This script does NOT touch cell data, sheet names, or any test values --
only the four metadata locations above. It never overwrites your
originals: it writes sanitized copies into a separate output directory,
mirroring the input directory structure.

Usage
-----
    python sanitize_ft_files.py --in-dir data/raw --out-dir data/raw_public

Then point .gitignore / your commit at data/raw_public instead of the
original data/raw, and run --verify against the output to confirm nothing
was missed.
"""

import argparse
import os
import re
import shutil
import zipfile

# Patterns to scrub, wherever they appear in the small set of metadata XML
# parts we touch. Matching is case-insensitive.
IDENTIFYING_PATTERNS = [
    re.compile(r"contrinex", re.IGNORECASE),
]

# Metadata files that can carry author/company/path info.
METADATA_PARTS = {
    "docProps/core.xml",
    "docProps/app.xml",
    "docProps/custom.xml",
    "xl/workbook.xml",
}


def scrub_core_xml(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="replace")
    # Blank out <dc:creator>...</dc:creator> and <cp:lastModifiedBy>...</cp:lastModifiedBy>
    text = re.sub(r"(<dc:creator>).*?(</dc:creator>)", r"\1\2", text, flags=re.DOTALL)
    text = re.sub(
        r"(<cp:lastModifiedBy>).*?(</cp:lastModifiedBy>)", r"\1\2", text, flags=re.DOTALL
    )
    text = re.sub(
        r"(<cp:category>).*?(</cp:category>)", r"\1\2", text, flags=re.DOTALL
    )
    return text.encode("utf-8")


def scrub_app_xml(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="replace")
    text = re.sub(r"(<Company>).*?(</Company>)", r"\1\2", text, flags=re.DOTALL)
    text = re.sub(r"(<Manager>).*?(</Manager>)", r"\1\2", text, flags=re.DOTALL)
    return text.encode("utf-8")


def scrub_workbook_xml(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="replace")
    # Remove the whole mc:AlternateContent block if it contains an absPath
    # (that's the SharePoint "last saved from" URL carrying company+person).
    text = re.sub(
        r"<mc:AlternateContent[^>]*>.*?</mc:AlternateContent>",
        "",
        text,
        flags=re.DOTALL,
    )
    # Belt-and-braces: strip any standalone absPath element too.
    text = re.sub(r"<x15ac:absPath[^/]*/>", "", text)
    return text.encode("utf-8")


def scrub_custom_xml(data: bytes) -> bytes:
    # docProps/custom.xml holds arbitrary org-defined properties; if the
    # whole part matches an identifying pattern, it's safest to blank all
    # <vt:lpwstr> string values rather than guess which property it's in.
    text = data.decode("utf-8", errors="replace")
    if any(p.search(text) for p in IDENTIFYING_PATTERNS):
        text = re.sub(
            r"(<vt:lpwstr>).*?(</vt:lpwstr>)", r"\1\2", text, flags=re.DOTALL
        )
    return text.encode("utf-8")


SCRUBBERS = {
    "docProps/core.xml": scrub_core_xml,
    "docProps/app.xml": scrub_app_xml,
    "docProps/custom.xml": scrub_custom_xml,
    "xl/workbook.xml": scrub_workbook_xml,
}


WORKSHEET_RELS_RE = re.compile(r"^xl/worksheets/_rels/(sheet\d+)\.xml\.rels$")


def find_identifying_hyperlink_rids(rels_xml_text: str) -> list[str]:
    """Return the Relationship Ids of hyperlink entries whose Target matches
    an identifying pattern (e.g. the company's internal CRM domain)."""
    rids = []
    for m in re.finditer(r"<Relationship\b[^>]*/>", rels_xml_text):
        elem = m.group(0)
        if 'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"' not in elem:
            continue
        target_m = re.search(r'Target="([^"]*)"', elem)
        if not target_m:
            continue
        target = target_m.group(1)
        if any(p.search(target) for p in IDENTIFYING_PATTERNS):
            id_m = re.search(r'Id="([^"]*)"', elem)
            if id_m:
                rids.append(id_m.group(1))
    return rids


def strip_hyperlink_relationships(rels_xml_text: str, rids: set[str]) -> str:
    def repl(m):
        elem = m.group(0)
        id_m = re.search(r'Id="([^"]*)"', elem)
        if id_m and id_m.group(1) in rids:
            return ""
        return elem

    return re.sub(r"<Relationship\b[^>]*/>", repl, rels_xml_text)


def strip_sheet_hyperlink_refs(sheet_xml_text: str, rids: set[str]) -> str:
    """Remove <hyperlink .../> entries in the sheet's <hyperlinks> block
    that reference one of the given relationship Ids, and drop the whole
    <hyperlinks> element if nothing is left in it."""

    def strip_one(m):
        elem = m.group(0)
        id_m = re.search(r'r:id="([^"]*)"', elem)
        if id_m and id_m.group(1) in rids:
            return ""
        return elem

    def repl_block(m):
        block = m.group(0)
        inner = re.sub(r"<hyperlink\b[^>]*/>", strip_one, block)
        if not re.search(r"<hyperlink\b", inner):
            return ""
        return inner

    return re.sub(r"<hyperlinks>.*?</hyperlinks>", repl_block, sheet_xml_text, flags=re.DOTALL)


def sanitize_file(src_path: str, dst_path: str) -> list[str]:
    """Sanitize one .xlsx. Returns a list of warnings (non-fatal)."""
    warnings = []
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with zipfile.ZipFile(src_path, "r") as zin:
        names = zin.namelist()
        infos = {i.filename: i for i in zin.infolist()}
        parts = {name: zin.read(name) for name in names}

    # Pass 1: find identifying hyperlinks in every worksheet's rels file,
    # and strip those relationship entries.
    rids_by_sheet = {}
    for name in list(parts.keys()):
        m = WORKSHEET_RELS_RE.match(name)
        if not m:
            continue
        text = parts[name].decode("utf-8", errors="replace")
        rids = find_identifying_hyperlink_rids(text)
        if rids:
            rids_by_sheet[m.group(1)] = set(rids)
            parts[name] = strip_hyperlink_relationships(text, set(rids)).encode("utf-8")

    # Pass 2: remove the matching <hyperlink r:id="..."/> refs from each
    # sheet's own XML so it doesn't point at a relationship that no longer
    # exists.
    for sheet, rids in rids_by_sheet.items():
        sheet_part = f"xl/worksheets/{sheet}.xml"
        if sheet_part in parts:
            text = parts[sheet_part].decode("utf-8", errors="replace")
            parts[sheet_part] = strip_sheet_hyperlink_refs(text, rids).encode("utf-8")

    # Pass 3: the existing metadata scrubs (docProps, workbook.xml absPath).
    for name in list(parts.keys()):
        if name in SCRUBBERS:
            parts[name] = SCRUBBERS[name](parts[name])

    # Safety net: flag anything still carrying identifying text that none
    # of the above passes targeted, so it gets manual review instead of
    # silently shipping.
    for name, data in parts.items():
        if name in METADATA_PARTS or WORKSHEET_RELS_RE.match(name):
            continue
        try:
            text = data.decode("utf-8", errors="ignore")
            if any(p.search(text) for p in IDENTIFYING_PATTERNS):
                warnings.append(
                    f"  ! {name}: identifying text found OUTSIDE metadata/"
                    f"hyperlink parts -- needs manual review, not auto-scrubbed"
                )
        except Exception:
            pass

    with zipfile.ZipFile(src_path, "r") as zin:
        infos = {i.filename: i for i in zin.infolist()}
    with zipfile.ZipFile(dst_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            zout.writestr(infos[name], parts[name])

    return warnings


def verify_file(path: str) -> bool:
    """Return True if the file is clean (no identifying strings anywhere)."""
    with zipfile.ZipFile(path, "r") as z:
        for name in z.namelist():
            data = z.read(name)
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if any(p.search(text) for p in IDENTIFYING_PATTERNS):
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, help="Directory of original .xlsx files")
    ap.add_argument("--out-dir", required=True, help="Directory to write sanitized copies")
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip sanitizing; just verify files already in --out-dir are clean",
    )
    args = ap.parse_args()

    if args.verify_only:
        target_dir = args.out_dir
        files = []
        for root, _, names in os.walk(target_dir):
            for n in names:
                if n.lower().endswith(".xlsx"):
                    files.append(os.path.join(root, n))
        dirty = [f for f in files if not verify_file(f)]
        print(f"Checked {len(files)} files in {target_dir}")
        if dirty:
            print(f"STILL IDENTIFYING: {len(dirty)} file(s):")
            for f in dirty:
                print(f"  {f}")
        else:
            print("All clean -- no identifying strings found.")
        return

    total = 0
    all_warnings = []
    for root, _, names in os.walk(args.in_dir):
        for n in names:
            if not n.lower().endswith(".xlsx"):
                continue
            src = os.path.join(root, n)
            rel = os.path.relpath(src, args.in_dir)
            dst = os.path.join(args.out_dir, rel)
            warnings = sanitize_file(src, dst)
            all_warnings.extend(warnings)
            total += 1

    print(f"Sanitized {total} file(s) -> {args.out_dir}")

    # Auto-verify every output file.
    dirty = []
    for root, _, names in os.walk(args.out_dir):
        for n in names:
            if n.lower().endswith(".xlsx"):
                p = os.path.join(root, n)
                if not verify_file(p):
                    dirty.append(p)

    if dirty:
        print(f"\nWARNING: {len(dirty)} sanitized file(s) STILL contain identifying text:")
        for f in dirty:
            print(f"  {f}")
        print("Do not commit these until resolved.")
    else:
        print("Verified: no identifying strings remain in any sanitized file.")

    if all_warnings:
        print("\nPer-file notes during sanitizing:")
        for w in all_warnings:
            print(w)


if __name__ == "__main__":
    main()
