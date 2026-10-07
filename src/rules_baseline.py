"""
rules_baseline.py

The "baseline method" required by the capstone's baseline-vs-improved-method
structure: a DETERMINISTIC classifier built directly from Contrinex's own
engineering documentation -- no machine learning, no training data, no
learned parameters. It exists to answer "how much better is the ML model
than just encoding what engineers already know?"

Source documents
-----------------
1. Error_Code.pdf / ASIC_Error_codes.pdf (C.CQ9.27.A37.x) -- section 8,
   "IC500 CTX510" (-7001 to -7029): the calibration-step error codes
   raised by the Final Tester for 500-series sensors, each with a
   Description (what calibration step failed) and sometimes a Hints line
   (why it might fail).
2. eeprom_root_cause.xlsx -- an engineer-authored table giving the actual
   PHYSICAL root cause for some of those -70xx codes (e.g. -7008
   CALIBR_SPV3 -> "Ferrite crack"). Not all codes have one: Contrinex's
   own document leaves many blank (process/software errors with no single
   physical cause, or not yet characterized).

Design: three confidence tiers, explicit about which is which
----------------------------------------------------------------
For a code with no engineer-confirmed root cause, the brief from Isuru was
"we can give the reason as a hint" -- i.e. don't just give up, fall back to
the error's own Description/Hints text and classify THAT. This mirrors
consolidate_categories.py's classify() (same keyword rules, reused here via
import), so baseline predictions land in the SAME family taxonomy as the ML
model's model_target, making the two directly comparable on the same test
set. Every prediction records which tier it came from, so the report can be
honest about confidence, not just accuracy:

  TIER 1 (root_cause)  -- eeprom_root_cause.xlsx gives an explicit physical
                           cause. Highest confidence: an engineer looked at
                           failed units and wrote this down.
  TIER 2 (hint)         -- no root cause on file, but the PDF's Description
                           and/or Hints text contains a family-identifying
                           keyword (e.g. "EEPROM" -> ASIC/EEPROM). Medium
                           confidence: inferred from the step name, not a
                           confirmed physical inspection.
  TIER 3 (unknown)      -- no root cause, and the description/hints carry no
                           family keyword (e.g. "Error during SPV 1
                           adjustment" names a calibration step, not a
                           physical part). Falls back to "Other / Minor
                           Defects" -- this is an honest "the rules don't
                           know", not a guess.

Which FT columns this baseline can actually use
-------------------------------------------------
Only 9 of the 28 FT feature columns are 0/1 pass-fail flags for one of
these -70xx calibration steps (the rest are continuous measurements with no
published spec thresholds in the documents provided, so the baseline can't
rule on them without guessing a threshold):

  CalibrSP1 [0/1]     <-> -7012 CALIBR_SPV1
  CalibrSP1Hys [0/1]  <-> -7011 CALIBR_SPV1_HYS
  CalibrSP2 [0/1]     <-> -7010 CALIBR_SPV2
  CalibrSP2Hys [0/1]  <-> -7009 CALIBR_SPV2_HYS
  CalibrSP3 [0/1]     <-> -7008 CALIBR_SPV3      (TIER 1: "Ferrite crack")
  CalibrSP3Hys [0/1]  <-> -7007 CALIBR_SPV3_HYS
  IcPreconfig [0/1]   <-> -7018 PRECONFIG         (TIER 1: "coil contact with shield")
  5V4trim [0/1]       <-> -7019 TRIM_5V4
  TemperatureComp[0/1]<-> -7013 TEMP_COMP

Two more flags aren't ASIC calibration steps at all but have an obvious,
documented physical meaning on their own (Final_test-_Inductive.pdf):
  TestIsolation [0/1] -- isolation test 500/1500 VAC -> Insulation/Isolation
  YellowLed [0/1]     -- LED function test -> LED

How "fail" actually shows up: a stopped SEQUENCE, not a 0
--------------------------------------------------------------
Isuru's correction (2026-10-07): these 11 flags are never literally "0" for
a fail. The tester runs them in a fixed order and ABORTS the whole
calibration sequence the moment one step errors -- so every step after the
failure point is simply never written (blank), not written as a fail value.
A '1' means "this step was reached and completed"; a blank means either
"not applicable to this test program" OR "the sequence stopped before
reaching this step because an earlier, unlisted step errored out". The
useful signal is therefore GAP POSITION: the first step in the sequence
that's blank while everything before it is '1' marks where testing
stopped.

This was verified empirically against the real labeled data (not just
assumed): grouping units by their reach/no-reach pattern across the 11
flags and cross-tabulating against the FAR's own root_cause_category
showed three clean clusters --

  pattern (I=IcPreconfig, 5=5V4trim, T=TemperatureComp, then the 6
  CalibrSP*/Hys flags, Y=YellowLed, Z=TestIsolation; 1=reached, 0=blank)

  00000000000  (nothing reached -- ASIC never initialized)
      -> almost entirely "ASIC loose" / "ASIC bridge" in the real labels
  11000000000  (IcPreconfig + 5V4trim reached, then stopped)
      -> almost entirely "High coil resistance" / "Eeprom 7015 - Coil
         damage" (-7015 VRCU_READING's own documented root cause is
         "Coil damage / coil resistance" -- this gap sits exactly where
         the VASS/VRCU coil-voltage BMUX readings, -7014/-7015, run in
         the real sequence, even though they have no dedicated FT column)
  11111111111  (everything in this block reached)
      -> "LED not lightup" / "Ferrite crack" / "Potting layer..." -- i.e.
         real defects the ASIC calibration sequence completed right
         through; these are caught by OTHER tests (LED, distance/
         hysteresis measurements) outside this 11-flag sequence entirely,
         which this baseline can't rule on without published spec
         thresholds for the continuous parameters

That first pass used a small leftover 37-row sample. It was then REDONE on
the real train_units.csv (1,094 rows -- diagnose_gap_patterns.py, run by
Isuru, 2026-10-07) and gave a cleaner, more important picture, with one
real surprise:

  TestIsolation is NOT part of the sequential abort chain. It shows up as
  reached even when every other flag in the chain is still blank -- e.g.
  30 units have the entire core 10-flag chain blank AND TestIsolation
  populated. That means TestIsolation runs as an independent, earlier test
  station, not gated by ASIC calibration at all. It was pulled out of
  TEST_ORDER and used instead as a secondary split at the two zones where
  it mattered:

  Zone "nothing in the core chain reached" (n=294 total):
    TestIsolation reached  (n=30):  100% Dimensional/Alignment (all
      "Trimming Failure") -- a completely different failure mode from...
    TestIsolation NOT reached (n=264): 58.3% ASIC/EEPROM, 28.0% Flex/PCB/
      Solder/Wiring, 13.6% Coil

  Zone "reached 5V4trim, gap before TemperatureComp" (n=725 total):
    TestIsolation reached (n=506): 90.3% Dimensional/Alignment
    TestIsolation NOT reached (n=219): 57.1% Coil, 20.1% Dimensional/
      Alignment, 11.4% Flex/PCB/Solder/Wiring, 11.0% ASIC/EEPROM

  Zone "full 10-flag core chain completed" (n=61):
    72.1% Dimensional/Alignment, 13.1% Insulation/Isolation, 8.2% LED,
    6.6% Flex/PCB/Solder/Wiring

  Zone "gap after CalibrSP2" (n=12, too small/mixed for a confident call):
    33.3% Flex/PCB/Solder/Wiring, 33.3% Coil, 25.0% ASIC/EEPROM, 8.3%
    Gluing/Potting -- no majority; kept as a weak, explicitly low-
    confidence guess

These are MAJORITY-VOTE rules fit on train data, a third evidentiary tier
("data_majority") distinct from an xlsx-confirmed physical root cause
("root_cause") or a PDF-text keyword fallback ("hint") -- every one of
them carries its own purity percentage in its explanation so the baseline
report can be honest about how reliable each rule actually is, not just
report one blended accuracy number. Since these were fit on train_units.csv
only, scoring the baseline on test_units.csv (never seen during this
derivation) is a fair comparison against the ML model, which obeyed the
same train/test boundary.

Multiple failed flags on one unit
-----------------------------------
Per Isuru: when a unit shows a clear EEPROM/ASIC calibration-sequence gap,
that takes priority over everything else -- it's the most direct engineered
signal available. Only once the whole 11-flag sequence completes (no gap)
does the baseline fall back to "no failure found in this rule set" rather
than inventing a cause from parameters it has no spec thresholds for.

Usage
-----
    python src/rules_baseline.py \\
        --in data/processed/test_units.csv \\
        --out reports/baseline_predictions.csv
"""

import argparse
import csv
import re
from collections import Counter

from consolidate_categories import FAMILY_RULES, MINOR_BUCKET_NAME, classify

# --- Section 8 (IC500/CTX510, -7001 to -7029) from Error_Code.pdf, in full ---
# code: (error_name, description, hints_or_None, root_cause_or_None)
CTX510_ERROR_CODES = {
    -7001: ("LOCK_SENSOR", "Error during lock test", None, None),
    -7002: ("SAVE_EEPROM_CONTENT", "Error during store test report measures",
            "Check database connection if exists", None),
    -7003: ("CHECK_EEPROM_MEMORY_CONTENT", "Error during EEPROM memory content control",
            "Can happen when coil wire is broken", None),
    -7004: ("SAVE_SENSOR_ID", "Error during product traceability",
            "Check database connection or path for traceability files.", None),
    -7005: ("WRITE_EEPROM_ONLY", "Error during write EEPROM only fields", None, None),
    -7006: ("COPY_RAM_TO_EEPROM", "Error during copying RAM field to EEPROM field", None, None),
    -7007: ("CALIBR_SPV3_HYS", "Error during hysteresis 3 (hyst field) adjustment", None, None),
    -7008: ("CALIBR_SPV3", "Error during SPV 3 adjustment", None, "Ferrite crack"),
    -7009: ("CALIBR_SPV2_HYS", "Error during hysteresis 2 (hyst field) adjustment", None, None),
    -7010: ("CALIBR_SPV2", "Error during SPV 2 adjustment", None, None),
    -7011: ("CALIBR_SPV1_HYS", "Error during hysteresis 1 (hyst field) adjustment", None, None),
    -7012: ("CALIBR_SPV1", "Error during SPV 1 adjustment", None, None),
    -7013: ("TEMP_COMP", "Error during LUT value writing in RAM",
            "Check in prm file if LUT values isn't bigger than 2^n (n= bit nbr of lut asic field)", None),
    -7014: ("VASS_READING", "Error during reading VASS value by the BMUX", None,
            "Bobbing melt / Ferrite Crack"),
    -7015: ("VRCU_READING", "Error during reading VRCU value by the BMUX", None,
            "Coil damage /coil resistance"),
    -7016: ("TEMPERATURE_MEASUREMENT", "Error during reading VPTAT value by the BMUX", None, None),
    -7017: ("THERMAL_CHARACTERIZATION", "Error during writing neutral LUT value in RAM",
            "In validation mode, check if the Excel file is the right one", None),
    -7018: ("PRECONFIG", "Error during ASIC initialization.", None, "coil contact with shield"),
    -7019: ("TRIM_5V4", "Error during 5V4 trimming", None, None),
    -7020: ("HANDLE_TRIM_DATA", "Error during handling trimming data (1V3, 3V5, VMV,VBG, ERRIRCUxm)", None, None),
    -7021: ("HANDLE_PRESET_DATA", "Error during preset data handling.", None, None),
    -7022: ("PREPARE_FOR_CALIBRATION",
            "Error during preparation for calibration. In general case, this is not possible to unlock the sensor.",
            "Happen when IC cannot start correctly (short-circuit on internal sensor DC voltage (5V, 7.5V) "
            "or when there is a problem with the output transistor or a bad connection with the power supply).",
            "Copper Shield not aligned to the ferrite edge. Check the shield allignment, ASIC loose, "
            "Possible flex PCB damaged by the folding point, LED damage, Diode damage, Ferrite detached "
            "and coil contacts with round PCB"),
    -7023: ("VRCUOFF_READING", "Read VRCUOFF value in the sensor", None, None),
    -7024: ("VRCUON_READING", "Read VRCUON value in the sensor", None, None),
    -7025: ("CALIBR_UDIST_1", "Adjustment of SELVOUTOFFSET, Analog distance 1",
            "Check in filter file -> 5V4V don't be sectioned if IcPreconfig isn't sectioned too / "
            "Check Voltmeter 0 calibration", None),
    -7026: ("CALIBR_UDIST_5", "Adjustment of VANAREF, Analog distance 5",
            "Check in filter file -> 5V4V don't be sectioned if IcPreconfig isn't sectioned too / "
            "Check Voltmeter 0 calibration", "ferrite not centered, Excessive Gap between two shield ends"),
    -7027: ("VRCUOFFSET_TRIM", "Trimming of VRCUOFFSET", None, None),
    -7028: ("CALIBR_UDIST_20", "Calibration of VANAREF. Distance = DMS Dist 20", None, None),
    -7029: ("LOCK_SENSOR_ISDU_AND_UNLOCK_IOL", "Error during the Lock ISDU / Unlock IOLINK", None, None),
}

# Which FT feature column observes which error code's pass/fail outcome.
# Only codes with a directly observable per-unit flag in the 28 exported FT
# features are wired up -- the rest of CTX510_ERROR_CODES above stays as
# reference documentation for when more FT columns become available.
FEATURE_TO_CODE = {
    "CalibrSP3 [0/1]": -7008,       # TIER 1: Ferrite crack
    "IcPreconfig [0/1]": -7018,     # TIER 1: coil contact with shield
    "CalibrSP3Hys [0/1]": -7007,
    "CalibrSP2 [0/1]": -7010,
    "CalibrSP2Hys [0/1]": -7009,
    "CalibrSP1 [0/1]": -7012,
    "CalibrSP1Hys [0/1]": -7011,
    "5V4trim [0/1]": -7019,
    "TemperatureComp [0/1]": -7013,
}

# The real execution order of the ASIC calibration sequence, as reached by
# the FT program -- derived empirically (see module docstring), NOT from
# the error codes' numeric order, which turned out not to match execution
# order at all (e.g. PRECONFIG/-7018 runs first, long before the -7007..
# -7012 SPV steps). TestIsolation is deliberately NOT in this list -- real
# train data shows it runs independently of this chain (see docstring) and
# is handled as a separate side-signal in classify_unit() instead.
TEST_ORDER = [
    "IcPreconfig [0/1]",      # -7018, runs 1st
    "5V4trim [0/1]",          # -7019, runs 2nd
    # <implicit gap zone>     -- VASS_READING/-7014, VRCU_READING/-7015 run
    #                            here but have no dedicated FT column; a
    #                            stall between 5V4trim and TemperatureComp
    #                            is attributed to them (see GAP_RULES)
    "TemperatureComp [0/1]",  # -7013
    "CalibrSP1Hys [0/1]",     # -7011
    "CalibrSP1 [0/1]",        # -7012
    "CalibrSP2Hys [0/1]",     # -7009
    "CalibrSP2 [0/1]",        # -7010
    "CalibrSP3Hys [0/1]",     # -7007
    "CalibrSP3 [0/1]",        # -7008, root cause: Ferrite crack
    "YellowLed [0/1]",        # LED function test (not a CTX510 code)
]

TEST_ISOLATION_COL = "TestIsolation [0/1]"

# Data-majority gap rules, fit on train_units.csv (1,094 rows, 2026-10-07)
# -- see module docstring for the full breakdown and purity percentages
# behind each one. key = index in TEST_ORDER of the LAST step successfully
# reached (-1 = nothing reached at all). Where TestIsolation reached/not
# materially changed the outcome, the rule is split; where it didn't (or
# there wasn't enough data to tell), it's a single entry.
GAP_RULES = {
    -1: {
        True: (  # TestIsolation reached despite the core chain being fully blank
            "Dimensional / Alignment",
            "data_majority",
            "Core chain never started, but TestIsolation ran independently -- "
            "100% of these (n=30 in train) are 'Trimming Failure'.",
        ),
        False: (
            "ASIC / EEPROM",
            "data_majority",
            "PRECONFIG (-7018) never reached and TestIsolation didn't run either "
            "-- 58.3% ASIC/EEPROM in train (n=264), with 28.0% Flex/PCB/Solder/"
            "Wiring and 13.6% Coil as notable minorities (mixed zone, not pure).",
        ),
    },
    1: {  # reached 5V4trim (index 1), stopped before TemperatureComp (index 2)
        True: (
            "Dimensional / Alignment",
            "data_majority",
            "Stopped between 5V4trim and TemperatureComp, but TestIsolation ran "
            "-- 90.3% Dimensional/Alignment in train (n=506).",
        ),
        False: (
            "Coil",
            "data_majority",
            "Stopped between 5V4trim and TemperatureComp, TestIsolation didn't run "
            "-- 57.1% Coil in train (n=219; matches -7015 VRCU_READING's documented "
            "root cause 'Coil damage / coil resistance', which runs in this gap but "
            "has no dedicated FT column), with Dimensional/Alignment (20.1%), Flex/"
            "PCB/Solder/Wiring (11.4%) and ASIC/EEPROM (11.0%) as minorities.",
        ),
    },
    6: {  # gap after CalibrSP2 (index 6), before CalibrSP3Hys
        None: (
            "Coil",
            "hint",
            "Gap after CalibrSP2 -- only n=12 in train, no clear majority "
            "(Flex/PCB/Solder/Wiring 33.3%, Coil 33.3%, ASIC/EEPROM 25.0%, "
            "Gluing/Potting 8.3%). Low-confidence tie-break toward Coil, "
            "consistent with the broader finding that mid-sequence SPV "
            "calibration stalls tend to be coil-related.",
        ),
    },
}

# The "full chain completed" case (no gap at all) -- also a data-majority
# call, not a shrug. n=61 in train, 72.1% Dimensional/Alignment (vs. 13.1%
# Insulation/Isolation, 8.2% LED, 6.6% Flex/PCB/Solder/Wiring).
FULL_CHAIN_PREDICTION = (
    "Dimensional / Alignment",
    "data_majority",
    "Full CTX510 core calibration sequence completed (no gap) -- 72.1% of "
    "these units in train (n=61) were still Dimensional/Alignment defects "
    "(ferrite crack, gap/cap issues) caught by tests outside this flag "
    "sequence; Insulation/Isolation (13.1%) and LED (8.2%) are notable "
    "minorities.",
)


def classify_unit(row: dict):
    """Returns (predicted_family, tier, explanation) for one FT row, using
    gap position in the calibration sequence -- NOT a literal fail value
    (these flags are blank-on-abort, not 0-on-fail; see module docstring).

    tier is one of "root_cause" (xlsx-confirmed physical cause),
    "data_majority" (majority vote fit on train_units.csv for this exact
    gap position), "hint" (PDF description/hints text, keyword-classified),
    "unknown" (no rule fires) -- kept separate from predicted_family so the
    evaluation can report accuracy broken down by confidence tier, not just
    one blended number.
    """
    test_isolation_reached = (row.get(TEST_ISOLATION_COL) or "").strip() != ""

    last_reached = -1
    for i, col in enumerate(TEST_ORDER):
        val = (row.get(col) or "").strip()
        if val == "":
            break  # gap starts here
        last_reached = i
    else:
        # Walked the whole TEST_ORDER with nothing blank -- full core chain
        # completed. Data-majority call, not a shrug (see FULL_CHAIN_PREDICTION).
        return FULL_CHAIN_PREDICTION

    if last_reached in GAP_RULES:
        rule = GAP_RULES[last_reached]
        if None in rule:  # single rule, no TestIsolation split
            return rule[None]
        return rule[test_isolation_reached]

    # No data-majority rule for this exact gap position (not observed, or
    # too rare, in train_units.csv) -- fall back to the specific step's own
    # documentation (xlsx root cause, else PDF description/hints text).
    stopped_at_col = TEST_ORDER[last_reached + 1]

    if stopped_at_col in DIRECT_FLAG_TEXT:
        text = DIRECT_FLAG_TEXT[stopped_at_col]
        family = classify(text)
        tier = "hint" if family != "Other" else "unknown"
        return family if family != "Other" else MINOR_BUCKET_NAME, tier, text

    code = FEATURE_TO_CODE[stopped_at_col]
    name, description, hints, root_cause = CTX510_ERROR_CODES[code]
    if root_cause:
        family = classify(root_cause)
        return (family if family != "Other" else MINOR_BUCKET_NAME,
                "root_cause",
                f"Stopped at {code} {name}: {root_cause}")

    # No engineer-confirmed root cause -- fall back to the error's own
    # description + hints text as a weaker signal ("reason as a hint").
    fallback_text = f"{name} {description} {hints or ''}"
    family = classify(fallback_text)
    if family != "Other":
        return family, "hint", f"Stopped at {code} {name}: {description}" + (f" ({hints})" if hints else "")
    return (MINOR_BUCKET_NAME, "unknown",
            f"Stopped at {code} {name}: {description} -- no family-identifying rule")


# Flags that are documented physical tests in their own right (not ASIC
# calibration-step codes), from Final_test-_Inductive.pdf.
DIRECT_FLAG_TEXT = {
    "YellowLed [0/1]": "Yellow LED function test failure",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                     help="test_units.csv (or train_units.csv) -- needs the 0/1 FT flag columns "
                          "and, if present, model_target for scoring")
    ap.add_argument("--out", default="reports/baseline_predictions.csv")
    args = ap.parse_args()

    with open(args.in_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    has_truth = rows and "model_target" in rows[0]

    out_rows = []
    correct = 0
    tier_counts = Counter()
    tier_correct = Counter()
    for row in rows:
        pred_family, tier, explanation = classify_unit(row)
        tier_counts[tier] += 1
        out_row = {
            "PO": row.get("PO", ""),
            "baseline_prediction": pred_family,
            "confidence_tier": tier,
            "explanation": explanation,
        }
        if has_truth:
            truth = row.get("model_target", "")
            out_row["model_target"] = truth
            is_correct = pred_family == truth
            out_row["correct"] = str(is_correct)
            if is_correct:
                correct += 1
                tier_correct[tier] += 1
        out_rows.append(out_row)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"{len(rows)} units classified -> {args.out}\n")
    print("Prediction confidence tiers:")
    for tier, n in tier_counts.most_common():
        if has_truth:
            tier_acc = tier_correct[tier] / n if n else 0
            print(f"  {n:4d}  {tier:<15} ({tier_correct[tier]}/{n} = {tier_acc:.1%} correct)")
        else:
            print(f"  {n:4d}  {tier}")

    if has_truth:
        acc = correct / len(rows) if rows else 0
        print(f"\nOverall baseline accuracy on this set: {correct}/{len(rows)} = {acc:.1%}")
        print("(Compare this against the improved method's classification_report.txt for "
              "the real baseline-vs-improved comparison. The per-tier breakdown above matters "
              "as much as the overall number: 'root_cause' and strong 'data_majority' rules "
              "(e.g. the 90.3%/58.3%/57.1% purity zones) are the baseline's genuinely confident "
              "predictions; 'hint' and 'unknown' rows are honest low-confidence guesses, not "
              "wrong answers dressed up -- this is the difference a plain accuracy number from "
              "the ML model's classification_report.txt can't show on its own.)")
    else:
        print("\nNo model_target column in input -- predictions written without scoring.")


if __name__ == "__main__":
    main()
