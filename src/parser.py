"""
parser.py

Converts a raw FT (final tester) report — wide format, one column per tested
unit, one row per parameter — into a tidy long-format table:

    unit_id | parameter | value | status | designation | tester | lot | date

status is derived from the tester's own encoding:
    - plain number  -> "pass"
    - "H..." prefix -> "high"   (out-of-spec, too high)
    - "L..." prefix -> "low"    (out-of-spec, too low)
    - "!-XXXX"       -> "error"  (hard fault code, not a measurement)
    - "--"           -> "skipped" (test not run, usually due to earlier failure)

TODO: implement parse_ft_report(filepath) -> pandas.DataFrame
"""

import pandas as pd


def parse_ft_report(filepath: str) -> pd.DataFrame:
    """Parse a single FT report file into a tidy DataFrame. Not yet implemented."""
    raise NotImplementedError
