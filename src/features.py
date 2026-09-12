"""
features.py

Builds per-unit features from the tidy FT table, for both the classifier and
the failure-signature clustering step. Expected outputs include:

    - per-parameter deviation from nominal/spec (magnitude, not just flag)
    - per-unit failure signature (which parameters flagged H/L/error, and how)
    - lot/tester/date context features

TODO: implement build_features(tidy_df) -> pandas.DataFrame
"""

import pandas as pd


def build_features(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Build the per-unit feature table. Not yet implemented."""
    raise NotImplementedError
