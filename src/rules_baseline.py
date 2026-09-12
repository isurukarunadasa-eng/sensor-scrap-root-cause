"""
rules_baseline.py

Encodes the engineering rules doc (parameter -> known likely cause) as a
deterministic lookup classifier. This is the project's BASELINE method,
which the trained ML classifier (model.py) is compared against.

TODO: load the rules doc (data/sample or a local config, not committed if
confidential) and implement predict_cause_rule_based(unit_features) -> str
"""


def predict_cause_rule_based(unit_features: dict) -> str:
    """Return the rule-based likely root cause for one unit. Not yet implemented."""
    raise NotImplementedError
