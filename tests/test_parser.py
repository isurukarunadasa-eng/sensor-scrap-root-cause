"""
Basic sanity checks for the parser. Expand as parser.py is implemented.

Example checks to add once parser.py is written:
    - a plain numeric cell parses to status "pass"
    - a cell like "H45.271" parses to value=45.271, status="high"
    - a cell like "L4.445" parses to value=4.445, status="low"
    - a cell like "!-4020" parses to status="error"
    - a cell "--" parses to status="skipped"
"""

import pytest


def test_placeholder():
    # TODO: replace with real parser tests once parser.py is implemented
    assert True
