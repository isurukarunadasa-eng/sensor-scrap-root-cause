# Sensor Scrap Root Cause Analysis

Capstone project (DS/AI) — predicting sensor scrap and identifying likely root causes
from final-tester (FT) parameter data, using a rule-based baseline compared against a
trained ML classifier.

## Problem statement

Every sensor produced goes through a final tester that records per-parameter
measurements and flags out-of-spec readings. Today, scrap rate is tracked only at the
series/type level — there is no automatic way to attribute scrap to a specific root
cause without manual teardown. This project builds a pipeline that:

1. Parses raw FT test reports into a tidy, analyzable format
2. Predicts scrap/pass per unit
3. Ranks the likely root cause per failed unit, using both a rule-based baseline
   (engineering domain knowledge) and a trained classifier (improved method)
4. Surfaces results in a dashboard showing scrap trends by root cause, not just by
   series/type

## Confidentiality note

Real production test data and confirmed root-cause records are **not** committed to
this repository. `data/raw/` and `data/processed/` are gitignored and exist only on
local machines. `data/sample/` contains small, anonymized, or synthetic examples used
for demonstration and testing only.

## Repository structure

```
data/
  raw/          gitignored — real FT reports, local only
  sample/       safe, anonymized/synthetic samples
  processed/    gitignored — tidy/joined tables
notebooks/      exploratory analysis, in numbered order
src/            reusable pipeline code (parser, features, models, dashboard)
docs/           milestone write-ups and architecture diagrams
tests/          sanity checks on parsing and feature logic
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Project start: [fill in date]
Current milestone: [fill in]

## Author

[Your name] — [course code], [institution]
