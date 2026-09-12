# Milestone 1 — Project Definition

## Problem statement


## Category / problem type
- Category: Anomaly/Fraud Detection System -> Sensor fault detection
- Problem type: Classification (pass/scrap) + diagnostic root-cause attribution
- Data type: Tabular (semi-structured FT exports)
- Technique category: Rule-based baseline + ML classification, with SHAP explainability
- System context: Pipeline + dashboard

## Scope
- In scope: [e.g. 500-series inductive sensors]
- Out of scope / future work: [e.g. 600/700-series generalization]

## Data source
- FT test reports: [description, volume, date range]
- Confirmed root-cause records: [description, volume, number of distinct causes]
- Engineering rules doc: [description]

## Intended methods
- Baseline: rules-based lookup from engineering doc
- Improved method: trained classifier + SHAP

## Expected outputs
- Predicted scrap status and ranked root cause per unit
- Dashboard: scrap rate by root cause (not just by series/type)

## Key risks and assumptions
- Class imbalance across root-cause categories
- Data volume sufficiency
- [others]
