"""
dashboard.py -- monitoring dashboard for the root-cause pipeline (Streamlit).

Run from the repository root:
    pip install streamlit
    streamlit run src/dashboard.py

Tab 1 lets a failure analyst upload one or more Final Tester reports and see the
failed sensors with a predicted root-cause family, confidence, the FT parameters
behind the call, the baseline's answer and a review flag. It calls the same code as
src/predict_root_cause.py, so the dashboard and the command line always agree.

Tab 2 re-scores both methods on the held-out test set so the headline comparison
can be checked live instead of trusted from a report.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import predict_root_cause as prc  # noqa: E402
import rules_baseline  # noqa: E402

st.set_page_config(page_title="Sensor root-cause monitor", layout="wide")

DEFAULTS = {
    "train": ROOT / "data/processed/train_units.csv",
    "model": ROOT / "models/defect_classifier.joblib",
    "test_units": ROOT / "data/processed/test_units.csv",
    "test_features": ROOT / "data/processed/test_features.csv",
}

with st.sidebar:
    st.header("Files")
    paths = {k: st.text_input(k, str(v)) for k, v in DEFAULTS.items()}
    use_shap = st.checkbox("Explain predictions (SHAP, slower)", value=True)
    st.caption("Paths are relative to where you started Streamlit unless absolute.")

missing = [k for k in ("train", "model") if not os.path.exists(paths[k])]
st.title("Sensor root-cause monitor")
st.caption("CS5998 capstone: failed 500-series inductive sensors, Final Tester data")
if missing:
    st.error("Missing file(s): " + ", ".join(f"{k} = {paths[k]}" for k in missing)
             + ". Run model.py first (the model file is not stored in git).")
    st.stop()

tab_predict, tab_compare = st.tabs(["Analyze a report", "Method comparison"])

# ------------------------------------------------------------------ tab 1
with tab_predict:
    uploads = st.file_uploader("Upload Final Tester report(s) (.xlsx)", type=["xlsx"],
                               accept_multiple_files=True)
    if st.button("Find failed sensors and predict root cause", disabled=not uploads):
        tmpdir = tempfile.mkdtemp()
        files = []
        for up in uploads:
            p = os.path.join(tmpdir, up.name)  # keep the name: the PO is parsed from it
            with open(p, "wb") as f:
                f.write(up.getbuffer())
            files.append(p)
        with st.spinner("Reading reports and predicting..."):
            try:
                df, summary = prc.predict_files(files, paths["train"], paths["model"],
                                                use_shap=use_shap, log=st.warning)
                st.session_state["result"] = (df, summary)
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("result", None)
                st.error(f"Prediction failed: {e}")

    if "result" in st.session_state:
        df, summary = st.session_state["result"]
        units = sum(s[1] for s in summary)
        failed = sum(s[2] for s in summary)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Units in report(s)", units)
        c2.metric("Failed (tester-flagged)", failed)
        c3.metric("Not flagged", units - failed)
        c4.metric("Need human review", int(df["needs_review"].sum()) if not df.empty else 0)
        st.caption("Unflagged units are not predicted on: FA can already see which sensors "
                   "passed. A sensor that fails without any tester flag is not caught.")
        if df.empty:
            st.info("No flagged units in the uploaded report(s).")
        else:
            left, right = st.columns([1, 2])
            with left:
                st.subheader("Predicted families")
                st.bar_chart(df["ml_prediction"].value_counts())
            with right:
                st.subheader("Failed sensors")
                only_review = st.checkbox("Show only units needing review")
                view = df[df["needs_review"]] if only_review else df
                show = ["source_file", "unit_index", "unique_number", "ml_prediction",
                        "ml_confidence", "baseline_prediction", "methods_agree",
                        "needs_review", "review_reason"]
                st.dataframe(view[show], hide_index=True)
            st.subheader("Unit detail")
            labels = [f"{r.source_file} / unit {r.unit_index}"
                      + (f" / serial {r.unique_number}" if str(r.unique_number) not in ("", "nan") else "")
                      for r in df.itertuples()]
            pick = st.selectbox("Unit", range(len(df)), format_func=lambda i: labels[i])
            r = df.iloc[pick]
            a, b = st.columns(2)
            with a:
                st.markdown("**Improved method (Random Forest)**")
                st.write(f"Prediction: {r['ml_prediction']} (confidence {r['ml_confidence']})")
                st.write(f"Runner-up: {r['ml_runner_up']}")
                st.write(f"Top FT parameters: {r['ml_top_features'] or 'SHAP off'}")
            with b:
                st.markdown("**Baseline (engineering rules)**")
                st.write(f"Prediction: {r['baseline_prediction']} (tier: {r['baseline_tier']})")
                st.write(r["baseline_reason"])
            st.write(f"Tester flags: {r['flagged_params']}")
            if r["needs_review"]:
                st.warning("Review: " + r["review_reason"])
            st.download_button("Download predictions.csv", df.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

# ------------------------------------------------------------------ tab 2
@st.cache_data(show_spinner="Scoring both methods on the test set...")
def score_both(test_units, test_features, model_path):
    feats = pd.read_csv(test_features)
    y = feats["model_target"].astype(str)
    model = joblib.load(model_path)
    cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else \
        [c for c in feats.columns if c not in ("PO", "model_target")]
    ml_pred = pd.Series(model.predict(feats[cols]), index=feats.index).astype(str)
    with open(test_units, encoding="latin-1") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(feats):
        raise ValueError(f"test_units has {len(rows)} rows but test_features has {len(feats)}")
    base_pred = pd.Series([rules_baseline.classify_unit(
        {k: ("" if v is None else str(v)) for k, v in r.items()})[0] for r in rows],
        index=feats.index).astype(str)
    labels = sorted(y.unique())
    table = pd.DataFrame({
        "Baseline F1": f1_score(y, base_pred, labels=labels, average=None, zero_division=0),
        "Random Forest F1": f1_score(y, ml_pred, labels=labels, average=None, zero_division=0),
        "Test units": [int((y == c).sum()) for c in labels]}, index=labels)
    headline = {
        "Baseline": (f1_score(y, base_pred, average="macro", zero_division=0), accuracy_score(y, base_pred)),
        "Random Forest": (f1_score(y, ml_pred, average="macro", zero_division=0), accuracy_score(y, ml_pred))}
    return table, headline, len(feats), feats["PO"].nunique() if "PO" in feats else None


with tab_compare:
    try:
        table, headline, n, n_po = score_both(paths["test_units"], paths["test_features"], paths["model"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not score the test set: {e}")
    else:
        st.write(f"Held-out test set: {n} units" + (f", {n_po} POs" if n_po else "")
                 + ". Both methods are scored on the same units.")
        m1, m2 = st.columns(2)
        for col, (name, (f1, acc)) in zip((m1, m2), headline.items()):
            col.metric(f"{name}: macro F1", f"{f1:.3f}", help=f"accuracy {acc:.0%}")
        st.dataframe(table.style.format({"Baseline F1": "{:.2f}", "Random Forest F1": "{:.2f}"}))
        st.bar_chart(table[["Baseline F1", "Random Forest F1"]])
        st.caption("Macro F1 is the headline metric because accuracy is misleading under class "
                   "imbalance. The gap comes almost entirely from the two smallest classes "
                   "(Insulation / Isolation and Gluing / Potting), so treat it as modest evidence.")
