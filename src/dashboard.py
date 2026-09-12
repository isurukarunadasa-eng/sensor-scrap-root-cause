"""
dashboard.py

Streamlit dashboard: upload a new FT report, get back predicted status and
ranked root cause per unit, plus scrap-rate-by-root-cause trend views.

Run with: streamlit run src/dashboard.py

TODO: build the upload -> parse -> predict -> display flow.
"""

import streamlit as st

st.title("Sensor Scrap Root Cause Dashboard")
st.write("Upload an FT report to get predicted scrap status and likely root cause.")

# TODO: file uploader, call parser.py + model.py, display results
