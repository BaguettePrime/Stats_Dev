"""Entry point – routes to Data Explorer and Statistical Analysis pages."""

import streamlit as st

st.set_page_config(page_title="Time Series Analysis", layout="wide")

pg = st.navigation([
    st.Page("pages/Data_Explorer.py", title="Data Explorer", default=True),
    st.Page("pages/Statistical_Analysis.py", title="Statistical Analysis"),
])
pg.run()
