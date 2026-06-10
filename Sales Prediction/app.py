# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:02:16 2026

@author: JESSI
"""

# ==========================================
# SALES PREDICTION STREAMLIT APP
# ==========================================

# ==========================================
# SALES PREDICTION STREAMLIT APP
# ==========================================

import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("sales_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📈",
    layout="centered"
)

# Title
st.title("📈 Sales Prediction using Machine Learning")

st.write("""
Predict product sales based on advertising budgets spent on:

- TV Advertising
- Radio Advertising
- Newspaper Advertising
""")

st.divider()

# User Inputs
tv = st.number_input(
    "TV Advertising Budget",
    min_value=0.0,
    value=100.0,
    step=1.0
)

radio = st.number_input(
    "Radio Advertising Budget",
    min_value=0.0,
    value=20.0,
    step=1.0
)

newspaper = st.number_input(
    "Newspaper Advertising Budget",
    min_value=0.0,
    value=10.0,
    step=1.0
)

# Prediction Button
if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Sales: {prediction[0]:.2f}"
    )

st.divider()

st.caption("Sales Prediction Project using Python, Scikit-Learn and Streamlit")