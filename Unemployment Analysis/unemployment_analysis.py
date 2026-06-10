# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:21:29 2026

@author: JESSI
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Unemployment Analysis in India",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Unemployment Analysis in India")
st.markdown("### Analysis of Unemployment Rate During COVID-19")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Unemployment in India.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---------------------------------------------------
# SHOW DATA
# ---------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Columns in Dataset")
st.write(df.columns.tolist())

# ---------------------------------------------------
# FIND COLUMN NAMES AUTOMATICALLY
# ---------------------------------------------------
unemployment_col = None
employment_col = None
labour_col = None

for col in df.columns:
    if "Unemployment Rate" in col:
        unemployment_col = col
    elif "Employed" in col:
        employment_col = col
    elif "Labour Participation" in col:
        labour_col = col

# ---------------------------------------------------
# DATE CONVERSION
# ---------------------------------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Remove missing values
df = df.dropna()

# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------
st.subheader("Statistical Summary")
st.dataframe(df.describe())

# ---------------------------------------------------
# DISTRIBUTION
# ---------------------------------------------------
if unemployment_col:

    st.subheader("Distribution of Unemployment Rate")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        df[unemployment_col],
        kde=True,
        ax=ax
    )

    ax.set_title("Distribution of Unemployment Rate")

    st.pyplot(fig)

# ---------------------------------------------------
# STATE ANALYSIS
# ---------------------------------------------------
if unemployment_col and "Region" in df.columns:

    st.subheader("Average Unemployment Rate by State")

    state_avg = (
        df.groupby("Region")[unemployment_col]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(
        x=state_avg.values,
        y=state_avg.index,
        ax=ax
    )

    ax.set_title("Average Unemployment Rate by State")

    st.pyplot(fig)

# ---------------------------------------------------
# TOP 10 STATES
# ---------------------------------------------------
if unemployment_col and "Region" in df.columns:

    st.subheader("Top 10 States with Highest Unemployment")

    top10 = (
        df.groupby("Region")[unemployment_col]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=top10.values,
        y=top10.index,
        ax=ax
    )

    ax.set_title("Top 10 Highest Unemployment States")

    st.pyplot(fig)

# ---------------------------------------------------
# MONTHLY TREND
# ---------------------------------------------------
if unemployment_col and "Date" in df.columns:

    st.subheader("Monthly Unemployment Trend")

    monthly = (
        df.groupby("Date")[unemployment_col]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    monthly.plot(
        marker="o",
        ax=ax
    )

    ax.set_title("Monthly Unemployment Trend")
    ax.set_ylabel("Unemployment Rate (%)")

    st.pyplot(fig)

# ---------------------------------------------------
# COVID IMPACT
# ---------------------------------------------------
if unemployment_col and "Date" in df.columns:

    st.subheader("COVID-19 Impact on Unemployment")

    covid = df[df["Date"] >= "2020-03-01"]

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        data=covid,
        x="Date",
        y=unemployment_col,
        ax=ax
    )

    ax.set_title("COVID-19 Impact")

    st.pyplot(fig)

# ---------------------------------------------------
# SCATTER PLOT
# ---------------------------------------------------
if unemployment_col and labour_col:

    st.subheader("Labour Participation vs Unemployment")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=labour_col,
        y=unemployment_col,
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ---------------------------------------------------
# KEY INSIGHTS
# ---------------------------------------------------
if unemployment_col and "Region" in df.columns:

    state_avg = (
        df.groupby("Region")[unemployment_col]
        .mean()
        .sort_values(ascending=False)
    )

    highest_state = state_avg.idxmax()
    highest_rate = round(state_avg.max(), 2)

    lowest_state = state_avg.idxmin()
    lowest_rate = round(state_avg.min(), 2)

    avg_rate = round(df[unemployment_col].mean(), 2)

    st.subheader("Key Insights")

    st.success(
        f"Highest Unemployment State: {highest_state} ({highest_rate}%)"
    )

    st.success(
        f"Lowest Unemployment State: {lowest_state} ({lowest_rate}%)"
    )

    st.success(
        f"Average Unemployment Rate: {avg_rate}%"
    )

st.markdown("---")
st.markdown("Developed using Streamlit, Pandas, Matplotlib, and Seaborn")