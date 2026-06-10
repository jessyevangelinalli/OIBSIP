# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:15:25 2026

@author: JESSI
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# Load Dataset
df = pd.read_csv("iris.csv")

# Features (excluding Id column)
X = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]

# Target
y = df["Species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Title
st.title("🌸 Iris Flower Classification System")
st.markdown("---")

st.write("Enter the flower measurements below:")

# Empty Input Fields
sepal_length = st.text_input("Sepal Length (cm)")
sepal_width = st.text_input("Sepal Width (cm)")
petal_length = st.text_input("Petal Length (cm)")
petal_width = st.text_input("Petal Width (cm)")

# Prediction
if st.button("Predict Species"):

    if (
        sepal_length.strip() == "" or
        sepal_width.strip() == "" or
        petal_length.strip() == "" or
        petal_width.strip() == ""
    ):
        st.warning("Please enter all measurements.")

    else:
        try:
            sample = [[
                float(sepal_length),
                float(sepal_width),
                float(petal_length),
                float(petal_width)
            ]]

            prediction = model.predict(sample)
            species = prediction[0]

            if species == "Iris-setosa":
                st.success("🌸 Predicted Species: Iris Setosa")

            elif species == "Iris-versicolor":
                st.success("🌺 Predicted Species: Iris Versicolor")

            elif species == "Iris-virginica":
                st.success("🌷 Predicted Species: Iris Virginica")

            else:
                st.success(f"Predicted Species: {species}")

        except ValueError:
            st.error("Please enter valid numeric values.")

st.markdown("---")
st.caption("Iris Flower Classification using Machine Learning and Streamlit")