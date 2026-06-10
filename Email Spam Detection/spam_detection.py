# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 22:04:22 2026

@author: JESSI
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📧 Email Spam Detection App")

# Load Dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Features and Target
X = df["message"]
y = df["label"]

# Text Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# User Input
message = st.text_area("Enter Email/SMS Message")

if st.button("Predict"):
    data = vectorizer.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam Message")