# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:07:46 2026

@author: JESSI
"""

# ==========================================
# SALES PREDICTION MODEL TRAINING
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load Dataset
df = pd.read_csv("Advertising.csv")

# Remove unwanted index column if present
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Features and Target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, y_pred)

print("Model Trained Successfully")
print("R² Score:", round(score, 4))

# Save Model
joblib.dump(model, "sales_model.pkl")

print("Model Saved as sales_model.pkl")