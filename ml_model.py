import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_FILE = "First_Part.csv"


print("Loading dataset sample...")

# Only load small portion
df = pd.read_csv(DATA_FILE, nrows=5000)

# Rename columns to easier names
df.columns = [
    "index",
    "time",
    "speed",
    "torque",
    "dc_voltage",
    "duty_a",
    "duty_b",
    "duty_c",
    "current_a",
    "current_b",
    "current_c"
]

print("Dataset loaded:", df.shape)

# Feature Engineering

features = pd.DataFrame()

features["speed"] = df["speed"]
features["torque"] = df["torque"]

features["current_mean"] = (
    df["current_a"] +
    df["current_b"] +
    df["current_c"]
) / 3

features["current_imbalance"] = (
    abs(df["current_a"] - df["current_b"]) +
    abs(df["current_b"] - df["current_c"]) +
    abs(df["current_c"] - df["current_a"])
)

features["duty_mean"] = (
    df["duty_a"] +
    df["duty_b"] +
    df["duty_c"]
) / 3

features["dc_voltage"] = df["dc_voltage"]

features = features.dropna()

print("Feature matrix:", features.shape)

# Normalize

scaler = StandardScaler()

X = scaler.fit_transform(features)

# Train Model

print("Training anomaly detection model...")

model = IsolationForest(
    n_estimators=80,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

model.fit(X)

print("Training complete")

# Save Model

joblib.dump(model, "motor_fault_model.pkl")
joblib.dump(scaler, "motor_scaler.pkl")

print("Model saved successfully")
