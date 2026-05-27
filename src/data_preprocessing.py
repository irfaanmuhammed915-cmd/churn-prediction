"""
data_preprocessing.py
─────────────────────
Loads, cleans, and feature-engineers the telecom churn dataset.

Dataset: IBM Telco Customer Churn (Kaggle)
         Fallback: synthetic dataset generated via make_classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Column groups ────────────────────────────────────────────────────────────
BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CATEGORICAL = ["InternetService", "Contract", "PaymentMethod", "gender"]
TARGET = "Churn"


def load_data(path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load CSV if available, else build a reproducible synthetic dataset."""
    fpath = Path(path)
    if fpath.exists():
        df = pd.read_csv(fpath)
        print(f"[INFO] Loaded real dataset: {df.shape}")
    else:
        print("[INFO] Real dataset not found → generating synthetic data…")
        df = _generate_synthetic(n=7043, seed=42)
    return df


def _generate_synthetic(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Mimic the Telco dataset schema with realistic distributions."""
    rng = np.random.default_rng(seed)

    tenure        = rng.integers(0, 73, n)
    monthly_charges = rng.uniform(18, 119, n).round(2)
    total_charges   = (tenure * monthly_charges * rng.uniform(0.95, 1.05, n)).round(2)

    contracts    = rng.choice(["Month-to-month", "One year", "Two year"],
                              n, p=[0.55, 0.24, 0.21])
    internet     = rng.choice(["DSL", "Fiber optic", "No"],
                              n, p=[0.34, 0.44, 0.22])
    payment      = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21])

    # Churn probability rises with Fiber + Month-to-month + high charges
    churn_logit = (
        -2.5
        + 0.03 * (monthly_charges - 65)
        - 0.04 * tenure
        + 1.2  * (contracts == "Month-to-month").astype(int)
        + 0.6  * (internet == "Fiber optic").astype(int)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn      = (rng.uniform(size=n) < churn_prob).astype(int)

    binary_cols = {
        col: rng.choice(["Yes", "No"], n)
        for col in BINARY_YES_NO
    }

    df = pd.DataFrame({
        "customerID":       [f"CUST-{i:05d}" for i in range(n)],
        "gender":           rng.choice(["Male", "Female"], n),
        "SeniorCitizen":    rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":          binary_cols["Partner"],
        "Dependents":       binary_cols["Dependents"],
        "tenure":           tenure,
        "PhoneService":     binary_cols["PhoneService"],
        "MultipleLines":    binary_cols["MultipleLines"],
        "InternetService":  internet,
        "OnlineSecurity":   binary_cols["OnlineSecurity"],
        "OnlineBackup":     binary_cols["OnlineBackup"],
        "DeviceProtection": binary_cols["DeviceProtection"],
        "TechSupport":      binary_cols["TechSupport"],
        "StreamingTV":      binary_cols["StreamingTV"],
        "StreamingMovies":  binary_cols["StreamingMovies"],
        "Contract":         contracts,
        "PaperlessBilling": binary_cols["PaperlessBilling"],
        "PaymentMethod":    payment,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges.astype(str),
        "Churn":            ["Yes" if c else "No" for c in churn],
    })
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, handle missing values, drop ID column."""
    df = df.copy()

    # TotalCharges sometimes has spaces instead of NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customer ID (no predictive value)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Binary-encode Yes/No columns; label-encode categoricals."""
    df = df.copy()

    # Yes/No → 1/0
    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = (df[col] == "Yes").astype(int)

    # Gender
    df["gender"] = (df["gender"] == "Male").astype(int)

    # Multi-category columns → label encode
    le = LabelEncoder()
    for col in ["InternetService", "Contract", "PaymentMethod"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Target
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed features that improve model signal."""
    df = df.copy()

    # Avg monthly spend proxy
    df["avg_monthly_charges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # High-value customer flag
    df["is_high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # Number of add-on services
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_addons"] = df[addon_cols].sum(axis=1)

    # Contract risk score (Month-to-month = 2, One year = 1, Two year = 0)
    # After label encoding Contract is 0/1/2 — map explicitly
    contract_risk = {0: 2, 1: 1, 2: 0}   # depends on LabelEncoder sort order
    df["contract_risk"] = df["Contract"].map(contract_risk).fillna(1)

    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Return X_train, X_test, y_train, y_test with stratification."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)


def get_preprocessed_data(path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Full pipeline: load → clean → encode → feature-engineer → split."""
    df = load_data(path)
    df = clean(df)
    df = encode(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split(df)
    print(f"[INFO] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[INFO] Churn rate — train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test, df
