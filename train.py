"""
Customer Churn Prediction — Training Pipeline
=============================================
Trains an XGBoost classifier on telecom churn data.
Outputs: trained model, preprocessor, feature names, and performance metrics.

Usage:
    python train.py                   # uses default data path
    python train.py --data path/to/data.csv
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "model"
ASSETS_DIR = ROOT / "assets"
MODEL_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)


# ── Feature definitions ────────────────────────────────────────────────────────
CAT_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUM_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"
DROP_COLS = ["customerID"]


# ── Data loading & cleaning ────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    log.info("Loading data from %s", path)
    df = pd.read_csv(path)
    # TotalCharges can be blank string in real Telco datasets
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    log.info("Loaded %d rows | churn rate %.1f%%", len(df), df[TARGET].mean() * 100)
    return df


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ChargesPerMonth"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).round(2)
    df["NumAddOns"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["HasFiberAndNoSecurity"] = (
        (df["InternetService"] == "Fiber optic") & (df["OnlineSecurity"] == "No")
    ).astype(int)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


# ── Preprocessing ──────────────────────────────────────────────────────────────
def build_preprocessor(num_features, cat_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="passthrough",
    )
    return preprocessor


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


# ── Training ───────────────────────────────────────────────────────────────────
def train(data_path: Path = DATA_PATH):
    df = load_data(data_path)
    df = engineer_features(df)

    # Update feature lists after engineering
    extra_num = ["ChargesPerMonth", "NumAddOns", "HasFiberAndNoSecurity", "IsMonthToMonth"]
    all_num = NUM_FEATURES + extra_num
    all_cat = CAT_FEATURES

    X = df[all_num + all_cat]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # Class imbalance ratio
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = round(neg / pos, 2)
    log.info("scale_pos_weight = %.2f", spw)

    preprocessor = build_preprocessor(all_num, all_cat)
    model = build_model(scale_pos_weight=spw)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Cross-validation
    log.info("Running 5-fold CV …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info("CV AUC: %.4f ± %.4f", cv_auc.mean(), cv_auc.std())

    # Final fit
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    log.info("Test AUC-ROC : %.4f", auc)
    log.info("Avg Precision: %.4f", ap)
    log.info("\n%s", classification_report(y_test, y_pred))

    # Persist artefacts
    joblib.dump(pipeline, MODEL_DIR / "churn_model.pkl")
    feature_meta = {"num": all_num, "cat": all_cat}
    with open(MODEL_DIR / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    metrics = {
        "cv_auc_mean": round(cv_auc.mean(), 4),
        "cv_auc_std": round(cv_auc.std(), 4),
        "test_auc_roc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "classification_report": report,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Model saved → %s", MODEL_DIR / "churn_model.pkl")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_roc(y_test, y_prob, auc)
    _plot_confusion(y_test, y_pred)
    _plot_shap(pipeline, X_test, all_num, all_cat)

    return pipeline, metrics


def _plot_roc(y_test, y_prob, auc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#2563EB", label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Churn Prediction", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)
    log.info("Saved ROC curve → %s", ASSETS_DIR / "roc_curve.png")


def _plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", ASSETS_DIR / "confusion_matrix.png")


def _plot_shap(pipeline, X_test, num_features, cat_features):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_test)
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    feature_names = num_features + cat_names

    # passthrough columns (none in our case, but handle generically)
    n_num = len(num_features)
    n_cat = len(cat_names)
    feature_names = feature_names[:n_num + n_cat]
    X_arr = X_transformed[:, :n_num + n_cat]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)

    # Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_arr,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance (Top 15)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("Saved SHAP summary → %s", ASSETS_DIR / "shap_summary.png")

    # Bar plot
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_arr,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_type="bar",
        plot_size=None,
    )
    plt.title("Mean |SHAP| — Feature Importance", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("Saved SHAP bar → %s", ASSETS_DIR / "shap_bar.png")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to CSV")
    args = parser.parse_args()
    train(Path(args.data))
