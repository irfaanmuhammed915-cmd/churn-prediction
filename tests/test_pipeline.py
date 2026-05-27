"""
Unit tests for the churn prediction pipeline.
Run: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.generate_data import generate
from train import (
    build_model,
    build_preprocessor,
    engineer_features,
    load_data,
    NUM_FEATURES,
    CAT_FEATURES,
    TARGET,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def raw_df():
    return load_data(ROOT / "data" / "telco_churn.csv")


@pytest.fixture(scope="module")
def engineered_df(raw_df):
    return engineer_features(raw_df)


# ── Data tests ─────────────────────────────────────────────────────────────────
class TestDataGeneration:
    def test_shape(self):
        df = generate()
        assert df.shape == (7043, 21)

    def test_churn_rate(self):
        df = generate()
        churn_rate = df["Churn"].mean()
        assert 0.15 <= churn_rate <= 0.55, f"Unusual churn rate: {churn_rate:.2%}"

    def test_no_nulls_after_generation(self):
        df = generate()
        assert df.isnull().sum().sum() == 0

    def test_customer_ids_unique(self):
        df = generate()
        assert df["customerID"].nunique() == len(df)


class TestDataLoading:
    def test_loads_without_error(self, raw_df):
        assert raw_df is not None
        assert len(raw_df) > 0

    def test_target_is_binary(self, raw_df):
        assert set(raw_df[TARGET].unique()).issubset({0, 1})

    def test_no_customerid_column(self, raw_df):
        """customerID should be dropped during load."""
        assert "customerID" not in raw_df.columns

    def test_total_charges_numeric(self, raw_df):
        assert pd.api.types.is_numeric_dtype(raw_df["TotalCharges"])


class TestFeatureEngineering:
    def test_new_features_exist(self, engineered_df):
        for col in ["ChargesPerMonth", "NumAddOns", "HasFiberAndNoSecurity", "IsMonthToMonth"]:
            assert col in engineered_df.columns, f"Missing: {col}"

    def test_num_addons_range(self, engineered_df):
        assert engineered_df["NumAddOns"].between(0, 6).all()

    def test_binary_features(self, engineered_df):
        for col in ["HasFiberAndNoSecurity", "IsMonthToMonth"]:
            assert set(engineered_df[col].unique()).issubset({0, 1})

    def test_charges_per_month_positive(self, engineered_df):
        assert (engineered_df["ChargesPerMonth"] >= 0).all()


class TestPreprocessor:
    def test_builds_without_error(self):
        prep = build_preprocessor(NUM_FEATURES, CAT_FEATURES)
        assert prep is not None

    def test_fit_transform(self, engineered_df):
        extra_num = ["ChargesPerMonth", "NumAddOns", "HasFiberAndNoSecurity", "IsMonthToMonth"]
        all_num = NUM_FEATURES + extra_num
        X = engineered_df[all_num + CAT_FEATURES]
        y = engineered_df[TARGET]
        prep = build_preprocessor(all_num, CAT_FEATURES)
        X_t = prep.fit_transform(X)
        assert X_t.shape[0] == len(X)
        assert not np.isnan(X_t).any()


class TestModel:
    def test_builds_without_error(self):
        model = build_model()
        assert model is not None

    def test_scale_pos_weight(self):
        model = build_model(scale_pos_weight=2.5)
        assert model.scale_pos_weight == 2.5


class TestTrainedModel:
    """Tests that run only if the model artefacts exist."""

    def test_model_file_exists(self):
        assert (ROOT / "model" / "churn_model.pkl").exists(), \
            "Run python train.py first"

    def test_metrics_file_exists(self):
        assert (ROOT / "model" / "metrics.json").exists()

    def test_auc_above_threshold(self):
        metrics_path = ROOT / "model" / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("metrics.json not found")
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert metrics["test_auc_roc"] >= 0.70, \
            f"AUC too low: {metrics['test_auc_roc']}"

    def test_pipeline_predict(self):
        model_path = ROOT / "model" / "churn_model.pkl"
        if not model_path.exists():
            pytest.skip("model not found")
        import joblib
        pipeline = joblib.load(model_path)
        extra_num = ["ChargesPerMonth", "NumAddOns", "HasFiberAndNoSecurity", "IsMonthToMonth"]
        all_num = NUM_FEATURES + extra_num
        sample = pd.DataFrame([{
            "SeniorCitizen": 0, "tenure": 24, "MonthlyCharges": 65.0,
            "TotalCharges": 1560.0, "ChargesPerMonth": 65.0,
            "NumAddOns": 2, "HasFiberAndNoSecurity": 0, "IsMonthToMonth": 0,
            "gender": "Male", "Partner": "Yes", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
        }])[all_num + CAT_FEATURES]
        prob = pipeline.predict_proba(sample)[0][1]
        assert 0.0 <= prob <= 1.0
