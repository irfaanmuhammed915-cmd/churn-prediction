"""
Customer Churn Prediction — Streamlit App
==========================================
Interactive UI to score individual customers and explain predictions via SHAP.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "churn_model.pkl"
META_PATH  = ROOT / "model" / "feature_meta.json"
METRICS_PATH = ROOT / "model" / "metrics.json"
ASSETS_DIR = ROOT / "assets"


# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return pipeline, meta, metrics


pipeline, meta, metrics = load_model()
NUM_FEATURES = meta["num"]
CAT_FEATURES = meta["cat"]

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: #1e293b; border-radius: 12px; padding: 20px;
    text-align: center; border: 1px solid #334155;
}
.metric-card h2 { color: #38bdf8; font-size: 2rem; margin: 0; }
.metric-card p  { color: #94a3b8; font-size: 0.85rem; margin: 4px 0 0; }
.risk-high   { background: #450a0a; border: 2px solid #ef4444; border-radius: 12px; padding: 20px; }
.risk-medium { background: #431407; border: 2px solid #f97316; border-radius: 12px; padding: 20px; }
.risk-low    { background: #052e16; border: 2px solid #22c55e; border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar — Customer Input ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Customer Profile")
    st.markdown("---")

    st.markdown("**Account**")
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    st.markdown("**Financials**")
    monthly_charges = st.slider("Monthly Charges (₹)", 18.0, 120.0, 65.0, step=0.5)
    total_charges = st.number_input(
        "Total Charges (₹)", min_value=0.0,
        value=float(round(tenure * monthly_charges, 2))
    )

    st.markdown("**Demographics**")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    senior = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Partner", ["Yes", "No"], horizontal=True)
    dependents = st.radio("Dependents", ["Yes", "No"], horizontal=True)

    st.markdown("**Services**")
    phone = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
    multiple_lines = st.radio(
        "Multiple Lines",
        ["Yes", "No"] if phone == "Yes" else ["No phone service"],
        horizontal=True
    )
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    no_internet = internet == "No"
    _opt = lambda label, disabled=no_internet: st.radio(
        label, ["No internet service"] if disabled else ["Yes", "No"],
        horizontal=True
    )
    online_security  = _opt("Online Security")
    online_backup    = _opt("Online Backup")
    device_prot      = _opt("Device Protection")
    tech_support     = _opt("Tech Support")
    streaming_tv     = _opt("Streaming TV")
    streaming_movies = _opt("Streaming Movies")

    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("# 📡 Customer Churn Predictor")
st.markdown("XGBoost + SHAP explainability · Telecom churn scoring")
st.markdown("---")

# Model performance summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <h2>{metrics['cv_auc_mean']:.3f}</h2>
        <p>CV AUC-ROC (5-fold)</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <h2>{metrics['test_auc_roc']:.3f}</h2>
        <p>Test AUC-ROC</p>
    </div>""", unsafe_allow_html=True)
with col3:
    f1 = metrics["classification_report"]["1"]["f1-score"]
    st.markdown(f"""<div class="metric-card">
        <h2>{f1:.3f}</h2>
        <p>F1 Score (Churn class)</p>
    </div>""", unsafe_allow_html=True)
with col4:
    recall = metrics["classification_report"]["1"]["recall"]
    st.markdown(f"""<div class="metric-card">
        <h2>{recall:.3f}</h2>
        <p>Recall (Churn class)</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    # Build input row
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Feature engineering (same as train.py)
    row["ChargesPerMonth"] = round(total_charges / max(tenure, 1), 2)
    row["NumAddOns"] = sum([
        online_security == "Yes", online_backup == "Yes",
        device_prot == "Yes", tech_support == "Yes",
        streaming_tv == "Yes", streaming_movies == "Yes"
    ])
    row["HasFiberAndNoSecurity"] = int(internet == "Fiber optic" and online_security == "No")
    row["IsMonthToMonth"] = int(contract == "Month-to-month")

    X_input = pd.DataFrame([row])[NUM_FEATURES + CAT_FEATURES]
    prob = pipeline.predict_proba(X_input)[0][1]
    pred = int(prob >= 0.5)

    # Risk display
    pcol1, pcol2 = st.columns([1, 2])
    with pcol1:
        if prob >= 0.7:
            cls, emoji, label = "risk-high", "🚨", "HIGH RISK"
        elif prob >= 0.4:
            cls, emoji, label = "risk-medium", "⚠️", "MEDIUM RISK"
        else:
            cls, emoji, label = "risk-low", "✅", "LOW RISK"

        st.markdown(f"""<div class="{cls}">
            <h3 style="margin:0">{emoji} {label}</h3>
            <h1 style="margin:8px 0;font-size:3rem">{prob:.1%}</h1>
            <p style="color:#94a3b8">Probability of churn</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        recommendation = {
            "risk-high":   "🎯 **Action**: Offer retention package immediately — discount or contract upgrade.",
            "risk-medium": "📞 **Action**: Proactive outreach to assess satisfaction and add-on value.",
            "risk-low":    "💚 **Action**: Maintain engagement; consider upsell opportunity.",
        }[cls]
        st.info(recommendation)

    # SHAP waterfall for this prediction
    with pcol2:
        with st.spinner("Computing SHAP explanation …"):
            preprocessor = pipeline.named_steps["preprocessor"]
            model = pipeline.named_steps["model"]
            X_transformed = preprocessor.transform(X_input)

            cat_encoder = preprocessor.named_transformers_["cat"]
            cat_names = cat_encoder.get_feature_names_out(CAT_FEATURES).tolist()
            feature_names = NUM_FEATURES + cat_names

            X_arr = X_transformed[:, :len(feature_names)]
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_arr)

            # Top-10 features by |SHAP|
            top_idx = np.argsort(np.abs(sv[0]))[-10:][::-1]
            top_names = [feature_names[i] for i in top_idx]
            top_vals  = sv[0][top_idx]

            colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_vals]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(range(len(top_vals)), top_vals[::-1], color=colors[::-1])
            ax.set_yticks(range(len(top_vals)))
            ax.set_yticklabels(top_names[::-1], fontsize=9)
            ax.axvline(0, color="white", lw=0.8)
            ax.set_xlabel("SHAP Value (impact on churn probability)", fontsize=9)
            ax.set_title("Why this prediction?", fontsize=12, fontweight="bold")
            fig.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")


# ── Model insights tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 ROC Curve", "🔍 SHAP Importance", "📋 About"])

with tab1:
    roc_path = ASSETS_DIR / "roc_curve.png"
    cm_path = ASSETS_DIR / "confusion_matrix.png"
    if roc_path.exists() and cm_path.exists():
        c1, c2 = st.columns(2)
        c1.image(str(roc_path), use_container_width=True)
        c2.image(str(cm_path), use_container_width=True)
    else:
        st.warning("Run `python train.py` to generate plots.")

with tab2:
    shap_path = ASSETS_DIR / "shap_summary.png"
    shap_bar   = ASSETS_DIR / "shap_bar.png"
    if shap_path.exists():
        c1, c2 = st.columns(2)
        c1.image(str(shap_path), caption="SHAP Beeswarm", use_container_width=True)
        c2.image(str(shap_bar),  caption="Mean |SHAP|",   use_container_width=True)

with tab3:
    st.markdown("""
### Customer Churn Prediction

**Model**: XGBoost Classifier with class-weight balancing  
**Explainability**: SHAP (SHapley Additive exPlanations) — TreeExplainer  
**Dataset**: Synthetic Telecom dataset (7 043 customers, ~30% churn)

#### Feature Engineering
| Feature | Description |
|---|---|
| `ChargesPerMonth` | TotalCharges / tenure — spending normalised by loyalty |
| `NumAddOns` | Count of value-added services subscribed |
| `HasFiberAndNoSecurity` | High-risk flag: Fiber internet without security subscription |
| `IsMonthToMonth` | Binary flag for most at-risk contract type |

#### Tech Stack
`Python` · `XGBoost` · `SHAP` · `scikit-learn` · `Streamlit` · `pandas` · `matplotlib`

---
Built by **Irfaan Muhammed** · [GitHub](https://github.com/irfaanmuhammed915-cmd) · [LinkedIn](https://www.linkedin.com/in/irfaan-muhammed-6002b2275/)
    """)
