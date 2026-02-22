"""
app/streamlit_app.py
====================
Interactive Streamlit front-end for the Sri Lanka Apartment Price Prediction
model.  Provides:
    - User input form  (sidebar)
    - Predicted price display (prominent hero card)
    - Model performance summary  (RMSE, MAE, R²)
    - Global SHAP explanation  (saved summary plot image)
    - Local SHAP explanation   (waterfall plot for single prediction)

Usage
-----
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import json
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
PLOTS_DIR    = OUTPUTS_DIR  / "plots"

sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Apartment Price Predictor",
    page_icon="🏢",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS for dark theme styling
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Prediction hero card ── */
    .prediction-card {
        background: linear-gradient(135deg, #00D4AA 0%, #00B894 100%);
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        margin: 10px 0 20px 0;
        box-shadow: 0 8px 32px rgba(0, 212, 170, 0.25);
    }
    .prediction-card h2 {
        color: #0E1117 !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
    }
    .prediction-card p {
        color: #1A1F2B !important;
        font-size: 1rem;
        margin: 5px 0 0 0;
        opacity: 0.85;
    }

    /* ── Negative prediction card (rent / edge cases) ── */
    .prediction-card-warn {
        background: linear-gradient(135deg, #FDCB6E 0%, #E17055 100%);
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        margin: 10px 0 20px 0;
        box-shadow: 0 8px 32px rgba(225, 112, 85, 0.25);
    }
    .prediction-card-warn h2 {
        color: #0E1117 !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }
    .prediction-card-warn p {
        color: #1A1F2B !important;
        font-size: 1rem;
        margin: 5px 0 0 0;
        opacity: 0.85;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #1A1F2B;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-card h3 {
        color: #00D4AA !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .metric-card p {
        color: #A0AEC0 !important;
        font-size: 0.85rem;
        margin: 4px 0 0 0;
    }

    /* ── Section headers ── */
    .section-header {
        border-left: 4px solid #00D4AA;
        padding-left: 12px;
        margin: 25px 0 15px 0;
    }
    .section-header h3 {
        color: #FAFAFA !important;
        margin: 0 !important;
    }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2D3748, transparent);
        margin: 20px 0;
    }

    /* ── Info box for no-prediction state ── */
    .info-box {
        background: #1A1F2B;
        border: 1px dashed #2D3748;
        border-radius: 12px;
        padding: 40px 20px;
        text-align: center;
        color: #A0AEC0;
        font-size: 1.1rem;
    }

    /* ── Sidebar button styling ── */
    div[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #00D4AA 0%, #00B894 100%);
        color: #0E1117;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 12px 0;
        width: 100%;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #00B894 0%, #009B77 100%);
        color: #0E1117;
    }

    /* ── Hide default streamlit footer ── */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "xgb_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load(MODELS_DIR / "label_encoders.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load(MODELS_DIR / "scaler.pkl")

@st.cache_resource
def load_feature_cols():
    return joblib.load(MODELS_DIR / "feature_columns.pkl")

@st.cache_data
def load_metrics():
    p = OUTPUTS_DIR / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────
# Sidebar - User inputs
# ──────────────────────────────────────────────────────────────────────

def build_sidebar(encoders: dict, feature_cols: list):
    """Create sidebar input widgets based on the feature columns."""
    st.sidebar.markdown("## 🏢 Apartment Details")
    st.sidebar.markdown("Fill in the details below to get a price prediction.")
    st.sidebar.markdown("---")

    inputs = {}

    # ── Categorical inputs ──
    for col in ("district", "location", "listing_type",
                "property_type", "apartment_model"):
        if col in encoders and col in feature_cols:
            classes = list(encoders[col].classes_)
            pretty = col.replace("_", " ").title()
            inputs[col] = st.sidebar.selectbox(pretty, sorted(classes))

    st.sidebar.markdown("---")

    # ── Numeric inputs ──
    if "bedrooms" in feature_cols:
        inputs["bedrooms"] = st.sidebar.slider("Bedrooms", 1, 10, 3)
    if "bathrooms" in feature_cols:
        inputs["bathrooms"] = st.sidebar.slider("Bathrooms", 1, 6, 2)
    if "property_size_sqft" in feature_cols:
        inputs["property_size_sqft"] = st.sidebar.number_input(
            "Apartment Size (sq ft)", min_value=100, max_value=20000,
            value=1200, step=50)
    if "floor" in feature_cols:
        inputs["floor"] = st.sidebar.number_input(
            "Floor Number", min_value=0, max_value=80, value=5, step=1)

    st.sidebar.markdown("---")

    if "is_furnished" in feature_cols:
        inputs["is_furnished"] = st.sidebar.selectbox(
            "Furnished?", ["No", "Yes"])
        inputs["is_furnished"] = 1 if inputs["is_furnished"] == "Yes" else 0
    if "has_parking" in feature_cols:
        inputs["has_parking"] = st.sidebar.selectbox(
            "Parking Available?", ["No", "Yes"])
        inputs["has_parking"] = 1 if inputs["has_parking"] == "Yes" else 0
    if "has_gym_pool" in feature_cols:
        inputs["has_gym_pool"] = st.sidebar.selectbox(
            "Gym / Pool?", ["No", "Yes"])
        inputs["has_gym_pool"] = 1 if inputs["has_gym_pool"] == "Yes" else 0

    return inputs


def prepare_input(inputs: dict, encoders: dict, scaler,
                  feature_cols: list) -> pd.DataFrame:
    """Convert user inputs into a model-ready DataFrame."""
    row = {}

    cat_cols = [c for c in ("district", "location", "listing_type",
                            "property_type", "apartment_model")
                if c in feature_cols]
    num_cols = [c for c in ("bedrooms", "bathrooms", "property_size_sqft",
                            "floor", "is_furnished", "has_parking",
                            "has_gym_pool")
                if c in feature_cols]

    for col in cat_cols:
        val = inputs.get(col, encoders[col].classes_[0])
        row[col] = encoders[col].transform([val])[0]

    for col in num_cols:
        row[col] = inputs.get(col, 0)

    df = pd.DataFrame([row], columns=feature_cols)

    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

    return df


# ──────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("# 🏢 Sri Lanka Apartment Price Predictor")
    st.markdown(
        "Predict apartment prices in Sri Lanka using a trained "
        "**XGBoost** model with **SHAP** explainability.  "
        "Enter the apartment details in the sidebar and click **Predict**."
    )
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Load artefacts ──
    try:
        model        = load_model()
        encoders     = load_encoders()
        scaler       = load_scaler()
        feature_cols = load_feature_cols()
    except FileNotFoundError as e:
        st.error(
            f"Model files not found. Run the training pipeline first.\n\n{e}"
        )
        st.stop()

    metrics = load_metrics()

    # ── Sidebar ──
    inputs = build_sidebar(encoders, feature_cols)
    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🔍  Predict Price")

    # ════════════════════════════════════════════════════════════════════
    # ROW 1 — Prediction (full width, prominent)
    # ════════════════════════════════════════════════════════════════════
    if predict_btn:
        X_input = prepare_input(inputs, encoders, scaler, feature_cols)
        pred = model.predict(X_input)[0]

        # Show the prediction as a big hero card
        if pred >= 0:
            st.markdown(
                f"""<div class="prediction-card">
                    <p>Estimated Apartment Price</p>
                    <h2>LKR {pred:,.0f}</h2>
                    <p>Based on your selected apartment features</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="prediction-card-warn">
                    <p>Estimated Apartment Price</p>
                    <h2>LKR {abs(pred):,.0f}</h2>
                    <p>⚠️ The model predicts a low/negative value — try adjusting inputs (e.g., change listing type to Sale)</p>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Metrics row ──
        if metrics:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>{metrics['RMSE']:,.0f}</h3>
                        <p>RMSE (Root Mean Squared Error)</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>{metrics['MAE']:,.0f}</h3>
                        <p>MAE (Mean Absolute Error)</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with mc3:
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>{metrics['R2']:.4f}</h3>
                        <p>R² Score (1.0 = perfect)</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ════════════════════════════════════════════════════════════════
        # ROW 2 — SHAP explanations side-by-side
        # ════════════════════════════════════════════════════════════════
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        left_col, right_col = st.columns(2)

        # ── Left: Local SHAP waterfall ──
        with left_col:
            st.markdown(
                '<div class="section-header"><h3>🔬 Why This Price?</h3></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "The waterfall chart shows **how each feature pushed the "
                "price up or down** for *your specific apartment*."
            )
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer(X_input)

                pretty = [c.replace("_", " ").title() for c in feature_cols]
                shap_vals.feature_names = pretty

                fig_w, _ = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(shap_vals[0], show=False)
                plt.title("Feature Contributions", fontsize=13, color="white")
                fig_w = plt.gcf()
                fig_w.patch.set_facecolor("#0E1117")
                for ax in fig_w.axes:
                    ax.set_facecolor("#0E1117")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    for spine in ax.spines.values():
                        spine.set_color("#2D3748")
                plt.tight_layout()
                st.pyplot(fig_w)
                plt.close(fig_w)
            except Exception as ex:
                st.warning(f"Could not generate local SHAP plot: {ex}")

        # ── Right: Global SHAP summary ──
        with right_col:
            st.markdown(
                '<div class="section-header"><h3>🌍 Global Feature Impact</h3></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "Which features have the **biggest impact** on apartment "
                "prices across *all* predictions."
            )
            shap_img = PLOTS_DIR / "shap_summary.png"
            if shap_img.exists():
                st.image(str(shap_img))
            else:
                st.info("Run `python src/explain.py` to generate the SHAP plot.")

        # ════════════════════════════════════════════════════════════════
        # ROW 3 — Feature importance bar chart (full width)
        # ════════════════════════════════════════════════════════════════
        fi_img = PLOTS_DIR / "feature_importance_bar.png"
        if fi_img.exists():
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header"><h3>📈 Feature Importance</h3></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "This bar chart ranks features by their **average impact** "
                "on predictions. Taller bars = more important features."
            )
            st.image(str(fi_img))

    else:
        # ── No prediction yet — show welcome state ──
        st.markdown(
            """<div class="info-box">
                👈 &nbsp; Fill in the apartment details in the sidebar
                and click <strong>Predict Price</strong> to see results here.
            </div>""",
            unsafe_allow_html=True,
        )

        # Still show metrics even before prediction
        if metrics:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header"><h3>📊 Model Performance</h3></div>',
                unsafe_allow_html=True,
            )
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>{metrics['RMSE']:,.0f}</h3>
                        <p>RMSE</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>{metrics['MAE']:,.0f}</h3>
                        <p>MAE</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with mc3:
                st.markdown(
                    f"""<div class="metric-card">
                        <h3>{metrics['R2']:.4f}</h3>
                        <p>R² Score</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Footer ──
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#4A5568; font-size:0.8rem;'>"
        "Built with XGBoost + SHAP + Streamlit &nbsp;|&nbsp; "
        "Data from properties.lk &nbsp;|&nbsp; "
        "2026 Academic Project"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
