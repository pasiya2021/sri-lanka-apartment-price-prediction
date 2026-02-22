"""
app/streamlit_app.py
====================
Interactive Streamlit front‑end for the Sri Lanka Apartment Price Prediction
model.  Provides:
    • User input form  (sidebar)
    • Predicted price display
    • Model performance summary  (RMSE, MAE, R²)
    • Global SHAP explanation  (saved summary plot image)
    • Local SHAP explanation   (waterfall plot for single prediction)

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

@st.cache_data
def load_processed_data():
    p = PROJECT_ROOT / "processed.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


# ──────────────────────────────────────────────────────────────────────
# Sidebar – User inputs
# ──────────────────────────────────────────────────────────────────────

def build_sidebar(encoders: dict, feature_cols: list):
    """Create sidebar input widgets based on the feature columns."""
    st.sidebar.header("🏢  Apartment Details")
    st.sidebar.markdown("Fill in the details below to get a price prediction.")

    inputs = {}

    # ── Categorical inputs ──
    for col in ("district", "location", "listing_type",
                "property_type", "apartment_model"):
        if col in encoders and col in feature_cols:
            classes = list(encoders[col].classes_)
            pretty = col.replace("_", " ").title()
            inputs[col] = st.sidebar.selectbox(pretty, sorted(classes))

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
    """Convert user inputs into a model‑ready DataFrame."""
    row = {}

    # Categorical columns
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

    # Scale numeric columns
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

    return df


# ──────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("🏢  Sri Lanka Apartment Price Predictor")
    st.markdown(
        "Predict apartment prices in Sri Lanka using a trained "
        "**XGBoost** model.  Enter the apartment details in the sidebar "
        "and click **Predict**."
    )

    # Load artefacts
    try:
        model        = load_model()
        encoders     = load_encoders()
        scaler       = load_scaler()
        feature_cols = load_feature_cols()
    except FileNotFoundError as e:
        st.error(f"Model files not found.  Run the training pipeline first.\n\n{e}")
        st.stop()

    metrics = load_metrics()

    # Sidebar
    inputs = build_sidebar(encoders, feature_cols)

    # Predict button
    st.sidebar.markdown("---")
    predict_btn = st.sidebar.button("🔍  Predict Price")

    # ── Two‑column layout ──
    col1, col2 = st.columns(2)

    # ── Column 1: Prediction ──
    with col1:
        st.subheader("💰  Price Prediction")
        if predict_btn:
            X_input = prepare_input(inputs, encoders, scaler, feature_cols)
            pred = model.predict(X_input)[0]

            st.metric(label="Estimated Apartment Price",
                      value=f"LKR {pred:,.0f}")
            st.caption("This is the model's best estimate based on the inputs "
                       "you provided.")

            # ── Local SHAP explanation ──
            st.markdown("---")
            st.subheader("🔬  Why This Price? (Local Explanation)")
            st.markdown(
                "The waterfall chart below shows **how each feature pushed "
                "the price up or down** for *your specific apartment*.  "
                "Red bars push the price higher; blue bars push it lower."
            )

            try:
                import shap
                explainer   = shap.TreeExplainer(model)
                shap_vals   = explainer(X_input)

                pretty = [c.replace("_", " ").title() for c in feature_cols]
                shap_vals.feature_names = pretty

                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(shap_vals[0], show=False)
                plt.title("Feature Contributions to This Prediction",
                          fontsize=13)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as ex:
                st.warning(f"Could not generate local SHAP plot: {ex}")

        else:
            st.info("👈  Fill in details in the sidebar and click **Predict**.")

    # ── Column 2: Performance + Global Explanation ──
    with col2:
        st.subheader("📊  Model Performance")
        if metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{metrics['RMSE']:,.0f}")
            m2.metric("MAE",  f"{metrics['MAE']:,.0f}")
            m3.metric("R² Score", f"{metrics['R2']:.4f}")
            st.caption(
                "**RMSE** = average prediction error (lower is better).  "
                "**MAE** = average absolute error.  "
                "**R²** = how well the model explains price variation "
                "(1.0 = perfect)."
            )
        else:
            st.warning("No metrics file found. Run `python src/evaluate.py` first.")

        st.markdown("---")
        st.subheader("🌍  Global Feature Explanation (SHAP)")
        st.markdown(
            "The summary plot below shows **which features have the biggest "
            "impact** on apartment prices across *all* predictions.\n\n"
            "- Each dot is one apartment.\n"
            "- **Red** dots = high feature value; **Blue** dots = low.\n"
            "- Dots further from the centre line have a stronger effect "
            "on the predicted price."
        )
        shap_img = PLOTS_DIR / "shap_summary.png"
        if shap_img.exists():
            st.image(str(shap_img), use_container_width=False)
        else:
            st.info("Run `python src/explain.py` to generate the SHAP summary plot.")

        # Feature importance bar chart
        fi_img = PLOTS_DIR / "feature_importance_bar.png"
        if fi_img.exists():
            st.subheader("📈  Feature Importance")
            st.markdown(
                "This bar chart ranks features by their **average impact** "
                "on predictions.  Taller bars = more important features."
            )
            st.image(str(fi_img), use_container_width=False)


if __name__ == "__main__":
    main()
