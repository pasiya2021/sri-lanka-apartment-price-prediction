"""
src/explain.py
==============
Generate SHAP‑based model explanations for the trained XGBoost regressor.

SHAP (SHapley Additive exPlanations) is a game‑theory approach that explains
the output of any ML model by computing the contribution of each feature to
a particular prediction.

Outputs  (saved to outputs/plots/)
-------
    shap_summary.png           – Global view: how every feature pushes
                                  predictions higher or lower.
    shap_dependence.png        – How the single most important feature
                                  interacts with the predicted price.
    feature_importance_bar.png – Simple bar chart of mean |SHAP| values
                                  (average impact of each feature).

Usage
-----
    python src/explain.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, ensure_dirs, MODELS_DIR, PLOTS_DIR

log = get_logger("explain")


def main():
    ensure_dirs()

    model_path  = MODELS_DIR / "xgb_model.pkl"
    splits_path = MODELS_DIR / "data_splits.pkl"
    feat_path   = MODELS_DIR / "feature_columns.pkl"

    for p in (model_path, splits_path, feat_path):
        if not p.exists():
            log.error("%s not found. Run previous steps first.", p)
            sys.exit(1)

    try:
        model        = joblib.load(model_path)
        splits       = joblib.load(splits_path)
        feature_cols = joblib.load(feat_path)

        X_test = splits["X_test"]
        log.info("Loaded model + test set  (%d samples, %d features)",
                 len(X_test), len(feature_cols))

        # ──────────────────────────────────────────────────────────────
        # SHAP TreeExplainer  (optimised for tree‑based models)
        # ──────────────────────────────────────────────────────────────
        log.info("Computing SHAP values (TreeExplainer) …")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Human‑readable feature names for plots
        pretty_names = [c.replace("_", " ").title() for c in feature_cols]

        # ──────────────────────────────────────────────────────────────
        # 1) SHAP Summary Plot
        # ──────────────────────────────────────────────────────────────
        # WHAT IT MEANS:
        #   Each dot is one apartment from the test set.
        #   Position on the x‑axis shows how much that feature pushed the
        #   predicted price UP (right) or DOWN (left) compared to the
        #   average prediction.  Colour shows the feature value (red = high,
        #   blue = low).  Features at the top have the biggest overall
        #   impact on the model's predictions.
        log.info("Generating SHAP summary plot …")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=pretty_names,
            show=False,
        )
        plt.title("SHAP Summary – Feature Impact on Apartment Price", fontsize=14)
        plt.xlabel("Impact on Predicted Price (SHAP Value in LKR)", fontsize=11)
        plt.tight_layout()
        path_summary = PLOTS_DIR / "shap_summary.png"
        plt.savefig(path_summary, dpi=150, bbox_inches="tight")
        plt.close("all")
        log.info("Saved → %s", path_summary)

        # ──────────────────────────────────────────────────────────────
        # 2) SHAP Dependence Plot for the most important feature
        # ──────────────────────────────────────────────────────────────
        # WHAT IT MEANS:
        #   Shows the relationship between one feature's actual value
        #   (x‑axis) and its SHAP value (y‑axis).  This reveals whether
        #   the feature has a linear, non‑linear, or threshold effect on
        #   the price.  The colour shows the interaction feature that
        #   SHAP automatically selects.
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = int(np.argmax(mean_abs_shap))
        top_feature = feature_cols[top_idx]
        log.info("Most important feature: '%s'  (mean |SHAP| = %.2f)",
                 top_feature, mean_abs_shap[top_idx])

        fig, ax = plt.subplots(figsize=(9, 6))
        shap.dependence_plot(
            top_idx,
            shap_values,
            X_test,
            feature_names=pretty_names,
            show=False,
            ax=ax,
        )
        ax.set_title(
            f"SHAP Dependence – How '{pretty_names[top_idx]}' Affects Price",
            fontsize=13,
        )
        ax.set_ylabel("Impact on Predicted Price (SHAP Value)", fontsize=11)
        fig.tight_layout()
        path_dep = PLOTS_DIR / "shap_dependence.png"
        fig.savefig(path_dep, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved → %s", path_dep)

        # ──────────────────────────────────────────────────────────────
        # 3) Feature Importance Bar Chart
        # ──────────────────────────────────────────────────────────────
        # WHAT IT MEANS:
        #   A simple ranking of features by their average absolute SHAP
        #   value.  The taller the bar, the more that feature matters to
        #   the model's predictions on average across all test apartments.
        log.info("Generating feature importance bar chart …")
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_idx = np.argsort(mean_abs_shap)
        ax.barh(
            [pretty_names[i] for i in sorted_idx],
            mean_abs_shap[sorted_idx],
            color="teal",
        )
        ax.set_xlabel("Mean Absolute SHAP Value (Average Impact on Price)",
                       fontsize=11)
        ax.set_title("Feature Importance – Which Features Matter Most?",
                      fontsize=14)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        path_bar = PLOTS_DIR / "feature_importance_bar.png"
        fig.savefig(path_bar, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved → %s", path_bar)

        log.info("✅  Explainability analysis complete.")

    except Exception:
        log.exception("Explain step failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
