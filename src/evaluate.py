"""
src/evaluate.py
===============
Evaluate the trained XGBoost model on the **test set**.

Metrics computed:
    • RMSE   (Root Mean Squared Error)
    • MAE    (Mean Absolute Error)
    • R² Score

Outputs
-------
    outputs/metrics.json
    outputs/metrics_table.csv
    outputs/plots/predicted_vs_actual.png
    outputs/plots/residual_histogram.png

Usage
-----
    python src/evaluate.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # non‑interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, ensure_dirs, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR

log = get_logger("evaluate")


# ──────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    """Return RMSE, MAE, R² as a dict."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"RMSE": round(rmse, 2),
            "MAE":  round(mae, 2),
            "R2":   round(r2, 4)}


# ──────────────────────────────────────────────────────────────────────
# Plot helpers – HUMAN‑READABLE titles & labels throughout
# ──────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(y_true, y_pred, save_path: Path):
    """Scatter plot: Predicted Apartment Price vs Actual Apartment Price."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, edgecolor="k", linewidth=0.3)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect Prediction")

    ax.set_xlabel("Actual Apartment Price (LKR)", fontsize=12)
    ax.set_ylabel("Predicted Apartment Price (LKR)", fontsize=12)
    ax.set_title("Predicted Apartment Price vs Actual Apartment Price", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", save_path)


def plot_residual_histogram(y_true, y_pred, save_path: Path):
    """Histogram of prediction residuals."""
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, kde=True, bins=50, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero Error")

    ax.set_xlabel("Prediction Error (Actual − Predicted) in LKR", fontsize=12)
    ax.set_ylabel("Number of Apartments", fontsize=12)
    ax.set_title("Distribution of Prediction Errors (Residuals)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    # ── Load model & data splits ──
    model_path  = MODELS_DIR / "xgb_model.pkl"
    splits_path = MODELS_DIR / "data_splits.pkl"

    for p in (model_path, splits_path):
        if not p.exists():
            log.error("%s not found. Run previous steps first.", p)
            sys.exit(1)

    try:
        model  = joblib.load(model_path)
        splits = joblib.load(splits_path)
        X_test = splits["X_test"]
        y_test = splits["y_test"]
        log.info("Loaded model and test set  (n=%d)", len(X_test))

        # ── Predict ──
        y_pred = model.predict(X_test)

        # ── Metrics ──
        metrics = compute_metrics(y_test.values, y_pred)
        log.info("RMSE  = {:,.2f}".format(metrics["RMSE"]))
        log.info("MAE   = {:,.2f}".format(metrics["MAE"]))
        log.info("R²    = {:.4f}".format(metrics["R2"]))

        # Save metrics.json
        json_path = OUTPUTS_DIR / "metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Saved → %s", json_path)

        # Save metrics_table.csv
        csv_path = OUTPUTS_DIR / "metrics_table.csv"
        pd.DataFrame([metrics]).to_csv(csv_path, index=False)
        log.info("Saved → %s", csv_path)

        # ── Plots ──
        plot_predicted_vs_actual(
            y_test.values, y_pred,
            PLOTS_DIR / "predicted_vs_actual.png",
        )
        plot_residual_histogram(
            y_test.values, y_pred,
            PLOTS_DIR / "residual_histogram.png",
        )

        log.info("✅  Evaluation complete.")

    except Exception:
        log.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
