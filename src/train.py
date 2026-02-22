"""
src/train.py
============
Train an **XGBoost Regressor** with RandomizedSearchCV hyperparameter tuning
and early stopping on the validation set.

Usage
-----
    python src/train.py --data processed.csv

Outputs
-------
    models/xgb_model.pkl        – best XGBoost model
    models/best_params.json     – chosen hyper‑parameters
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, ensure_dirs, MODELS_DIR

log = get_logger("train")


# ──────────────────────────────────────────────────────────────────────
# Hyper‑parameter search space
# ──────────────────────────────────────────────────────────────────────
# We use RandomizedSearchCV (not GridSearchCV) because:
#   1. The search space is large – exhaustive grid would be very slow.
#   2. Random search is statistically more efficient at finding good
#      regions of the hyper‑parameter space for a fixed budget of
#      iterations (Bergstra & Bengio, 2012).
#   3. Combined with early stopping on the validation set, this gives
#      a strong model without excessive compute time.
PARAM_DISTRIBUTIONS = {
    "n_estimators":      randint(200, 1500),
    "max_depth":         randint(3, 12),
    "learning_rate":     uniform(0.01, 0.29),   # 0.01 – 0.30
    "subsample":         uniform(0.5, 0.5),      # 0.50 – 1.00
    "colsample_bytree":  uniform(0.5, 0.5),
    "min_child_weight":  randint(1, 10),
    "gamma":             uniform(0, 0.5),
    "reg_alpha":         uniform(0, 1.0),
    "reg_lambda":        uniform(0.5, 1.5),
}


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val, n_iter: int = 40):
    """
    Run RandomizedSearchCV, then refit the best model with early stopping.
    Returns the fitted model and the best parameters dict.
    """
    log.info("Starting RandomizedSearchCV  (n_iter=%d) …", n_iter)

    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",          # fast histogram‑based training
        random_state=42,
        verbosity=0,
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    log.info("Best CV RMSE: %.2f", -search.best_score_)
    log.info("Best hyper‑parameters:")
    for k, v in best_params.items():
        log.info("    %-22s = %s", k, v)

    # ── Refit with early stopping on the validation set ──
    log.info("Refitting best model with early stopping (validation set) …")
    refit_params = {k: v for k, v in best_params.items() if k != "n_estimators"}
    model = XGBRegressor(
        **refit_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        verbosity=0,
        n_estimators=2000,        # high ceiling; early stopping will cut it
        early_stopping_rounds=50,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    log.info("Best iteration (early stop): %d", model.best_iteration)

    return model, best_params


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost regressor")
    parser.add_argument("--data", default="processed.csv",
                        help="Processed CSV (unused if splits exist)")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="RandomizedSearchCV iterations")
    args = parser.parse_args()

    ensure_dirs()
    splits_path = MODELS_DIR / "data_splits.pkl"

    if not splits_path.exists():
        log.error("data_splits.pkl not found. Run preprocess.py first.")
        sys.exit(1)

    try:
        splits = joblib.load(splits_path)
        X_train = splits["X_train"]
        X_val   = splits["X_val"]
        y_train = splits["y_train"]
        y_val   = splits["y_val"]

        log.info("Loaded splits: train=%d  val=%d", len(X_train), len(X_val))

        model, best_params = train(X_train, y_train, X_val, y_val,
                                   n_iter=args.n_iter)

        # ── Persist ──
        model_path = MODELS_DIR / "xgb_model.pkl"
        joblib.dump(model, model_path)
        log.info("Saved model → %s", model_path)

        params_path = MODELS_DIR / "best_params.json"
        # Convert numpy types to Python builtins for JSON
        clean_params = {k: (int(v) if isinstance(v, (np.integer,))
                            else float(v) if isinstance(v, (np.floating,))
                            else v)
                        for k, v in best_params.items()}
        with open(params_path, "w") as f:
            json.dump(clean_params, f, indent=2)
        log.info("Saved best params → %s", params_path)

        log.info("✅  Training complete.")

    except Exception:
        log.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
