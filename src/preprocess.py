"""
src/preprocess.py
=================
Load the raw scraped CSV, clean it, engineer features, encode categoricals,
normalise numerics, split into train / validation / test, and persist everything.

Usage
-----
    python src/preprocess.py --input apartment_data.csv --output processed.csv

Outputs
-------
    processed.csv                    – full cleaned dataset
    models/label_encoders.pkl        – fitted LabelEncoders (one per cat column)
    models/scaler.pkl                – fitted StandardScaler for numeric features
    models/feature_columns.pkl       – ordered list of feature column names
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Allow running from repo root: python src/preprocess.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    get_logger,
    ensure_dirs,
    clean_price,
    detect_target_column,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)

log = get_logger("preprocess")

# ──────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ──────────────────────────────────────────────────────────────────────

# Common Sri‑Lankan apartment project / brand keywords
_BRAND_KEYWORDS = [
    "Altair", "Cinnamon Life", "Clearpoint", "Havelock City",
    "Capitol TwinPeaks", "TwinPeaks", "OnThree20", "Platinum One",
    "Emperor", "The One", "Monarch", "96 Residencies",
    "Astoria", "Shangri-La", "ITC", "Fairway", "Colombo City Centre",
    "CCC", "7th Sense", "Sapphire", "Iconic", "Blue Ocean",
    "Prime", "Trillium", "Ariyana", "Kings Garden",
    "Rajagiriya", "Crescat", "Capitol", "Kotte",
]

_BRAND_RE = re.compile(
    "|".join(re.escape(b) for b in _BRAND_KEYWORDS),
    re.IGNORECASE,
)


def extract_apartment_model(title: str) -> str:
    """
    Extract a recognisable apartment brand / project name from the ad title.
    Falls back to 'Other' when nothing matches.
    """
    if not isinstance(title, str):
        return "Other"
    m = _BRAND_RE.search(title)
    return m.group(0).strip().title() if m else "Other"


def extract_floor_from_title(title: str) -> float:
    """Try to pull a floor / storey number from the title."""
    if not isinstance(title, str):
        return np.nan
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)\s*floor", title, re.I)
    return float(m.group(1)) if m else np.nan


def extract_furnished(desc: str) -> int:
    """Binary flag: 1 if description mentions furnished, else 0."""
    if not isinstance(desc, str):
        return 0
    return 1 if re.search(r"\b(fully\s+)?furnished\b", desc, re.I) else 0


def extract_parking(desc: str) -> int:
    """Binary flag: 1 if parking is mentioned."""
    if not isinstance(desc, str):
        return 0
    return 1 if re.search(r"\bparking\b", desc, re.I) else 0


def extract_gym_pool(desc: str) -> int:
    """Binary flag: 1 if gym or pool is mentioned."""
    if not isinstance(desc, str):
        return 0
    return 1 if re.search(r"\b(gym|pool|swimming)\b", desc, re.I) else 0


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load raw CSV and perform basic cleaning."""
    log.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    log.info("Raw shape: %s", df.shape)

    # ── Detect and clean target column ──
    target_col = detect_target_column(df.columns.tolist())
    df["price"] = df[target_col].apply(clean_price)

    before = len(df)
    df.dropna(subset=["price"], inplace=True)
    log.info("Dropped %d rows with no valid price  (%d → %d)",
             before - len(df), before, len(df))

    # Remove extreme outliers (price < 100 000 or > 5 billion LKR)
    df = df[(df["price"] >= 100_000) & (df["price"] <= 5_000_000_000)]
    log.info("After outlier filter: %d rows", len(df))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and normalise raw ones."""
    df = df.copy()

    # ── Numeric conversions ──
    for col in ("bedrooms", "bathrooms", "property_size_sqft", "land_size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Fill numeric NaNs with median ──
    for col in ("bedrooms", "bathrooms", "property_size_sqft"):
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            log.info("Filled missing '%s' with median = %.1f", col, median_val)

    # ── Feature: apartment brand / project name (CRITICAL) ──
    if "title" in df.columns:
        df["apartment_model"] = df["title"].apply(extract_apartment_model)
        log.info("Extracted 'apartment_model' – %d unique values",
                 df["apartment_model"].nunique())

    # ── Feature: floor number ──
    if "title" in df.columns:
        df["floor"] = df["title"].apply(extract_floor_from_title)
        df["floor"].fillna(df["floor"].median() if df["floor"].notna().any() else 0,
                           inplace=True)

    # ── Binary features from description ──
    desc_col = "description" if "description" in df.columns else None
    if desc_col:
        df["is_furnished"] = df[desc_col].apply(extract_furnished)
        df["has_parking"]  = df[desc_col].apply(extract_parking)
        df["has_gym_pool"] = df[desc_col].apply(extract_gym_pool)

    # ── Normalise string columns ──
    for col in ("location", "district", "listing_type", "property_type"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col].replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"},
                            inplace=True)

    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Label‑encode categoricals, standard‑scale numerics.
    Returns (X DataFrame, y Series, encoders dict, scaler).
    """
    # ── Choose feature columns ──
    cat_cols = [c for c in ("district", "location", "listing_type",
                            "property_type", "apartment_model")
                if c in df.columns]

    num_cols = [c for c in ("bedrooms", "bathrooms", "property_size_sqft",
                            "floor", "is_furnished", "has_parking",
                            "has_gym_pool")
                if c in df.columns]

    feature_cols = cat_cols + num_cols
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── Label encoding ──
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        log.info("Encoded '%s' → %d classes", col, len(le.classes_))

    # ── Scaling ──
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df[feature_cols].copy()
    y = df["price"].copy()

    return X, y, encoders, scaler, feature_cols


def split_data(X, y):
    """
    70 / 15 / 15 split with random_state = 42.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    log.info("Split → train=%d  val=%d  test=%d",
             len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess apartment data")
    parser.add_argument("--input",  default="apartment_data.csv",
                        help="Path to raw scraped CSV")
    parser.add_argument("--output", default="processed.csv",
                        help="Where to save the cleaned CSV")
    args = parser.parse_args()

    ensure_dirs()

    try:
        # 1. Load & clean
        df = load_and_clean(args.input)

        # 2. Feature engineering
        df = engineer_features(df)

        # 3. Encode & scale
        X, y, encoders, scaler, feature_cols = encode_and_scale(df)

        # 4. Split
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # 5. Save processed dataset (features + target)
        processed = X.copy()
        processed["price"] = y.values
        processed.to_csv(args.output, index=False)
        log.info("Saved processed dataset → %s  (%d rows)", args.output, len(processed))

        # 6. Save splits
        splits = {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        }
        joblib.dump(splits, MODELS_DIR / "data_splits.pkl")
        log.info("Saved train/val/test splits → models/data_splits.pkl")

        # 7. Save encoders, scaler, feature list
        joblib.dump(encoders,     MODELS_DIR / "label_encoders.pkl")
        joblib.dump(scaler,       MODELS_DIR / "scaler.pkl")
        joblib.dump(feature_cols, MODELS_DIR / "feature_columns.pkl")
        log.info("Saved encoders, scaler, feature_columns → models/")

        log.info("✅  Preprocessing complete.")

    except Exception:
        log.exception("Preprocessing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
