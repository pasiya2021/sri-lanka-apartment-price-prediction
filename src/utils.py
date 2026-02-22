"""
src/utils.py
============
Shared helper functions used across the project.
Provides:
    - Logging configuration
    - Path helpers (ensures output dirs exist)
    - Price‑cleaning utilities (Rs, commas, Lakh, Mn → float)
"""

import os
import re
import sys
import logging
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Project root is the repo folder (one level above src/)
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
PLOTS_DIR    = OUTPUTS_DIR  / "plots"


def ensure_dirs():
    """Create models/ and outputs/plots/ if they don't exist."""
    for d in (MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Return a logger that prints to stdout with a simple format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ──────────────────────────────────────────────────────────────────────
# Price cleaning
# ──────────────────────────────────────────────────────────────────────
_MULTIPLIERS = {
    "mn":   1_000_000,
    "m":    1_000_000,
    "mill": 1_000_000,
    "million":  1_000_000,
    "lakh": 100_000,
    "lac":  100_000,
    "l":    100_000,
    "k":    1_000,
    "cr":   10_000_000,
    "crore":10_000_000,
}

# Regex: optional currency prefix → number → optional multiplier suffix
_PRICE_RE = re.compile(
    r"(?:rs\.?|lkr|රු\.?|₨)?\s*"          # optional currency marker
    r"([\d,]+(?:\.\d+)?)"                   # the number (with commas / decimals)
    r"\s*"
    r"(mn|m|mill(?:ion)?|lakh|lac|l|k|cr(?:ore)?)?"  # optional multiplier
    , re.IGNORECASE
)


def clean_price(raw) -> float | None:
    """
    Convert a raw price string to a plain float in LKR.

    Handles formats such as:
        "Rs 12,500,000"   → 12500000.0
        "LKR 2.5 Mn"      → 2500000.0
        "45 Lakh"          → 4500000.0
        "15,000"           → 15000.0
        NaN / None / ""    → None
    """
    if raw is None:
        return None

    text = str(raw).strip()
    if text == "" or text.lower() in ("nan", "none", "-", "n/a", "negotiable"):
        return None

    m = _PRICE_RE.search(text)
    if not m:
        return None

    number_str = m.group(1).replace(",", "")
    try:
        value = float(number_str)
    except ValueError:
        return None

    suffix = (m.group(2) or "").lower()
    if suffix in _MULTIPLIERS:
        value *= _MULTIPLIERS[suffix]

    return value if value > 0 else None


def detect_target_column(columns: list[str],
                         candidates=("price_lkr", "price", "total_price",
                                     "asking_price", "amount")) -> str:
    """
    Automatically detect which column holds the target variable.
    Prints a clear message about the detected column.
    """
    col_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in col_lower:
            actual = col_lower[cand]
            print(f"[INFO] Target column detected → '{actual}'")
            return actual

    # Fallback: any column containing "price"
    for c in columns:
        if "price" in c.lower():
            print(f"[INFO] Target column detected (fuzzy) → '{c}'")
            return c

    raise ValueError(
        "Could not detect a price / target column. "
        f"Available columns: {columns}"
    )
