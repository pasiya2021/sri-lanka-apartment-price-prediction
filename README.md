# Sri Lanka Apartment Price Prediction

> **Regression task** — Predict apartment prices in Sri Lanka using data scraped from [properties.lk](https://properties.lk/allads?category=apartmentforsale).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sri-lanka-apartment-price-prediction-9gxrf3yjipawizhc2uz6n.streamlit.app)

## 🌐 Live Demo

🚀 **Try the app now →** [sri-lanka-apartment-price-prediction.streamlit.app](https://sri-lanka-apartment-price-prediction-9gxrf3yjipawizhc2uz6n.streamlit.app)

The app lets you **predict apartment prices in Sri Lanka** by entering details like district, location, size, bedrooms, and amenities. It uses a trained **XGBoost** regression model and provides **SHAP-based explanations** showing exactly which features drive the predicted price up or down.

---

## 📁 Project Structure

```
├── scrape.py                  # requests + BeautifulSoup scraper
├── scraper.py                 # Selenium‑based scraper (for SPA sites)
├── processed.csv              # Cleaned dataset (generated)
├── apartment_data.csv         # Raw scraped data
├── requirements.txt
├── README.md
├── report_outline.md
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Shared helpers (logging, price cleaning)
│   ├── preprocess.py          # Data cleaning & feature engineering
│   ├── train.py               # XGBoost training + hyperparameter tuning
│   ├── evaluate.py            # RMSE, MAE, R² + plots
│   └── explain.py             # SHAP explainability analysis
│
├── models/                    # Saved model, encoders, scaler, splits
│
├── outputs/
│   ├── metrics.json
│   ├── metrics_table.csv
│   └── plots/                 # All generated charts
│
└── app/
    └── streamlit_app.py       # Interactive web front‑end
```

---

## ⚙️ Installation

```bash
# 1. Clone or unzip the project
cd "House Price Prediction ML Model"

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Run Commands (Step by Step)

### Step 1 — Scrape data

```bash
# Option A: requests + BeautifulSoup (assignment requirement)
python scrape.py --max-pages 50 --output apartment_data.csv

# Option B: Selenium scraper (recommended — site is a JS SPA)
python scraper.py --target 5000
```

### Step 2 — Preprocess

```bash
python src/preprocess.py --input apartment_data.csv --output processed.csv
```

### Step 3 — Train

```bash
python src/train.py --data processed.csv
```

### Step 4 — Evaluate

```bash
python src/evaluate.py
```

### Step 5 — Explain

```bash
python src/explain.py
```

### Step 6 — Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Model Details

| Item                    | Value                            |
|-------------------------|----------------------------------|
| Algorithm               | XGBoost Regressor                |
| Tuning                  | RandomizedSearchCV (40 iters)    |
| Early stopping          | 50 rounds on validation set      |
| Split                   | 70 / 15 / 15 (train/val/test)    |
| Random state            | 42                               |

### Evaluation Metrics

| Metric | Description                             |
|--------|-----------------------------------------|
| RMSE   | Root Mean Squared Error (LKR)           |
| MAE    | Mean Absolute Error (LKR)               |
| R²     | Coefficient of Determination (0–1)      |

---

## 🧠 Explainability

- **SHAP Summary Plot** — Shows which features most influence predictions globally.
- **SHAP Dependence Plot** — Reveals how the top feature's value relates to its impact on price.
- **Feature Importance Bar Chart** — Simple ranking of features by average SHAP impact.

All plots are saved to `outputs/plots/`.

---

## 🌐 Streamlit App Features

- Sidebar with dropdown / slider inputs for apartment details
- Predicted price in LKR
- Model performance metrics (RMSE, MAE, R²)
- Global SHAP explanation (summary plot)
- Local SHAP explanation (waterfall plot for each individual prediction)

---

## 📝 Notes

- The scraper respects `robots.txt` and includes configurable delays.
- Price values are cleaned from formats like `Rs 12,500,000`, `2.5 Mn`, `45 Lakh`.
- Feature engineering extracts apartment brand/project name from listing titles.

---

## 📜 License

Academic use only. Data sourced from [properties.lk](https://properties.lk).
