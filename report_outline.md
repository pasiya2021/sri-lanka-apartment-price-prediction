# Report Outline — Sri Lanka Apartment Price Prediction

---

## 1. Introduction
- Problem statement: Predict apartment sale prices in Sri Lanka.
- Motivation: Help buyers, sellers, and agents estimate fair market value.
- Dataset source: properties.lk (web‑scraped apartment listings).

## 2. Data Collection
- Scraping methodology (requests + BeautifulSoup; Selenium for SPA).
- Ethical considerations: robots.txt compliance, rate limiting, academic use.
- Raw dataset size and column descriptions.

## 3. Data Preprocessing
- Currency normalisation (Rs, commas, Lakh, Mn → numeric LKR).
- Missing‑value treatment (median imputation for numerics; "Unknown" for categoricals).
- Outlier removal (prices below 100 000 or above 5 billion LKR).
- Feature engineering:
  - **Apartment Model** extracted from ad title (brand / project name).
  - Floor number parsed from title.
  - Binary flags: furnished, parking, gym/pool.
- Encoding: LabelEncoder for categoricals; StandardScaler for numerics.
- Data split: 70 % train / 15 % validation / 15 % test (random_state = 42).

## 4. Model Training
- Algorithm chosen: **XGBoost Regressor**.
- Justification: handles mixed feature types, robust to outliers, fast histogram training.
- Hyperparameter tuning: RandomizedSearchCV (40 iterations, 3‑fold CV).
  - Search space: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, etc.
- Early stopping: re‑train best params with 2 000‑round ceiling, stop after 50 rounds of no improvement on validation set.
- Final chosen hyperparameters (logged and saved to `models/best_params.json`).

## 5. Evaluation
- Metrics on the held‑out **test set**:
  - RMSE (Root Mean Squared Error)
  - MAE  (Mean Absolute Error)
  - R² Score
- Visualisations:
  - Predicted vs Actual scatter plot.
  - Residual histogram.
- Discussion of performance and potential improvements.

## 6. Model Explainability (SHAP)
- SHAP TreeExplainer applied to XGBoost.
- **Summary plot**: global overview of feature impacts.
- **Dependence plot**: interaction effect of the most important feature.
- **Feature importance bar chart**: average absolute SHAP values.
- Plain‑language interpretation of each plot.

## 7. Deployment (Streamlit App)
- Interactive web interface for price prediction.
- Input form auto‑generated from dataset columns.
- Displays: predicted price, model metrics, global SHAP summary, local SHAP waterfall.

## 8. Conclusion
- Summary of findings.
- Key predictors of apartment price in Sri Lanka.
- Limitations: data freshness, geographic coverage, SPA scraping challenges.
- Future work: incorporate images, map data, time‑series trends.

## 9. References
- properties.lk
- XGBoost documentation
- SHAP library (Lundberg & Lee, 2017)
- Scikit‑learn documentation
