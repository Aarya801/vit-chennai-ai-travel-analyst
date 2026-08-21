✈️ AI Travel Analyst
End-to-End Flight Price Prediction & Analytics

A production-oriented machine learning pipeline that transforms
flight pricing data into predictive insights through data
preprocessing, feature engineering, model training, evaluation,
and interactive deployment.

**Track 3 — Data Science & Visualisation | AI Travel Analyst**

## What this project does
An end-to-end flight-price intelligence system using the **official `flight_pricing_dataset.csv` supplied through the club's Google Drive**.

### Completed
- Part 1: cleaning, preprocessing, 5+ visualizations, factor analysis, recommendations
- Part 2: feature engineering, CatBoost regression, MAE/RMSE/R², feature importance
- Part 3: **Cheapest Booking Time Analysis**
- Streamlit interactive dashboard
- Tests + GitHub Actions
- README, report, demo script, interview cheat sheet

## Dataset audit
- Rows: **100,000**
- Columns: **18**
- Missing Price: **5,053**
- Duplicate rows: **1,961**
- Rows used for model: **94,947**

## Model results
| Metric | Result |
|---|---:|
| Baseline median-price MAE | ₹55,774 |
| CatBoost MAE | ₹13,432 |
| CatBoost RMSE | ₹48,115 |
| CatBoost R² | 0.619 |

The model predicts `log1p(Price)` and converts predictions back to rupees for evaluation.

## Why CatBoost?
The dataset contains multiple categorical variables, missing values and nonlinear relationships. CatBoost can handle categorical features directly and capture interactions without creating a huge one-hot encoded matrix.

## Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Retrain:
```bash
python scripts/train_model.py
```

## Engineering decisions
- Normalize mixed duration formats (`1.67`, `0h 45m`, `177 min`) to minutes.
- Normalize stop representations (`non-stop`, `1 stop`, `1`, etc.).
- Convert written passenger counts to numbers.
- Parse departure date into month/day-of-year.
- Keep missing categories as `Unknown`.
- Train only where Price exists.
- Use a held-out 20% test split.
- Use booking-window sample threshold of 20 for recommendations.

## Key challenge
The biggest challenge was **data quality**, not model complexity: many fields mix numeric and textual representations and have missing values.

## Future improvements
Airport alias normalization, richer time-of-day features, time-aware validation, hyperparameter tuning, uncertainty intervals and route-specific forecasting.

## Files
`app.py` • `src/` • `tests/` • `models/` • `artifacts/` • `docs/` • `data/`
