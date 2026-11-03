# Project Report

## 1. Overview
AI Travel Analyst turns the official VIT Chennai AIML flight-pricing dataset into an interactive analytics and decision-support system.

## 2. Problem
Flight prices vary with airline, route, class, stops, distance, duration, season and booking lead time. The system explores these factors and identifies historically cheaper booking windows.

## 3. Dataset
The supplied file contains 100,000 rows and 18 columns. It has 5,053 missing prices and 1,961 duplicate rows.

## 4. Preprocessing
Mixed numeric values are parsed; durations are converted to minutes; stop counts and passenger counts are normalized; dates are decomposed; categorical missing values become `Unknown`.

## 5. Modeling
`src/modeling.py` trains a `CatBoostRegressor` on `log1p(Price)` for all rows with a known price, using a fixed random seed. It saves the fitted model. It does not create a held-out split or compute evaluation metrics.

## 6. Evaluation status
`artifacts/metrics.json` stores MAE, RMSE, and R² values, but the checked-in training and test code does not reproduce their evaluation procedure. They are therefore not reported here as held-out results or evidence of expected prediction quality. The current tests check data cleaning and booking-window logic, not model performance.

## 7. Part 3
Booking lead time is grouped into eight windows. The system chooses the lowest historical median among groups with at least 20 observations. Overall, the lowest eligible median is **91–180 days at ₹38,659**.

## 8. Technologies
Python, Pandas, NumPy, Scikit-learn, CatBoost, Plotly, Streamlit, Pytest, GitHub Actions.

## 9. Limitations
Historical prices do not guarantee future fares. The project is a historical analytics and model-training experiment, not a live fare engine. The dataset has no verified timestamp showing when each fare was observed; this prevents faithful prospective chronological validation of booking-time information.

## 10. Future Work
First add a reproducible held-out evaluation, with the split and baseline documented and computed without test leakage. For prospective validation, obtain trustworthy fare-observation timestamps. Then consider airport alias normalization, time-of-day features, model comparisons, and prediction uncertainty.
