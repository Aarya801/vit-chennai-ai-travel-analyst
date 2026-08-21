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
CatBoostRegressor is trained on `log1p(Price)` with an 80/20 held-out split and random seed 42.

## 6. Results
MAE: ₹13,432  
RMSE: ₹48,115  
R²: 0.619

## 7. Part 3
Booking lead time is grouped into eight windows. The system chooses the lowest historical median among groups with at least 20 observations. Overall, the lowest eligible median is **91–180 days at ₹38,659**.

## 8. Technologies
Python, Pandas, NumPy, Scikit-learn, CatBoost, Plotly, Streamlit, Pytest, GitHub Actions.

## 9. Limitations
Historical prices do not guarantee future fares. The current system is a historical analytics/recommendation tool rather than a live fare engine.

## 10. Future Work
Normalize airport aliases, add time-of-day features, use time-aware validation, compare gradient-boosting models and add prediction uncertainty.
