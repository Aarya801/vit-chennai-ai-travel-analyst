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
`src/modeling.py` and `python -m scripts.train_model` fit a `CatBoostRegressor` on `log1p(Price)` for all rows with a known price and save a model artifact. That command does not create a held-out split or compute evaluation metrics. The separate `python -m scripts.evaluate_model` command fits the same fixed configuration on a chronological training partition and reports validation and test scores without replacing the saved artifact.

## 6. Evaluation status
The reproducible evaluator partitions eligible rows by sorted, unique departure date: the first 60% of dates for training, the next 20% for validation, and the final 20% for testing. Rows sharing a departure date remain together. It fits only on training rows, reports validation and test MAE, RMSE, and R², and compares them with a baseline that always predicts the training-set median price. The validation partition is diagnostic only; the code does not tune or select a model with it. Run instructions and output metadata are in [`docs/EVALUATION.md`](EVALUATION.md).

`artifacts/metrics.json` contains legacy values whose evaluation procedure is not reproducible from the checked-in source. They are not reported here as held-out results. No new scores are claimed by this report; the evaluator must be run against a specified data file and environment to produce them.

## 7. Part 3
Booking lead time is grouped into eight windows. The system chooses the lowest historical median among groups with at least 20 observations. Overall, the lowest eligible median is **91–180 days at ₹38,659**.

## 8. Technologies
Python, Pandas, NumPy, Scikit-learn, CatBoost, Plotly, Streamlit, Pytest, GitHub Actions.

## 9. Limitations
Historical prices do not guarantee future fares. The project is a historical analytics and model-training experiment, not a live fare engine. The dataset contains departure dates but no verified timestamp showing when each fare quote was observed or captured. The implemented split orders travel dates and cannot establish the information available at each historical booking time, rule out every form of temporal leakage, or serve as a prospective quote-time backtest. Availability of `Days_Before_Departure` and other recorded features at quote time is also unverified. Rows missing price or a valid departure date are excluded from evaluation and counted in its output.

## 10. Future Work
Run and review the implemented evaluation, record exact dependency versions when reporting results, and obtain trustworthy fare-observation timestamps before attempting prospective booking-time validation. Then consider airport alias normalization, time-of-day features, model comparisons, and prediction uncertainty.
