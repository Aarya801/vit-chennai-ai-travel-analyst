# AI Travel Analyst

A Python and Streamlit project for exploring a supplied flight-pricing dataset, preparing model features, training a CatBoost regressor, and comparing historical booking lead-time groups.

> **Evaluation status:** the repository stores model metrics, but its checked-in training code does not implement or reproduce the reported held-out evaluation. The app therefore does not present those stored numbers as validated test performance. This project is exploratory and does not forecast guaranteed future fares.

## What is implemented

- Dataset cleaning for numeric text, durations, stop counts, passenger counts, and missing categorical values.
- Exploratory views of prices by airline, class, route, distance, stops, and booking lead time.
- CatBoost training on rows with a known price, with the target transformed using `log1p`.
- Historical booking-window summaries and a recommendation based on eligible groups with at least 20 observations.
- A small pytest suite for basic data cleaning and booking-window behavior.

## Data and evaluation limits

The included dataset has 100,000 rows and 18 columns; 5,053 rows have no price and 1,961 rows are duplicates. The training script filters to rows with a price and fits on the remaining rows. It does not split out a test set or compute evaluation metrics. `artifacts/metrics.json` contains saved values, but the current repository does not include enough code or provenance to establish how they were measured. Treat them as unverified artifact values, not evidence of generalization.

The features include departure date components and `Days_Before_Departure`, but there is no verified timestamp for when each fare was observed or captured. Without that timestamp, a chronological simulation of information available at booking time is not possible. The booking-window analysis is retrospective; its historical medians are not prospective fare advice.

No lockfile is included. `AI_Travel_Analyst_VIT_Chennai_Submission.zip` is a packaged submission archive, not a clean reproducible release: it contains a nested copy of the project and generated Python cache files. The repository source and `requirements.txt` are the reviewable inputs.

## Run locally

Python 3.11 is used by the checked-in GitHub Actions workflow.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Tests and model training

Run the current tests:

```bash
pytest -q
```

The tests check basic cleaning and booking-window output. They do **not** evaluate CatBoost model quality.

Retrain the model artifact:

```bash
python scripts/train_model.py
```

The current script fits the model on all priced rows and saves `models/price_model.cbm`; it does not calculate a held-out score. Do not use the stored metrics as validation until a reproducible evaluation split and scoring procedure are added.

## Feature preparation

The pipeline normalizes duration strings to minutes, normalizes stop and passenger-count values, converts selected numeric text, parses the departure date into month and day-of-year, and represents missing categorical values as `Unknown`. The model uses categorical columns directly through CatBoost.

## Project layout

- `app.py` — Streamlit dashboard
- `src/data_utils.py` — cleaning and feature preparation
- `src/modeling.py` — model fit and artifact save
- `src/booking_analysis.py` — historical booking-window summaries
- `tests/` — lightweight data-pipeline checks
- `artifacts/` — stored metrics and analysis outputs; metric provenance is incomplete
- `docs/` — report, demo, and interview notes

## Next engineering steps

1. Implement a documented, reproducible evaluation split and compute the baseline and model metrics from that split only.
2. Record split details and data provenance alongside evaluation outputs.
3. Obtain a trustworthy fare observation or booking timestamp before attempting prospective chronological validation.
4. Add tests for feature preparation and the evaluation path.

Until those steps are complete, describe the model as a training pipeline experiment rather than a validated fare predictor.
