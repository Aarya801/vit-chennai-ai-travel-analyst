# AI Travel Analyst

A Python and Streamlit project for exploring a supplied flight-pricing dataset, preparing model features, training a CatBoost regressor, and comparing historical booking lead-time groups.

> **Evaluation status:** the repository now provides a reproducible departure-date-ordered train/validation/test evaluation. It is a calendar-time proxy, not a prospective fare-quote backtest: the dataset has no verified quote-observation timestamp. No model metrics are claimed here; run the documented evaluation command to produce scores for the supplied data. This project is exploratory and does not forecast guaranteed future fares.

## What is implemented

- Dataset cleaning for numeric text, durations, stop counts, passenger counts, and missing categorical values.
- Exploratory views of prices by airline, class, route, distance, stops, and booking lead time.
- CatBoost training on rows with a known price, with the target transformed using `log1p`.
- Historical booking-window summaries and a recommendation based on eligible groups with at least 20 observations.
- A small pytest suite for basic data cleaning and booking-window behavior.

## Data and evaluation limits

The included dataset has 100,000 rows and 18 columns; 5,053 rows have no price and 1,961 rows are duplicates. The artifact-training script filters to rows with a price and fits on the remaining rows. The separate evaluator uses only dated, priced rows for its chronological split. `artifacts/metrics.json` contains legacy values, but the repository does not include enough provenance to establish how they were measured. Treat them as unverified artifact values, not evidence of generalization.

The evaluation splits rows by unique `Departure_Date`, keeping all rows for one departure date in the same partition. This orders travel dates, not fare observation times. There is no verified timestamp for when a fare was observed or captured, so the evaluation cannot reconstruct the information available at historical booking time, rule out every form of temporal leakage, or validate prospective fare advice. `Days_Before_Departure` and other recorded features are treated as supplied; their availability at quote time is unverified. The booking-window analysis is retrospective.

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

The tests check basic cleaning, booking-window output, and deterministic chronological date partitioning. They do **not** evaluate CatBoost model quality.

Retrain the model artifact:

```bash
python -m scripts.train_model
```

The training script fits the model on all priced rows and saves `models/price_model.cbm`; it does not calculate a held-out score. Run the separate evaluation command to fit on the chronological training dates and score validation and test partitions against a training-median baseline:

```bash
python -m scripts.evaluate_model --data data/flight_pricing_dataset.csv --output artifacts/chronological_evaluation.json
```

See [the evaluation protocol](docs/EVALUATION.md) for split rules, reproducibility details, interpretation, and the missing-timestamp limitation. The evaluator does not overwrite the saved model artifact. Do not present the legacy `artifacts/metrics.json` values as verified held-out performance.

## Feature preparation

The pipeline normalizes duration strings to minutes, normalizes stop and passenger-count values, converts selected numeric text, parses the departure date into month and day-of-year, and represents missing categorical values as `Unknown`. The model uses categorical columns directly through CatBoost.

## Project layout

- `app.py` — Streamlit dashboard
- `src/data_utils.py` — cleaning and feature preparation
- `src/modeling.py` — model fit and artifact save
- `src/evaluation.py` — unique-departure-date chronological split and evaluation
- `scripts/evaluate_model.py` — reproducible CLI for validation/test metrics
- `docs/EVALUATION.md` — evaluation protocol and limitations
- `src/booking_analysis.py` — historical booking-window summaries
- `tests/` — lightweight data-pipeline checks
- `artifacts/` — stored metrics and analysis outputs; metric provenance is incomplete
- `docs/` — report, demo, and interview notes

## Next engineering steps

1. Run and review the documented evaluation in an environment with the required dependencies; preserve its output and exact dependency versions if reporting the scores.
2. Obtain a trustworthy fare-observation or booking timestamp before attempting prospective booking-time validation.
3. Add tests for feature preparation and end-to-end evaluation once the evaluation environment is available.

Until those steps are complete, describe the model as a training pipeline experiment rather than a validated fare predictor.
