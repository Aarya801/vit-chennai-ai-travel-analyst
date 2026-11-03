# Model evaluation protocol

## What this evaluation measures

`python -m scripts.evaluate_model` runs the fixed CatBoost configuration through a reproducible, departure-date-ordered train/validation/test split. It reports MAE, RMSE, and R² for both CatBoost and a median-price baseline. No score is checked in or claimed by this document; run the command with the required packages to produce the values and split metadata for the supplied dataset.

This is a **calendar-time proxy evaluation**, not a prospective fare-quote backtest. The CSV has `Departure_Date` and `Days_Before_Departure`, but no verified timestamp for when each fare quote was observed or captured. We therefore cannot reconstruct which prices and fields were available at a historical booking time, prevent every form of temporal leakage, or validate a true booking-time recommendation. Treat all metrics as exploratory only.

## Reproduce

Use Python 3.11, as in `.github/workflows/tests.yml`, and install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.evaluate_model --data data/flight_pricing_dataset.csv --output artifacts/chronological_evaluation.json
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.evaluate_model --data data/flight_pricing_dataset.csv --output artifacts/chronological_evaluation.json
```

The command also prints the JSON result. It records the input file's SHA-256, Python and relevant library versions, split fractions, date bounds, row counts, excluded-row counts, feature list, fixed model parameters, baseline definition, and scores. Use the same dataset file, code revision, and dependency versions to reproduce a result; the requirements specify minimum versions, not a fully locked environment.

## Split and fitting rules

1. Keep rows with a parseable `Price` and `Departure_Date`. Rows without a price are not scoreable; priced rows without a valid departure date are excluded from this chronological evaluation and counted in the output.
2. Sort the distinct eligible departure dates. Assign the first 60% of unique dates to train, the next 20% to validation, and the final 20% to test. Integer cutoffs are `int(0.60 * date_count)` and `int(0.80 * date_count)`, with at least one date in each partition. Rows sharing a departure date stay together, including exact duplicate rows.
3. Fit CatBoost only on training rows, using the existing fixed configuration (350 iterations, depth 8, learning rate 0.08, RMSE loss, random seed 42, one CPU thread) and `log1p(Price)` target. There is no hyperparameter search or validation-based model selection; validation is reported as a diagnostic partition.
4. Compare against a baseline that predicts the training-partition median price for every validation and test row. Metrics use original price units after inverse-transforming CatBoost predictions with `expm1`.
5. Report validation and test scores separately. Do not call them future fare accuracy or prospective booking-time results.

The split is by **unique departure dates**, not by row count, so exact 60/20/20 row proportions are not guaranteed. The input's missing dates and uneven number of rows per date affect each partition's row count. Grouping by date prevents one departure date from crossing partitions, but does not establish when its fares were observed.

## Training a final artifact

`python -m scripts.train_model` is a separate artifact-training command: it fits on all rows with a known price and saves `models/price_model.cbm`. It does not create or score a split. Do not use that model artifact or `artifacts/metrics.json` as evaluation evidence. The evaluation command fits its own model on the chronological training partition and does not overwrite the saved artifact.

## Tests

The standard project tests remain:

```bash
pytest -q
```

The added split tests use only Python's standard library and can also be run on their own:

```bash
python -m unittest discover -s tests -p test_evaluation.py -v
```

These tests verify deterministic date ordering, date-group separation, and invalid split handling. They do not run CatBoost or prove the model's predictive quality.
