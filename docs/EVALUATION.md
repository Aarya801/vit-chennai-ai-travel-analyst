# Model evaluation protocol

## What this evaluation measures

`python -m scripts.evaluate_model` runs the fixed CatBoost configuration through a reproducible, departure-date-ordered train/validation/test split. It reports MAE, RMSE, and R² for both CatBoost and a median-price baseline. The current run report is saved at [`../artifacts/chronological_evaluation.json`](../artifacts/chronological_evaluation.json). It records the input hash, runtime versions, split metadata, configuration, limitations, and scores. Re-run the command below to regenerate it.

This is a **calendar-time proxy evaluation**, not a prospective fare-quote backtest. The CSV has `Departure_Date` and `Days_Before_Departure`, but no verified timestamp for when each fare quote was observed or captured. We therefore cannot reconstruct which prices and fields were available at a historical booking time, prevent every form of temporal leakage, or validate a true booking-time recommendation. Treat all metrics as exploratory only.

## Current run results

The checked run used dataset SHA-256 `0b41ae9d512c2b6cb83f93ca48f9ff87d51172d1dd3aab13dac37220f982ef3d`, Python 3.11.15, NumPy 2.4.6, pandas 3.1.0, CatBoost 1.2.10, and scikit-learn 1.9.1.

| Partition | Model MAE | Model RMSE | Model R² | Training-median MAE | Training-median RMSE | Training-median R² |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 13,279.13 | 45,984.02 | 0.6435 | 55,831.93 | 80,448.87 | -0.0911 |
| Test | 13,100.29 | 46,156.53 | 0.6363 | 54,091.92 | 79,360.87 | -0.0753 |

MAE and RMSE use the dataset's price units. CatBoost beats this median baseline on all three reported metrics for both partitions in this run. This is a comparison on the supplied dataset and split only; it is not an independent validation and does not establish quote-time availability or prospective fare accuracy. The requirements file sets minimum dependency versions, so reproducing the same scores requires the versions recorded in the JSON report as well as the same code and data.

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
