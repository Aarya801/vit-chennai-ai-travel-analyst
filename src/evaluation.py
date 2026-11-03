"""Reproducible departure-date-ordered evaluation for the exploratory fare model.

This split is a calendar-time proxy only. The dataset has no verified fare quote
timestamp, so the results cannot establish prospective booking-time performance.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable


def chronological_date_splits(
    departure_dates: Iterable[date],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, tuple[date, ...]]:
    """Split unique departure dates in order so one date cannot cross partitions."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave dates for test")

    dates = sorted(set(departure_dates))
    if len(dates) < 3:
        raise ValueError("at least three distinct departure dates are required")

    train_end = max(1, int(len(dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    if validation_end >= len(dates):
        raise ValueError("split fractions leave no dates for test")

    return {
        "train": tuple(dates[:train_end]),
        "validation": tuple(dates[train_end:validation_end]),
        "test": tuple(dates[validation_end:]),
    }


def evaluate_chronological(
    data_path: str = "data/flight_pricing_dataset.csv",
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict:
    """Fit the fixed CatBoost model on train dates and report validation/test scores."""
    import hashlib
    import platform
    from pathlib import Path
    import catboost
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor
    import sklearn
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from .data_utils import CAT_FEATURES, MODEL_FEATURES, clean_dataframe
    from .modeling import prepare_features

    source = Path(data_path)
    raw = pd.read_csv(source)
    departure_dates = pd.to_datetime(raw["Departure_Date"], errors="coerce")
    cleaned = clean_dataframe(raw)
    has_price = cleaned["Price"].notna()
    has_date = departure_dates.notna()
    undated_priced_rows = int((has_price & ~has_date).sum())
    eligible = cleaned.loc[has_price & has_date].copy()
    eligible["_departure_date"] = departure_dates.loc[eligible.index].dt.date

    date_partitions = chronological_date_splits(
        eligible["_departure_date"].tolist(), train_fraction, validation_fraction
    )
    masks = {
        name: eligible["_departure_date"].isin(dates)
        for name, dates in date_partitions.items()
    }
    partitions = {name: eligible.loc[mask] for name, mask in masks.items()}

    model = CatBoostRegressor(
        iterations=350,
        depth=8,
        learning_rate=0.08,
        loss_function="RMSE",
        random_seed=42,
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
    )
    train = partitions["train"]
    model.fit(
        prepare_features(train),
        np.log1p(train["Price"].astype(float)),
        cat_features=CAT_FEATURES,
    )
    training_median = float(train["Price"].median())

    def score(frame: pd.DataFrame) -> dict:
        actual = frame["Price"].to_numpy(dtype=float)
        predicted = np.expm1(model.predict(prepare_features(frame)))
        baseline = np.full(actual.shape, training_median, dtype=float)
        return {
            "model": _regression_metrics(actual, predicted, mean_absolute_error, mean_squared_error, r2_score),
            "training_median_baseline": _regression_metrics(actual, baseline, mean_absolute_error, mean_squared_error, r2_score),
        }

    split_summary = {}
    for name, frame in partitions.items():
        dates = date_partitions[name]
        split_summary[name] = {
            "rows": int(len(frame)),
            "unique_departure_dates": len(dates),
            "first_departure_date": dates[0].isoformat(),
            "last_departure_date": dates[-1].isoformat(),
        }

    return {
        "evaluation": "departure-date-ordered proxy; not prospective quote-time validation",
        "source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "runtime_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "catboost": catboost.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "split_fractions": {
            "train_unique_dates": train_fraction,
            "validation_unique_dates": validation_fraction,
            "test_unique_dates": 1 - train_fraction - validation_fraction,
        },
        "excluded_rows": {
            "missing_price": int((~has_price).sum()),
            "priced_rows_missing_departure_date": undated_priced_rows,
        },
        "features": MODEL_FEATURES,
        "target_transform": "log1p(Price); predictions transformed back with expm1",
        "model_parameters": {
            "iterations": 350,
            "depth": 8,
            "learning_rate": 0.08,
            "loss_function": "RMSE",
            "random_seed": 42,
            "thread_count": 1,
        },
        "baseline": "training-set median Price, used unchanged on validation and test",
        "validation_role": "reported only; no tuning or model selection is performed",
        "splits": split_summary,
        "scores": {
            "validation": score(partitions["validation"]),
            "test": score(partitions["test"]),
        },
        "limitations": [
            "Departure_Date is not a fare observation or booking timestamp.",
            "The chronological split orders travel dates, not when the fare became observable.",
            "Days_Before_Departure and other fields are assumed available as recorded; their quote-time availability is unverified.",
            "These scores do not establish prospective booking-time performance or future fare accuracy.",
        ],
    }


def _regression_metrics(actual, predicted, mae, mse, r2) -> dict:
    import numpy as np

    return {
        "mae": float(mae(actual, predicted)),
        "rmse": float(np.sqrt(mse(actual, predicted))),
        "r2": float(r2(actual, predicted)),
    }


def write_evaluation(result: dict, output_path: str) -> None:
    import json
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
