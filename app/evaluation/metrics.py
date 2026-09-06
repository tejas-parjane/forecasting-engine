"""Forecast evaluation metrics.

All metrics are computed on actuals vs forecasts. For time-series
forecasting we emphasize:
  - MAE: interpretable average absolute error
  - RMSE: penalizes large errors disproportionately
  - MAPE: percentage error, undefined/explosive when actuals near zero
  - sMAPE: symmetric MAPE, bounded but asymmetric at times
  - WAPE: weighted absolute percentage error, robust to small denominators
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """Mean Absolute Percentage Error (percent).

    Returns None when actuals contain zeros (division undefined).
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if np.any(a == 0):
        return None
    return float(np.mean(np.abs((a - f) / a)) * 100.0)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (percent)."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(np.abs(a - f) / denom) * 100.0)


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (percent)."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(a))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(a - f)) / denom * 100.0)


@dataclass
class MetricResult:
    metric: str
    value: float | None
    note: str


def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, Any]:
    """Compute the full metric suite with notes on pitfalls."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if len(a) != len(f):
        raise ValueError("actual and forecast must have same length")

    results: dict[str, Any] = {}
    results["mae"] = round(mae(a, f), 4)
    results["rmse"] = round(rmse(a, f), 4)

    m = mape(a, f)
    results["mape"] = round(m, 4) if m is not None else None
    results["smape"] = round(smape(a, f), 4)
    results["wape"] = round(wape(a, f), 4)

    results["n"] = int(len(a))

    results["notes"] = {
        "mae": "Mean absolute error. Interpretable in target units; robust to outliers.",
        "rmse": "Root mean squared error. Penalizes large errors; sensitive to outliers.",
        "mape": (
            "Mean absolute percentage error. Undefined when actuals are 0; "
            "asymmetric and can explode for near-zero actuals."
        ),
        "smape": (
            "Symmetric MAPE. Bounded by 200%, symmetric in relative terms, "
            "but can be misleading when both values are near 0."
        ),
        "wape": (
            "Weighted APE. Sum of absolute errors divided by sum of actuals; "
            "robust to small denominators and stable for low-volume series."
        ),
    }
    return results


def compute_scaled_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, Any]:
    """Alternative: compute metrics normalized by mean absolute shift of naive.

    MASE-like helper for comparative model ranking.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    naive_errors = np.abs(np.diff(a))
    if len(naive_errors) > 0 and naive_errors.mean() > 0:
        mase = (np.abs(a - f).mean()) / naive_errors.mean()
    else:
        mase = None
    return {"mase": round(float(mase), 4) if mase is not None else None}