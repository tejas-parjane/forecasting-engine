"""Anomaly detection for time-series data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_anomalies(
    series: pd.Series, method: str = "iqr", iqr_multiple: float = 3.0, window: int | None = None
) -> dict[str, Any]:
    """Detect anomalies in a time series.

    Methods:
        - 'iqr': static IQR-based outlier detection on the full series.
        - 'rolling': residual-based detection on a rolling baseline.

    Returns:
        dict with anomaly indices, count, and a per-point boolean mask.
    """
    if method == "rolling":
        return _rolling_anomalies(series, window=window or 28)
    return _iqr_anomalies(series, iqr_multiple=iqr_multiple)


def _iqr_anomalies(series: pd.Series, iqr_multiple: float) -> dict[str, Any]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        mask = pd.Series(False, index=series.index)
    else:
        lower = q1 - iqr_multiple * iqr
        upper = q3 + iqr_multiple * iqr
        mask = (series < lower) | (series > upper)

    anomaly_mask = mask.reset_index(drop=True)
    anomaly_indices = series.index[mask.values].tolist()

    return {
        "method": f"iqr_x{iqr_multiple}",
        "count": int(mask.sum()),
        "anomaly_dates": [str(d.date()) for d in anomaly_indices],
        "mask": anomaly_mask.values.tolist(),
    }


def _rolling_anomalies(series: pd.Series, window: int = 28) -> dict[str, Any]:
    """Detect anomalies relative to a rolling median ± 3 * rolling MAD."""
    rolling_med = series.rolling(window=window, min_periods=max(7, window // 2)).median()
    mad = (series - rolling_med).abs().rolling(window=window, min_periods=max(7, window // 2)).median()
    mad = mad.replace(0, np.nan)
    threshold = 3.0 * 1.4826 * mad
    mask = (series - rolling_med).abs() > threshold
    mask = mask.fillna(False).astype(bool)

    anomaly_indices = series.index[mask.values].tolist()
    return {
        "method": f"rolling_mad_window{window}",
        "count": int(mask.sum()),
        "anomaly_dates": [str(d.date()) for d in anomaly_indices],
        "mask": mask.values.tolist(),
    }
