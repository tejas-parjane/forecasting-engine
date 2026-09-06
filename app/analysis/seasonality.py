"""Seasonality detection utilities for time-series data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.data.validator import detect_frequency


def detect_seasonality(
    series: pd.Series, frequency: str | None = None, min_periods: int | None = None
) -> dict[str, Any]:
    """Detect meaningful seasonal patterns in a series.

    Returns a structured dictionary describing which periods are present
    and their estimated strength (0-1).
    """
    freq = frequency or detect_frequency(series.index)

    candidates: dict[str, int] = {}
    if freq == "D":
        candidates = {"weekly": 7, "monthly": 30, "quarterly": 91, "yearly": 365}
    elif freq in ("W", "W-SUN", "W-MON"):
        candidates = {"monthly": 4, "quarterly": 13, "yearly": 52}
    elif freq in ("MS", "M"):
        candidates = {"monthly": 12, "quarterly": 4}
    elif freq in ("QS", "Q"):
        candidates = {"yearly": 4}
    elif freq in ("YS", "A", "Y"):
        candidates = {}

    max_period = min_periods or max(candidates.values(), default=7)

    # Require at least 3 full seasonal cycles for a period to be credible;
    # with fewer cycles, seasonal_decompose conflates trend with seasonality
    # and inflates strength estimates.
    min_cycles = 3
    if len(series) < min_cycles * min(candidates.values() if candidates else [7], default=7):
        return {
            "detected": False,
            "seasonality_type": "none",
            "periods": {},
            "note": "Insufficient data to detect seasonality",
        }

    from statsmodels.tsa.seasonal import seasonal_decompose

    periods_detected: dict[str, dict] = {}

    for name, period in candidates.items():
        if len(series) < min_cycles * period:
            continue
        try:
            decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
            residual_var = np.nanvar(decomp.resid)
            total_var = np.nanvar(series)
            if total_var == 0 or np.isnan(residual_var):
                strength = 0.0
            else:
                strength = max(0.0, min(1.0, 1.0 - residual_var / total_var))
            if strength >= 0.10:
                periods_detected[name] = {"period": period, "strength": round(strength, 3)}
        except Exception:
            continue

    if periods_detected and max(p["strength"] for p in periods_detected.values()) >= 0.10:
        primary = max(periods_detected, key=lambda k: periods_detected[k]["strength"])
        return {
            "detected": True,
            "seasonality_type": primary,
            "period": periods_detected[primary]["period"],
            "strength": periods_detected[primary]["strength"],
            "periods": periods_detected,
        }
    return {
        "detected": False,
        "seasonality_type": "none",
        "periods": periods_detected,
        "note": "No strong seasonal patterns detected",
    }
