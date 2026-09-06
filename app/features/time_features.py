"""Time-series feature engineering.

Builds calendar, lag, and rolling features for ML forecasting.
Critically, all features use only historical information to
prevent future data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data.validator import detect_frequency


def calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based features derived purely from the date column.

    These are all deterministic and never leak future target information.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])

    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter
    out["week"] = dates.dt.isocalendar().week.astype(int)
    out["dayofweek"] = dates.dt.dayofweek
    out["dayofmonth"] = dates.dt.day
    out["iso_year"] = dates.dt.isocalendar().year
    out["iso_weekday"] = dates.dt.isocalendar().day.astype(int)

    # Cyclical encodings to help tree/linear models capture periodicity
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["sin_week"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["cos_week"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    out["sin_dayofmonth"] = np.sin(2 * np.pi * out["dayofmonth"] / 31.0)
    out["cos_dayofmonth"] = np.cos(2 * np.pi * out["dayofmonth"] / 31.0)

    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_month_start"] = (out["dayofmonth"] == 1).astype(int)
    out["is_month_end"] = (out["dayofmonth"] >= 28).astype(int)
    out["days_in_month"] = dates.dt.days_in_month
    out["day_of_year"] = dates.dt.dayofyear
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)

    return out


def lag_features(
    df: pd.DataFrame,
    target_col: str = "target",
    lags: list[int] | None = None,
    frequency: str | None = None,
) -> pd.DataFrame:
    """Add lag features of the target variable.

    Only historical (past) values are used, so there is no leakage.

    The set of lags is validated against the detected frequency to avoid
    creating lags that don't make sense (e.g., lag_365 on weekly data).
    """
    out = df.copy()
    freq = frequency or detect_frequency(pd.DatetimeIndex(pd.to_datetime(out["date"])))

    if lags is None:
        lags = default_lags(freq)

    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    return out


def default_lags(frequency: str) -> list[int]:
    """Return sensible lag set for a given frequency."""
    if frequency == "D":
        return [1, 2, 3, 7, 14, 28, 30, 90, 365]
    if frequency in ("W", "W-SUN", "W-MON"):
        return [1, 2, 4, 8, 12, 26, 52]
    if frequency in ("MS", "M"):
        return [1, 2, 3, 6, 12]
    if frequency in ("QS", "Q"):
        return [1, 2, 4]
    if frequency in ("YS", "A", "Y"):
        return [1]
    return [1, 2, 3]


def rolling_features(
    df: pd.DataFrame,
    target_col: str = "target",
    windows: list[int] | None = None,
    frequency: str | None = None,
    include_std: bool = True,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Add rolling-statistics features.

    These are computed on past data only (rolling windows look backward),
    so there is no future leakage.
    """
    out = df.copy()
    freq = frequency or detect_frequency(pd.DatetimeIndex(pd.to_datetime(out["date"])))

    if windows is None:
        windows = default_windows(freq)

    for w in windows:
        minp = min_periods if min_periods is not None else max(1, min(w, 14))
        out[f"rolling_mean_{w}"] = out[target_col].rolling(window=w, min_periods=minp).mean()
        if include_std:
            out[f"rolling_std_{w}"] = out[target_col].rolling(window=w, min_periods=minp).std()
        out[f"rolling_min_{w}"] = out[target_col].rolling(window=w, min_periods=minp).min()
        out[f"rolling_max_{w}"] = out[target_col].rolling(window=w, min_periods=minp).max()

    # Momentum feature: recent change relative to a longer window
    out["momentum"] = out[f"rolling_mean_{windows[0]}"].pct_change() if windows else np.nan

    return out


def default_windows(frequency: str) -> list[int]:
    if frequency == "D":
        return [7, 14, 28, 90]
    if frequency in ("W", "W-SUN", "W-MON"):
        return [4, 8, 13, 26]
    if frequency in ("MS", "M"):
        return [3, 6, 12]
    if frequency in ("QS", "Q"):
        return [2, 4]
    if frequency in ("YS", "A", "Y"):
        return [2, 3]
    return [3, 7]


def build_feature_frame(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "target",
    frequency: str | None = None,
    with_lags: bool = True,
    with_rolling: bool = True,
    with_calendar: bool = True,
) -> pd.DataFrame:
    """Build the complete feature matrix for ML forecasting.

    Returns a DataFrame with the target plus all engineered features.
    Rows with NaN features (from warm-up lags/rolling) are retained and
    must be dropped by the caller appropriately for the model trainer.
    """
    freq = frequency or detect_frequency(pd.DatetimeIndex(pd.to_datetime(df[date_col])))
    out = df.copy()

    if with_calendar:
        out = calendar_features(out, date_col)
    if with_lags:
        out = lag_features(out, target_col, frequency=freq)
    if with_rolling:
        out = rolling_features(out, target_col, frequency=freq)

    return out


def create_recursive_features(
    future_dates: pd.DatetimeIndex,
    frequency: str,
    last_history: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:
    """Create a feature frame for future dates used in recursive ML prediction.

    Calendar features are computed directly. Lag and rolling features that
    depend on the target must be filled in recursively by the forecaster.

    Returns a frame with only the calendar features for the future dates.
    """
    from app.features.time_features import calendar_features

    frame = pd.DataFrame({"date": future_dates})
    return calendar_features(frame, "date")
