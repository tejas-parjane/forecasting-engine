"""Automated Exploratory Data Analysis for time-series data.

Produces structured summaries of trend, cycle, and volatility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.data.validator import _iqr_outliers, detect_frequency


@dataclass
class EDAResult:
    n_observations: int
    date_range: dict[str, str]
    frequency: str
    trend: dict[str, Any]
    seasonality: dict[str, Any]
    volatility: dict[str, Any]
    anomalies: dict[str, Any]
    summary_stats: dict[str, float]
    decomposition: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "date_range": self.date_range,
            "frequency": self.frequency,
            "trend": self.trend,
            "seasonality": self.seasonality,
            "volatility": self.volatility,
            "anomalies": self.anomalies,
            "summary_stats": self.summary_stats,
            "decomposition": self.decomposition,
        }


def _detect_periodicity(series: pd.Series, freq: str) -> dict[str, Any]:
    """Detect likely seasonal periods based on frequency.

    Returns a dict with detected seasonalities and their strengths.
    """
    periods: dict[str, int] = {}
    if freq == "D":
        periods["weekly"] = 7
        periods["yearly"] = 365
    elif freq in ("W", "MS", "M"):
        periods["yearly"] = 12 if freq != "W" else 52
        periods["monthly"] = 4 if freq == "W" else 12
    elif freq in ("QS", "Q"):
        periods["yearly"] = 4
    elif freq == "YS" or freq == "A":
        periods = {}

    result: dict[str, Any] = {
        "primary": None,
        "strength": None,
        "periods": {},
    }

    for name, period in periods.items():
        if len(series) < 3 * period:
            continue
        strength = _seasonal_strength(series, period)
        result["periods"][name] = {"period": period, "strength": round(strength, 3)}
        if result["primary"] is None or strength > result["strength"]:
            result["primary"] = name
            result["strength"] = strength

    return result


def _seasonal_strength(series: pd.Series, period: int) -> float:
    """Estimate seasonal strength (0-1) for a given period."""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        if len(series) < 2 * period:
            return 0.0
        decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
        resid_var = np.nanvar(decomp.resid)
        total_var = np.nanvar(series)
        if total_var == 0 or np.isnan(resid_var) or np.isnan(total_var):
            return 0.0
        return max(0.0, 1.0 - resid_var / total_var)
    except Exception:
        return 0.0


def _detect_trend(series: pd.Series) -> dict[str, Any]:
    """Estimate trend direction and slope via least-squares linear fit."""
    x = np.arange(len(series), dtype=float)
    valid = ~np.isnan(series.values) & ~np.isinf(series.values)
    if valid.sum() < 2:
        return {
            "direction": "unknown",
            "slope_per_day": None,
            "slope_pct_per_period": None,
            "p_value": None,
        }

    y = series.values[valid].astype(float)
    xs = x[valid]
    n = len(xs)
    x_mean = xs.mean()
    slope = np.sum((xs - x_mean) * (y - y.mean())) / np.sum((xs - x_mean) ** 2)
    intercept = y.mean() - slope * x_mean

    # t-statistic for slope
    resid = y - (intercept + slope * xs)
    sse = np.sum(resid**2)
    if n > 2 and sse > 0:
        mse = sse / (n - 2)
        se_slope = math.sqrt(mse / np.sum((xs - x_mean) ** 2))
        if se_slope > 0:
            t_stat = slope / se_slope
            p_value = 2 * (1 - _t_cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0
    else:
        p_value = 1.0

    mean_y = float(np.mean(y))
    slope_pct = (slope / mean_y * 100.0) if mean_y != 0 else None

    if p_value is not None and p_value > 0.05:
        direction = "stable"
    elif slope > 0:
        direction = "increasing"
    elif slope < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "slope_per_period": round(float(slope), 6),
        "slope_pct_per_period": round(slope_pct, 4) if slope_pct is not None else None,
        "p_value": round(p_value, 4) if p_value is not None else None,
    }


def _t_cdf(t: float, df: float) -> float:
    """Approximate the Student's t CDF using a normal approximation for large df,
    and a simple numeric integration otherwise."""
    if df > 100:
        return _norm_cdf(t)
    # Simpson's rule integration of the t pdf from -inf to t
    def t_pdf(x):
        return (
            math.gamma((df + 1) / 2)
            / (math.sqrt(df * math.pi) * math.gamma(df / 2))
            * (1 + x**2 / df) ** (-(df + 1) / 2)
        )
    a = -50.0
    n_steps = 2000
    h = (t - a) / n_steps
    if h <= 0:
        return 0.0
    total = 0.5 * (t_pdf(a) + t_pdf(t))
    for i in range(1, n_steps):
        total += t_pdf(a + i * h) if i % 2 == 1 else 2 * t_pdf(a + i * h)
    return total * h / 3.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _detect_volatility(series: pd.Series) -> dict[str, Any]:
    """Measure rolling variability and identify volatility shifts."""
    if len(series) < 15:
        return {
            "mean_sd": None,
            "recent_sd": None,
            "sd_ratio": None,
            "trend": "unknown",
        }
    window = max(7, min(28, len(series) // 5))
    rolling_sd = series.rolling(window=window, min_periods=window // 2).std()
    overall_sd = float(series.std())

    recent = rolling_sd.dropna().iloc[-1] if rolling_sd.notna().any() else float("nan")
    early = rolling_sd.dropna().iloc[0] if rolling_sd.notna().any() else float("nan")

    sd_ratio = (recent / early) if early and early > 0 and not np.isnan(early) else None

    if sd_ratio is None or np.isnan(sd_ratio):
        trend = "unknown"
    elif sd_ratio > 1.25:
        trend = "increasing"
    elif sd_ratio < 0.8:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "overall_std": round(float(overall_sd), 4) if not np.isnan(overall_sd) else None,
        "recent_std": round(float(recent), 4) if not np.isnan(recent) else None,
        "early_std": round(float(early), 4) if not np.isnan(early) else None,
        "ratio_recent_to_early": round(sd_ratio, 4) if sd_ratio is not None else None,
        "trend": trend,
    }


class EDAAnalyzer:
    """Runs the full EDA pipeline on a preprocessed time series."""

    def analyze(self, series: pd.Series, frequency: str | None = None) -> EDAResult:
        freq = frequency or detect_frequency(series.index)

        trend = _detect_trend(series)
        seasonality = _detect_periodicity(series, freq)
        volatility = _detect_volatility(series)

        outlier_mask = _iqr_outliers(series)
        anomaly_indices = series.index[outlier_mask].tolist()
        anomalies = {
            "count": int(outlier_mask.sum()),
            "method": "IQR (3x) on target",
            "dates": [str(d.date()) for d in anomaly_indices][:50],
        }

        summary_stats = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "recent_value": float(series.iloc[-1]),
            "first_value": float(series.iloc[0]),
            "latest_change_pct": round(
                float((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] * 100), 4
            )
            if len(series) > 1 and series.iloc[-2] != 0
            else None,
        }

        decomposition = None
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            period = seasonality.get("periods", {}).get("weekly", {}).get("period") or 7
            if freq == "D" and len(series) >= 2 * 7:
                decomp = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
                decomposition = {
                    "method": "additive",
                    "period": period,
                    "trend_last": float(decomp.trend.dropna().iloc[-1]) if decomp.trend.dropna().size else None,
                    "seasonal_magnitude": float(np.nanmax(np.abs(decomp.seasonal))) if decomp.seasonal.notna().any() else None,
                    "residual_std": float(np.nanstd(decomp.resid)) if decomp.resid.notna().any() else None,
                }
        except Exception:
            decomposition = None

        return EDAResult(
            n_observations=len(series),
            date_range={
                "start": str(series.index.min().date()),
                "end": str(series.index.max().date()),
            },
            frequency=freq,
            trend=trend,
            seasonality=seasonality,
            volatility=volatility,
            anomalies=anomalies,
            summary_stats=summary_stats,
            decomposition=decomposition,
        )
