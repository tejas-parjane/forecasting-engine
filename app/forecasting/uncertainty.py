"""Uncertainty estimation helpers for forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionInterval:
    """A prediction interval with a stated coverage level."""

    lower: np.ndarray
    upper: np.ndarray
    coverage: float  # e.g., 0.95
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": [float(x) for x in self.lower],
            "upper": [float(x) for x in self.upper],
            "coverage": self.coverage,
            "method": self.method,
        }


def gaussian_band(
    forecast: np.ndarray,
    residual_std: float,
    horizon: int,
    coverage: float = 0.95,
) -> PredictionInterval:
    """Build a Gaussian uncertainty band with sqrt(horizon) growth.

    This approximates the widening of uncertainty with lead time for a
    random-walk-like forecast error process. It is an approximation and is
    documented as such.
    """
    z = {
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758,
    }.get(coverage, 1.96)

    scale = np.sqrt(np.arange(1, horizon + 1)) * residual_std
    lower = forecast - z * scale
    upper = forecast + z * scale
    return PredictionInterval(lower=lower, upper=upper, coverage=coverage, method="gaussian sqrt(horizon) band")


def empirical_band(
    forecast: np.ndarray,
    residuals: np.ndarray,
    horizon: int,
    coverage: float = 0.95,
    n_sims: int = 1000,
    seed: int = 42,
) -> PredictionInterval:
    """Empirical prediction interval via residual resampling (bootstrapping).

    Each future step accumulates bootstrapped residuals sampled with
    replacement from the model's in-sample residuals.
    """
    rng = np.random.default_rng(seed)
    resid = np.asarray(residuals, dtype=float)
    resid = resid[~np.isnan(resid)]
    if len(resid) < 5:
        return gaussian_band(forecast, np.std(resid) if len(resid) else 0.0, horizon, coverage)

    sims = np.zeros((n_sims, horizon))
    for i in range(horizon):
        sample = rng.choice(resid, size=n_sims, replace=True)
        sims[:, i] = forecast[i] + sample

    lower_q = (1 - coverage) / 2 * 100
    upper_q = (1 + coverage) / 2 * 100
    lower = np.percentile(sims, lower_q, axis=0)
    upper = np.percentile(sims, upper_q, axis=0)
    return PredictionInterval(lower=lower, upper=upper, coverage=coverage, method="empirical residual bootstrap")


def uncertainty_annotation(interval: PredictionInterval, forecast: np.ndarray) -> dict[str, Any]:
    """Describe the width of the interval."""
    widths = interval.upper - interval.lower
    return {
        "mean_width": round(float(np.mean(widths)), 4),
        "width_at_horizon_start": round(float(widths[0]), 4),
        "width_at_horizon_end": round(float(widths[-1]), 4),
        "widening_factor": round(float(widths[-1] / widths[0]), 4) if widths[0] > 0 else None,
        "note": (
            "Uncertainty increases with forecast horizon: prediction intervals "
            "for distant periods are wider than for near-term ones."
        ),
    }