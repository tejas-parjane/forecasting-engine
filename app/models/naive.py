"""Naive and seasonal-naive baseline forecasting models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.models.base import BaseForecaster, ForecastResult


class NaiveForecaster(BaseForecaster):
    """Naive model: forecast equals the most recent observed value."""

    name = "naive"

    def __init__(self):
        super().__init__()
        self.last_value_: float | None = None

    def fit(self, series: pd.Series, **kwargs: Any) -> "NaiveForecaster":
        self.last_value_ = float(series.iloc[-1])
        return self

    def predict(self, horizon: int, **kwargs: Any) -> ForecastResult:
        if self.last_value_ is None:
            raise RuntimeError("Model not fitted")
        fc = np.full(horizon, self.last_value_)
        return ForecastResult(
            model_name=self.name,
            horizon=horizon,
            forecast=fc,
            method="last observed value",
            metadata={"last_value": self.last_value_},
        )


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive model: forecast equals the value from one season back.

    For daily data with weekly seasonality, the forecast for a given weekday
    equals the observed value from the same weekday last week.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 7):
        super().__init__()
        self.season_length = season_length
        self.series_: pd.Series | None = None

    def fit(self, series: pd.Series, **kwargs: Any) -> "SeasonalNaiveForecaster":
        self.series_ = series
        return self

    def predict(self, horizon: int, **kwargs: Any) -> ForecastResult:
        if self.series_ is None:
            raise RuntimeError("Model not fitted")
        s = self.series_

        if len(s) < self.season_length:
            fc = np.full(horizon, float(s.iloc[-1]))
            method = "naive fallback (insufficient history for season length)"
        else:
            fc = np.empty(horizon)
            for i in range(horizon):
                idx = len(s) - self.season_length + (i % self.season_length)
                if idx < 0:
                    idx = 0
                fc[i] = s.iloc[idx]
            method = f"seasonal naive (L={self.season_length})"

        return ForecastResult(
            model_name=self.name,
            horizon=horizon,
            forecast=fc,
            method=method,
            metadata={"season_length": self.season_length},
        )