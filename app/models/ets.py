"""Exponential smoothing (ETS) forecasting via Statsmodels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.models.base import BaseForecaster, ForecastResult


class ExponentialSmoothingForecaster(BaseForecaster):
    """Holt-Winters / exponential smoothing model.

    - Simple exponential smoothing for non-seasonal data.
    - Holt's linear trend when a trend is present.
    - Holt-Winters with automatic seasonal component selection when
      sufficient data exists for a detected period.
    """

    name = "ets"

    SUPPORTED_SEASONS = [7, 12, 30, 52, 91, 365]

    def __init__(self, seasonality_detected: bool = False, trend_detected: bool = True, periods: dict[str, dict] | None = None):
        super().__init__()
        self.seasonality_detected = seasonality_detected
        self.trend_detected = trend_detected
        self.periods = periods or {}
        self.results_: Any = None
        self.use_model_ = "ets"

    def _choose_seasonal_period(self, series: pd.Series) -> int | None:
        """Pick the strongest seasonal period that fits the available data.

        At least three full seasonal cycles are required so the seasonal
        component can be estimated credibly.
        """
        if not self.seasonality_detected:
            return None

        # Prefer the detected primary period if valid; else scan candidates.
        candidates = [
            p["period"] for p in self.periods.values()
        ]
        for period in sorted(candidates):
            if period in self.SUPPORTED_SEASONS and len(series) >= 3 * period:
                return int(period)

        for period in self.SUPPORTED_SEASONS:
            if len(series) >= 3 * period:
                return int(period)
        return None

    def fit(self, series: pd.Series, **kwargs: Any) -> "ExponentialSmoothingForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        values = series.astype(float)
        if (values <= 0).any():
            # ExponentialSmoothing requires positive values for multiplicative.
            use_multiplicative = False
        else:
            use_multiplicative = True

        seasonal_period = self._choose_seasonal_period(values)
        trend_ = "add" if self.trend_detected else None
        seas_ = "add" if seasonal_period else None

        try:
            if seas_ and use_multiplicative and len(values) >= 2 * seasonal_period:
                model = ExponentialSmoothing(
                    values, trend=trend_, seasonal=seas_, seasonal_periods=seasonal_period,
                )
            else:
                model = ExponentialSmoothing(
                    values, trend=trend_, seasonal=seas_, seasonal_periods=seasonal_period,
                )
            fitted = model.fit(optimized=True, remove_bias=False)
            self.results_ = fitted
            self.use_model_ = "ets"
            return self
        except Exception:
            # Fall back to a simpler form if the seasonal fit fails.
            try:
                model = ExponentialSmoothing(values, trend=trend_)
                fitted = model.fit(optimized=True)
                self.results_ = fitted
                self.use_model_ = "ets_simple"
                return self
            except Exception:
                self.use_model_ = "ets_fallback"
                self.results_ = ExponentialSmoothing(values).fit()
                return self

    def predict(self, horizon: int, **kwargs: Any) -> ForecastResult:
        if self.results_ is None:
            raise RuntimeError("Model not fitted")

        fc = self.results_.forecast(horizon)
        fc_vals = np.asarray(fc, dtype=float)

        # Approximate prediction interval from in-sample residuals.
        resid = np.asarray(self.results_.resid, dtype=float)
        resid_std = np.std(resid) if len(resid) > 1 else 0.0
        z = 1.96
        sqrt_scale = np.sqrt(np.arange(1, horizon + 1))
        lower = fc_vals - z * resid_std * sqrt_scale
        upper = fc_vals + z * resid_std * sqrt_scale

        return ForecastResult(
            model_name=self.name,
            horizon=horizon,
            forecast=fc_vals,
            lower=lower,
            upper=upper,
            method=f"exponential smoothing ({self.use_model_})",
            metadata={
                "seasonal_period": self.results_.model.seasonal_periods,
                "trend": bool(self.results_.model.trend),
            },
        )