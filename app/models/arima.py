"""ARIMA / SARIMA forecasting via Statsmodels with automatic order selection."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from app.models.base import BaseForecaster, ForecastResult

warnings.filterwarnings("ignore")


class ARIMAForecaster(BaseForecaster):
    """Auto-determined ARIMA/SARIMA model.

    Order selection uses a curated grid of order candidates scored by AIC,
    preferring small, interpretable models. A time budget caps the search so
    backtesting remains practical on real data.

    Order selection:

      - Detects a seasonal period when sufficient history exists.
      - Scores a bounded set of candidate orders by AIC (with a small
        complexity penalty to break ties toward simpler models).
      - Falls back to a simple ARIMA(1,1,1) if the search fails.
    """

    name = "arima"

    def __init__(self, seasonal: bool = True, time_budget_seconds: int = 45):
        super().__init__()
        self.seasonal = seasonal
        self.time_budget_seconds = time_budget_seconds
        self.results_: Any = None
        self.order_: tuple = ()
        self.seasonal_order_: tuple = ()
        self.selected_frequency_: str = ""

    def _find_seasonal_period(self, series: pd.Series, frequency: str = "D") -> int | None:
        candidates = [7, 12, 30, 52, 91, 365]
        preferred = {"D": 7, "W": 12, "MS": 12, "M": 12, "QS": 4}
        pref = preferred.get(frequency)
        if pref and len(series) >= 4 * pref:
            return pref
        for c in candidates:
            if len(series) >= 4 * c:
                return c
        return None

    def fit(self, series: pd.Series, frequency: str = "D", **kwargs: Any) -> "ARIMAForecaster":
        from statsmodels.tsa.arima.model import ARIMA

        self.selected_frequency_ = frequency
        values = series.astype(float)
        values = values.reset_index(drop=True)

        seasonal_period = None
        if self.seasonal:
            seasonal_period = self._find_seasonal_period(values, frequency)

        import time

        deadline = time.monotonic() + self.time_budget_seconds
        best_aic = float("inf")
        best_fit = None
        best_order = None
        best_seas = None

        if seasonal_period and len(values) >= 4 * seasonal_period:
            candidates = [
                ((1, 1, 1), (0, 0, 0, seasonal_period)),
                ((1, 1, 0), (0, 0, 0, seasonal_period)),
                ((2, 1, 0), (0, 0, 0, seasonal_period)),
                ((1, 1, 2), (0, 0, 0, seasonal_period)),
                ((0, 1, 1), (1, 0, 0, seasonal_period)),
                ((1, 1, 1), (1, 0, 0, seasonal_period)),
                ((1, 0, 1), (1, 1, 0, seasonal_period)),
                ((2, 1, 1), (1, 1, 0, seasonal_period)),
                ((1, 1, 2), (1, 1, 0, seasonal_period)),
            ]
        else:
            candidates = [
                ((0, 1, 0), None),
                ((1, 1, 0), None),
                ((0, 1, 1), None),
                ((1, 1, 1), None),
                ((2, 1, 0), None),
                ((2, 1, 1), None),
                ((1, 1, 2), None),
                ((2, 1, 2), None),
                ((1, 0, 1), None),
                ((2, 0, 2), None),
            ]

        for order, seas_order in candidates:
            if time.monotonic() > deadline:
                break
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if seas_order:
                        fit = ARIMA(values, order=order, seasonal_order=seas_order).fit()
                    else:
                        fit = ARIMA(values, order=order).fit()
                complexity_penalty = 0.001 * (sum(order) + sum(seas_order[:3] if seas_order else []))
                aic = fit.aic + complexity_penalty
                if aic < best_aic:
                    best_aic = aic
                    best_fit = fit
                    best_order = order
                    best_seas = seas_order
            except Exception:
                continue

        if best_fit is None:
            best_order = (1, 1, 1)
            best_fit = ARIMA(values, order=best_order).fit()

        self.results_ = best_fit
        self.order_ = best_order
        self.seasonal_order_ = best_seas
        return self

    def predict(self, horizon: int, **kwargs: Any) -> ForecastResult:
        if self.results_ is None:
            raise RuntimeError("Model not fitted")

        steps = max(int(horizon), 1)
        fc = self.results_.get_forecast(steps=steps)
        mean = fc.predicted_mean.to_numpy()
        try:
            conf = fc.conf_int(alpha=0.05)
            lower = conf.iloc[:, 0].to_numpy()
            upper = conf.iloc[:, 1].to_numpy()
        except Exception:
            resid = np.asarray(self.results_.resid, dtype=float)
            resid_std = np.std(resid) if len(resid) > 1 else 0.0
            sqrt_scale = np.sqrt(np.arange(1, steps + 1))
            lower = mean - 1.96 * resid_std * sqrt_scale
            upper = mean + 1.96 * resid_std * sqrt_scale

        return ForecastResult(
            model_name=self.name,
            horizon=steps,
            forecast=mean,
            lower=lower,
            upper=upper,
            method=f"SARIMA{self.order_}" if self.seasonal_order_ else f"ARIMA{self.order_}",
            metadata={
                "order": list(self.order_),
                "seasonal_order": list(self.seasonal_order_) if self.seasonal_order_ else None,
            },
        )