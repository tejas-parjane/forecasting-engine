"""Model registry: constructs and names forecasting models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.analysis.seasonality import detect_seasonality
from app.data.validator import detect_frequency
from app.models.arima import ARIMAForecaster
from app.models.base import BaseForecaster
from app.models.ets import ExponentialSmoothingForecaster
from app.models.naive import NaiveForecaster, SeasonalNaiveForecaster
from app.models.xgboost_model import XGBoostForecaster


def create_models(
    series: pd.Series,
    frequency: str | None = None,
    with_xgboost: bool = True,
    min_xgboost_points: int = 40,
) -> list[BaseForecaster]:
    """Instantiate all forecasting models appropriate for the given data.

    Seasonality information is passed to models that can exploit it.
    XGBoost is included only when enough history exists for meaningful
    lag/rolling features.
    """
    freq = frequency or detect_frequency(series.index)
    se = detect_seasonality(series, frequency=freq)

    models: list[BaseForecaster] = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(season_length=se.get("period", 7) or 7),
    ]

    # ETS
    periods = se.get("periods", {})
    ets = ExponentialSmoothingForecaster(
        seasonality_detected=bool(se.get("detected", False)),
        trend_detected=True,
        periods=periods,
    )
    models.append(ets)

    # ARIMA / SARIMA
    arima = ARIMAForecaster(seasonal=True)
    models.append(arima)

    # XGBoost when enough data
    if with_xgboost and len(series) >= min_xgboost_points:
        xgb = XGBoostForecaster()
        models.append(xgb)

    return models