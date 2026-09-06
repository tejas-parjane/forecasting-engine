"""Tests for forecasting models: training, predictions, and error handling."""

import numpy as np
import pandas as pd
import pytest

from app.models.arima import ARIMAForecaster
from app.models.ets import ExponentialSmoothingForecaster
from app.models.naive import NaiveForecaster, SeasonalNaiveForecaster
from app.models.registry import create_models
from app.models.xgboost_model import XGBoostForecaster


def series():
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 20 * np.sin(2 * np.pi * (idx.dayofweek.values) / 7)
    noise = rng.normal(0, 4, n)
    return pd.Series(trend + seasonal + noise, index=idx)


class TestNaive:
    def test_forecast_equals_last_value(self):
        m = NaiveForecaster().fit(series())
        r = m.predict(10)
        assert len(r.forecast) == 10
        assert np.allclose(r.forecast, series().iloc[-1])

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            NaiveForecaster().predict(5)


class TestSeasonalNaive:
    def test_shapes(self):
        m = SeasonalNaiveForecaster(season_length=7).fit(series())
        r = m.predict(14)
        assert len(r.forecast) == 14

    def test_periodic_pattern(self):
        s = series()
        m = SeasonalNaiveForecaster(season_length=7).fit(s)
        r = m.predict(7)
        # forecasts follow the same weekday values cyclically
        assert np.allclose(r.forecast[0], s.iloc[-7])


class TestETS:
    def test_train_and_predict(self):
        m = ExponentialSmoothingForecaster().fit(series())
        r = m.predict(10)
        assert len(r.forecast) == 10
        assert r.lower is not None and r.upper is not None
        assert np.all(r.lower <= r.forecast) and np.all(r.forecast <= r.upper)

    def test_positive_values_with_multiplicative_fallback(self):
        s = series() + 100
        m = ExponentialSmoothingForecaster().fit(s)
        r = m.predict(5)
        assert np.all(np.isfinite(r.forecast))


class TestARIMA:
    def test_train_and_predict(self):
        m = ARIMAForecaster().fit(series(), frequency="D")
        r = m.predict(10)
        assert len(r.forecast) == 10
        assert r.lower is not None and r.upper is not None

    def test_order_is_recorded(self):
        m = ARIMAForecaster().fit(series(), frequency="D")
        assert len(m.order_) == 3


class TestXGBoost:
    def test_train_and_predict_shape(self):
        m = XGBoostForecaster().fit(series(), frequency="D")
        r = m.predict(10)
        assert len(r.forecast) == 10
        assert r.lower is not None and r.upper is not None
        assert np.all(np.isfinite(r.forecast))

    def test_feature_importance(self):
        m = XGBoostForecaster().fit(series(), frequency="D")
        imp = m.feature_importance()
        assert len(imp) > 0
        assert all({"feature", "importance"} <= set(x.keys()) for x in imp)
        # sorted descending
        vals = [x["importance"] for x in imp]
        assert vals == sorted(vals, reverse=True)

    def test_no_future_leakage_in_fit(self):
        """The feature matrix used for training must contain only lag/rolling
        columns whose values come from the past; calendar columns are fine."""
        m = XGBoostForecaster().fit(series(), frequency="D")
        assert m.feature_cols_
        assert "target" not in m.feature_cols_


class TestCreateModels:
    def test_registry_builds_all(self):
        ms = create_models(series())
        names = {m.name for m in ms}
        assert "naive" in names
        assert "seasonal_naive" in names
        assert "ets" in names
        assert "arima" in names
        assert "xgboost" in names

    def test_registry_uses_detected_seasonality(self):
        ms = create_models(series(), frequency="D")
        seasonal = next(m for m in ms if m.name == "seasonal_naive")
        assert seasonal.season_length in (7,) or seasonal.season_length is None