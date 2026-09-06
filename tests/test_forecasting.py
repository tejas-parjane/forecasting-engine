"""Tests for the forecasting pipeline: horizon generation, intervals, end-to-end."""

import numpy as np
import pandas as pd
import pytest

from app.data.sample_data import generate_business_demand
from app.forecasting.pipeline import ForecastingPipeline, PipelineConfig, expected_change
from app.forecasting.uncertainty import gaussian_band, empirical_band, uncertainty_annotation


def df():
    return generate_business_demand(n_days=365, seed=1)


class TestExpectedChange:
    def test_increase(self):
        ec = expected_change(np.array([110.0, 115.0, 120.0]), 100.0)
        assert ec["direction"] == "increase"
        assert ec["absolute"] == pytest.approx(20.0)
        assert ec["pct_change"] == pytest.approx(20.0)

    def test_decrease(self):
        ec = expected_change(np.array([90.0, 80.0]), 100.0)
        assert ec["direction"] == "decrease"


class TestUncertainty:
    def test_gaussian_band_widens(self):
        fc = np.ones(10)
        band = gaussian_band(fc, 5.0, 10)
        width_start = band.upper[0] - band.lower[0]
        width_end = band.upper[-1] - band.lower[-1]
        assert width_end > width_start
        assert np.all(band.lower <= fc) and np.all(fc <= band.upper)

    def test_empirical_band_shape(self):
        rng = np.random.default_rng(3)
        resid = rng.normal(0, 6, 200)
        fc = np.zeros(7)
        band = empirical_band(fc, resid, 7)
        assert len(band.lower) == 7 and len(band.upper) == 7
        assert np.all(band.lower <= band.upper)

    def test_annotation(self):
        band = gaussian_band(np.ones(5), 4.0, 5)
        ann = uncertainty_annotation(band, np.ones(5))
        assert ann["widening_factor"] > 1
        assert "increases" in ann["note"]


class TestPipeline:
    def test_full_run(self):
        cfg = PipelineConfig(horizon=7, backtest_folds=2, initial_train_frac=0.6)
        result = ForecastingPipeline(config=cfg).run(df())
        assert len(result.forecast["forecast"]) == 7
        assert result.forecast["lower"] and result.forecast["upper"]
        assert len(result.forecast["lower"]) == 7
        assert result.backtest["best_model"] in (
            "naive", "seasonal_naive", "ets", "arima", "xgboost"
        )
        assert result.explainability["model"] == result.backtest["best_model"]
        assert result.explainability["drivers"]
        assert result.eda["n_observations"] > 0

    def test_interval_conserves_ordering(self):
        cfg = PipelineConfig(horizon=7, backtest_folds=2)
        result = ForecastingPipeline(config=cfg).run(df())
        lower = np.array(result.forecast["lower"])
        upper = np.array(result.forecast["upper"])
        fc_vals = np.array(result.forecast["forecast"])
        assert np.all(lower <= fc_vals) and np.all(fc_vals <= upper)

    def test_horizon_configuration_respected(self):
        cfg = PipelineConfig(horizon=90, backtest_folds=2)
        result = ForecastingPipeline(config=cfg).run(df())
        assert len(result.forecast["forecast"]) == 90

    def test_insufficient_data_raises(self):
        small = generate_business_demand(n_days=20, seed=1)
        cfg = PipelineConfig(horizon=7, backtest_folds=2)
        with pytest.raises(ValueError):
            ForecastingPipeline(config=cfg).run(small)

    def test_forecast_dates_are_future_and_continuous(self):
        cfg = PipelineConfig(horizon=14, backtest_folds=2)
        result = ForecastingPipeline(config=cfg).run(df())
        dates = pd.to_datetime(result.forecast["forecast_dates"])
        last_hist = result.eda["date_range"]["end"]
        assert all(date > pd.Timestamp(last_hist) for date in dates)
        assert (np.diff(dates.values.astype("datetime64[D]")) == np.timedelta64(1, "D")).all()