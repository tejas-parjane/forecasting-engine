"""Tests for evaluation metrics and time-series backtesting."""

import numpy as np
import pandas as pd
import pytest

from app.evaluation.backtesting import Backtester, size_guess
from app.evaluation.metrics import (
    mae,
    mape,
    compute_metrics,
    rmse,
    smape,
    wape,
)


class TestMetrics:
    def test_mae(self):
        a = np.array([1.0, 2.0, 3.0])
        f = np.array([2.0, 2.0, 2.0])
        assert mae(a, f) == pytest.approx(2.0 / 3.0)

    def test_rmse(self):
        a = np.array([0.0, 0.0, 0.0])
        f = np.array([3.0, 4.0, 0.0])
        assert rmse(a, f) == pytest.approx(np.sqrt((9 + 16) / 3.0))

    def test_mape(self):
        a = np.array([100.0, 200.0])
        f = np.array([110.0, 180.0])
        assert mape(a, f) == pytest.approx(10.0)  # (10% + 10%) / 2

    def test_mape_zero_denominator_returns_none(self):
        a = np.array([0.0, 10.0])
        f = np.array([1.0, 10.0])
        assert mape(a, f) is None

    def test_smape(self):
        a = np.array([100.0, 200.0])
        f = np.array([110.0, 180.0])
        val = smape(a, f)
        assert val > 0 and val < 100

    def test_wape(self):
        a = np.array([100.0, 200.0])
        f = np.array([110.0, 180.0])
        # |100-110| + |200-180| = 30; sum actuals = 300
        assert wape(a, f) == pytest.approx(10.0)

    def test_compute_metrics_notes(self):
        a = np.array([100.0, 200.0, 300.0])
        f = np.array([110.0, 190.0, 290.0])
        res = compute_metrics(a, f)
        for key in ["mae", "rmse", "mape", "smape", "wape"]:
            assert key in res
            assert res[key] is not None
        assert "notes" in res

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_metrics(np.array([1.0]), np.array([1.0, 2.0]))


class TestBacktester:
    def _series(self):
        rng = np.random.default_rng(7)
        n = 150
        idx = pd.date_range("2022-01-01", periods=n, freq="D")
        values = 100 + 0.3 * np.arange(n) + rng.normal(0, 4, n)
        return pd.Series(values, index=idx)

    def test_backtest_runs_and_ranks(self):
        from app.models.naive import NaiveForecaster, SeasonalNaiveForecaster

        models = [NaiveForecaster(), SeasonalNaiveForecaster(season_length=7)]
        bt = Backtester(n_folds=2, selection_metric="smape", initial_train_frac=0.6).backtest(
            self._series(), models=models, frequency="D"
        )
        assert len(bt.models) == 2
        assert bt.best_model is not None
        ranks = {m.model_name: m.rank for m in bt.models if m.rank is not None}
        assert len(ranks) == 2
        best_rank_name = min(ranks, key=ranks.get)
        assert best_rank_name == bt.best_model

    def test_metrics_present_in_folds(self):
        from app.models.naive import NaiveForecaster

        bt = Backtester(n_folds=2, initial_train_frac=0.6).backtest(
            self._series(), models=[NaiveForecaster()], frequency="D"
        )
        m = bt.models[0]
        assert len(m.scores_per_fold) == 2
        for f in m.scores_per_fold:
            assert "smape" in f["metrics"]
            assert len(f["forecast"]) == len(f["actual"])

    def test_best_model_by_smape_is_lowest(self):
        from app.models.base import BaseForecaster, ForecastResult

        class GoodForecaster(BaseForecaster):
            name = "good_model"

            def fit(self, series, **kwargs):
                self.base = float(series.median())
                return self

            def predict(self, horizon, **kwargs):
                return ForecastResult(
                    model_name=self.name, horizon=horizon,
                    forecast=np.full(horizon, self.base),
                )

        class BadForecaster(BaseForecaster):
            name = "bad_model"

            def fit(self, series, **kwargs):
                self.base = float(series.min()) * 3
                return self

            def predict(self, horizon, **kwargs):
                return ForecastResult(
                    model_name=self.name, horizon=horizon,
                    forecast=np.full(horizon, self.base),
                )

        bt = Backtester(n_folds=2, initial_train_frac=0.6).backtest(
            self._series(), models=[GoodForecaster(), BadForecaster()], frequency="D"
        )
        good = next(m for m in bt.models if m.model_name == "good_model")
        bad = next(m for m in bt.models if m.model_name == "bad_model")
        assert good.aggregated["smape"] < bad.aggregated["smape"]
        assert good.rank == 1

    def test_insufficient_data_raises(self):
        s = pd.Series(np.arange(10), index=pd.date_range("2022-01-01", periods=10, freq="D"))
        from app.models.naive import NaiveForecaster

        with pytest.raises(ValueError):
            Backtester(n_folds=3, initial_train_frac=0.6).backtest(
                s, models=[NaiveForecaster()], frequency="D"
            )


class TestSizeGuess:
    def test_sane_default(self):
        assert size_guess(100, 4, "D") >= 7