"""Forecast dates must follow the detected frequency (daily behaviour unchanged)."""
import numpy as np
import pandas as pd
import pytest

from app.features.time_features import future_dates
from app.forecasting.pipeline import _future_dates
from app.models.xgboost_model import XGBoostForecaster


def test_daily_unchanged():
    assert _future_dates(pd.Timestamp("2024-03-10"), 3, "D") == ["2024-03-11", "2024-03-12", "2024-03-13"]


def test_weekly_steps_seven_days_from_last_observation():
    # Wednesday 2024-03-06: next points are Wednesdays, not a fixed Sunday.
    assert _future_dates(pd.Timestamp("2024-03-06"), 3, "W") == ["2024-03-13", "2024-03-20", "2024-03-27"]


def test_monthly_and_quarterly_period_starts():
    assert _future_dates(pd.Timestamp("2024-03-01"), 2, "MS") == ["2024-04-01", "2024-05-01"]
    assert _future_dates(pd.Timestamp("2024-01-01"), 2, "QS") == ["2024-04-01", "2024-07-01"]


def test_unknown_frequency_falls_back_to_daily():
    assert _future_dates(pd.Timestamp("2024-03-10"), 2, "IR") == ["2024-03-11", "2024-03-12"]


@pytest.mark.parametrize("freq,step", [("D", 1), ("W", 7)])
def test_xgboost_recursive_dates_follow_frequency(freq, step, monkeypatch):
    n = 160
    idx = pd.date_range("2022-01-05", periods=n, freq=f"{step}D")  # Wednesdays for weekly
    rng = np.random.default_rng(0)
    series = pd.Series(100 + np.arange(n) * 0.1 + rng.normal(0, 1, n), index=idx)
    seen = []
    import app.models.xgboost_model as m
    real = m.future_dates_for
    monkeypatch.setattr(m, "future_dates_for", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    model = XGBoostForecaster(n_estimators=20).fit(series, frequency=freq)
    res = model.predict(4)
    assert len(res.forecast) == 4
    d = seen[0]
    assert d[0] == idx[-1] + pd.Timedelta(days=step)
    assert (d[1:] - d[:-1] == pd.Timedelta(days=step)).all()
