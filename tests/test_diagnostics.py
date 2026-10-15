"""Tests for residual diagnostics and naive-model intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation.diagnostics import residual_diagnostics
from app.models.naive import NaiveForecaster, SeasonalNaiveForecaster


def test_diagnostics_white_noise_passes_ljungbox():
    rng = np.random.default_rng(1)
    rep = residual_diagnostics(rng.normal(0, 1, 500))
    assert rep["n"] == 500
    assert rep["ljung_box"]["white_noise"] is True
    assert rep["jarque_bera"]["normal"] is True
    assert abs(rep["mean"]) < 0.2


def test_diagnostics_autocorrelated_fails_ljungbox():
    rng = np.random.default_rng(1)
    eps = rng.normal(0, 1, 500)
    resid = np.cumsum(eps)  # random walk increments? no — cumsum IS autocorrelated
    rep = residual_diagnostics(resid)
    assert rep["ljung_box"]["white_noise"] is False


def test_diagnostics_too_few_returns_note():
    rep = residual_diagnostics(np.array([1.0, 2.0, 3.0]))
    assert rep["ljung_box"] is None
    assert "note" in rep


def test_naive_intervals_present_and_widen():
    rng = np.random.default_rng(3)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 200)) + 100.0)
    res = NaiveForecaster().fit(s).predict(5)
    assert res.lower is not None and res.upper is not None
    widths = np.asarray(res.upper) - np.asarray(res.lower)
    assert bool(np.all(np.diff(widths) > 0))  # uncertainty grows with horizon
    assert bool(np.all(np.asarray(res.lower) <= res.forecast))
    assert bool(np.all(np.asarray(res.upper) >= res.forecast))


def test_seasonal_naive_intervals_present():
    s = pd.Series(np.tile(np.arange(7, dtype=float), 20))
    res = SeasonalNaiveForecaster(season_length=7).fit(s).predict(7)
    assert res.lower is not None and res.upper is not None
    assert len(res.forecast) == 7
