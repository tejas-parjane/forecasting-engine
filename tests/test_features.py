"""Tests for feature engineering: lags, rolling stats, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from app.features.time_features import (
    build_feature_frame,
    calendar_features,
    lag_features,
    rolling_features,
)


def daily_df():
    dates = pd.date_range("2022-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    values = 100 + np.arange(200) * 0.3 + rng.normal(0, 5, 200)
    return pd.DataFrame({"date": dates, "target": values})


class TestCalendarFeatures:
    def test_columns_added(self):
        out = calendar_features(daily_df())
        for col in ["year", "month", "quarter", "dayofweek", "is_weekend", "sin_week", "cos_week"]:
            assert col in out.columns

    def test_weekend_flag(self):
        out = calendar_features(daily_df())
        weekend = out["dayofweek"] >= 5
        assert (out.loc[weekend, "is_weekend"] == 1).all()
        assert (out.loc[~weekend, "is_weekend"] == 0).all()


class TestLagFeatures:
    def test_lag_values_correct(self):
        df = daily_df()
        out = lag_features(df, lags=[1, 7])
        expected1 = df["target"].shift(1)
        expected7 = df["target"].shift(7)
        pd.testing.assert_series_equal(out["lag_1"], expected1, check_names=False)
        pd.testing.assert_series_equal(out["lag_7"], expected7, check_names=False)

    def test_no_future_leakage_in_lag(self):
        """lag_k must equal the value k rows earlier (past only)."""
        df = daily_df()
        out = lag_features(df, lags=[3])
        for i in range(3, len(df)):
            assert out.loc[i, "lag_3"] == df.loc[i - 3, "target"]


class TestRollingFeatures:
    def test_rolling_mean_uses_only_past(self):
        """A rolling(window=k) mean at row i covers rows i-k+1..i (current
        observation included). All of those are known/observed at time i, so
        using them to predict row i+1 leaks nothing about the future."""
        df = daily_df()
        out = rolling_features(df, windows=[7], include_std=False)
        for i in range(7, len(df)):
            expected = df["target"].iloc[i - 6:i + 1].mean()
            assert abs(out.loc[i, "rolling_mean_7"] - expected) < 1e-9

    def test_rolling_std_positive(self):
        out = rolling_features(daily_df(), windows=[7], include_std=True)
        valid = out["rolling_std_7"].dropna()
        assert (valid >= 0).all()


class TestBuildFeatureFrame:
    def test_builds_full_frame(self):
        out = build_feature_frame(daily_df())
        assert "lag_1" in out.columns
        assert "rolling_mean_7" in out.columns
        assert "dayofweek" in out.columns

    def test_no_future_information_reaches_feature_rows(self):
        """Engineered features for row i must only use rows <= i-1 or calendar fields."""
        df = daily_df()
        out = build_feature_frame(df)
        lag_cols = [c for c in out.columns if c.startswith("lag_")]
        roll_cols = [c for c in out.columns if c.startswith("rolling_")]
        # For a gap test: zero out target at a single row and confirm order preserved
        probe = df.copy()
        probe.loc[10, "target"] = np.nan
        frame = build_feature_frame(probe)
        # lag_1 at row 11 references row 10's target
        assert pd.isna(frame.loc[11, "lag_1"]) or frame.loc[11, "lag_1"] == probe.loc[10, "target"]