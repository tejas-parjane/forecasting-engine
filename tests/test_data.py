"""Tests for data loading, validation, frequency detection, and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from app.data.preprocessing import Preprocessor
from app.data.sample_data import generate_business_demand
from app.data.validator import DataValidator, detect_frequency


def make_series(dates=None, values=None) -> pd.DataFrame:
    if dates is None:
        dates = pd.date_range("2022-01-01", periods=100, freq="D")
    if values is None:
        values = np.linspace(100, 200, len(dates)) + np.random.RandomState(0).normal(0, 5, len(dates))
    return pd.DataFrame({"date": dates, "target": values})


class TestFrequencyDetection:
    def test_daily(self):
        assert detect_frequency(pd.date_range("2022-01-01", periods=30, freq="D")) == "D"

    def test_weekly(self):
        assert detect_frequency(pd.date_range("2022-01-01", periods=30, freq="W")) == "W"

    def test_irregular(self):
        idx = pd.to_datetime(["2022-01-01", "2022-01-03", "2022-02-01"])
        assert detect_frequency(idx) == "IR"


class TestDataValidator:
    def test_valid_data(self):
        v = DataValidator()
        report = v.validate(make_series())
        assert report.is_valid
        assert report.quality_score == 100.0
        assert report.detected_frequency == "D"

    def test_missing_values_detected(self):
        df = make_series()
        df.loc[5, "target"] = np.nan
        report = DataValidator().validate(df)
        assert report.missing_values == 1
        assert report.quality_score < 100

    def test_duplicate_dates_detected(self):
        df = make_series()
        df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
        report = DataValidator().validate(df)
        assert report.duplicate_records >= 1

    def test_missing_date_column(self):
        df = make_series().rename(columns={"date": "foo"})
        report = DataValidator().validate(df)
        assert not report.is_valid

    def test_outliers_detected(self):
        df = make_series()
        df.loc[20, "target"] = 100000.0
        report = DataValidator().validate(df)
        assert report.outlier_count >= 1

    def test_negative_values_flag(self):
        df = make_series()
        df.loc[10, "target"] = -50.0
        report = DataValidator(allow_negative=False).validate(df)
        assert any(i.severity == "error" for i in report.issues)


class TestPreprocessor:
    def test_sorts_chronologically(self):
        df = make_series()
        shuffled = df.sample(frac=1, random_state=1)
        prepped = Preprocessor().preprocess(shuffled)
        assert (prepped.series.index == pd.to_datetime(df["date"])).all()

    def test_drops_duplicate_dates(self):
        df = make_series()
        df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
        prepped = Preprocessor().preprocess(df)
        assert prepped.series.index.is_unique

    def test_handles_non_numeric_target(self):
        df = make_series()
        df.loc[0:5, "target"] = "bad"
        prepped = Preprocessor().preprocess(df)
        assert pd.api.types.is_numeric_dtype(prepped.series)


class TestSampleData:
    def test_generates_expected_shape(self):
        df = generate_business_demand(n_days=365, seed=42)
        assert len(df) == 365
        assert list(df.columns) == ["date", "target"]

    def test_reproducible_with_seed(self):
        a = generate_business_demand(n_days=60, seed=7)
        b = generate_business_demand(n_days=60, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_contains_trend_and_variation(self):
        df = generate_business_demand(n_days=730, seed=42)
        first_half = df["target"].iloc[:365].mean()
        second_half = df["target"].iloc[365:].mean()
        assert second_half > first_half
        assert df["target"].std() > 0