"""Preprocessing utilities for time-series data.

Prepares a validated dataset into a clean chronological series
ready for feature engineering and modeling. Any transformation
performed here is recorded and returned for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PreprocessedData:
    """Container for preprocessed series and metadata."""

    series: pd.Series  # DatetimeIndex-indexed target series
    dataframe: pd.DataFrame  # original dataframe with parsed/sorted date
    date_col: str
    target_col: str
    frequency: str
    transformations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_col": self.date_col,
            "target_col": self.target_col,
            "frequency": self.frequency,
            "n_observations": len(self.series),
            "start_date": str(self.series.index.min().date()),
            "end_date": str(self.series.index.max().date()),
            "transformations": self.transformations,
            "target_summary": {
                "mean": float(self.series.mean()),
                "std": float(self.series.std()),
                "min": float(self.series.min()),
                "max": float(self.series.max()),
                "recent_value": float(self.series.iloc[-1]),
            },
        }


class Preprocessor:
    """Cleans and structures a time-series dataset for modeling."""

    def __init__(self, date_col: str = "date", target_col: str = "target", dropna_target: bool = True):
        self.date_col = date_col
        self.target_col = target_col
        self.dropna_target = dropna_target

    def preprocess(self, df: pd.DataFrame, frequency: str | None = None) -> PreprocessedData:
        """Return a clean, sorted time series indexed by date."""
        transformations: list[str] = []
        work = df.copy()

        if self.date_col not in work.columns or self.target_col not in work.columns:
            raise ValueError("Required columns missing for preprocessing.")

        work[self.date_col] = pd.to_datetime(work[self.date_col], errors="coerce")
        work = work[work[self.date_col].notna()].copy()
        transformations.append("Dropped rows with invalid dates")

        orig_n = len(work)

        # Drop duplicate dates keeping the last occurrence
        work = work.sort_values(self.date_col)
        work = work[~work[self.date_col].duplicated(keep="last")].copy()
        if len(work) < orig_n:
            transformations.append(f"Dropped {orig_n - len(work)} duplicate date records (kept last)")

        # Numeric target
        work[self.target_col] = pd.to_numeric(work[self.target_col], errors="coerce")
        if self.dropna_target:
            before = len(work)
            work = work[work[self.target_col].notna()].copy()
            if len(work) < before:
                transformations.append(f"Dropped {before - len(work)} rows with missing target values")

        work = work.sort_values(self.date_col).reset_index(drop=True)

        series = pd.Series(work[self.target_col].values, index=pd.DatetimeIndex(work[self.date_col]))

        # Ensure strictly increasing monotonic index
        series = series[~series.index.duplicated(keep="last")]
        series = series.sort_index()
        series.index = series.index.tz_localize(None) if series.index.tz is not None else series.index

        freq = frequency or self._infer_frequency(series.index)

        return PreprocessedData(
            series=series,
            dataframe=work,
            date_col=self.date_col,
            target_col=self.target_col,
            frequency=freq,
            transformations=transformations,
        )

    @staticmethod
    def _infer_frequency(index: pd.DatetimeIndex) -> str:
        from app.data.validator import detect_frequency

        return detect_frequency(index)
