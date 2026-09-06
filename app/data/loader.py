"""Data loading utilities for time-series datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.exceptions import DataLoadError


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.
        **kwargs: Additional arguments passed to pandas.read_csv.

    Returns:
        Loaded DataFrame.

    Raises:
        DataLoadError: If the file cannot be read.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError as exc:
        raise DataLoadError(f"File not found: {path}") from exc
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV {path}: {exc}") from exc


def load_dataframe(data: pd.DataFrame | dict | str | Path | None = None) -> pd.DataFrame | None:
    """Load data from a variety of sources (used by the API).

    Accepts an existing DataFrame, a dict convertible to a DataFrame, or a file path.

    Returns None when no data is provided.
    """
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if isinstance(data, (str, Path)):
        return load_csv(data)
    raise DataLoadError(f"Unsupported data source type: {type(data).__name__}")


def infer_columns(df: pd.DataFrame, date_col: str | None = None, target_col: str | None = None) -> tuple[str, str, list[str]]:
    """Infer the date and target columns if not specified.

    Returns:
        (date_col, target_col, candidate_target_cols)
    """
    date_like = [c for c in df.columns if _looks_like_date(df, c)]
    numeric_like = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    dcol = date_col or (date_like[0] if date_like else None)
    tcol = target_col or (numeric_like[0] if numeric_like else None)

    return dcol, tcol, numeric_like


def _looks_like_date(df: pd.DataFrame, col: str, sample: int = 50) -> bool:
    if col in ("date", "Date", "DATE", "timestamp", "Timestamp", "time", "Time", "ds", "d"):
        return True
    if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
        return False
    try:
        pd.to_datetime(df[col].head(sample), errors="raise")
        return True
    except Exception:
        return False
