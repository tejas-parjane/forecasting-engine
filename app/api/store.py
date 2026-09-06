"""In-memory dataset store for the API.

Datasets are uploaded, validated, and held in memory keyed by name.
This keeps the demo stateless and simple while still supporting
multiple datasets in a session.
"""

from __future__ import annotations

import threading
from typing import Any

import pandas as pd


class DatasetStore:
    """Thread-safe in-memory store of uploaded datasets."""

    def __init__(self):
        self._lock = threading.Lock()
        self._datasets: dict[str, pd.DataFrame] = {}

    def put(self, name: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._datasets[name] = df.copy()

    def get(self, name: str) -> pd.DataFrame | None:
        with self._lock:
            df = self._datasets.get(name)
            return df.copy() if df is not None else None

    def list(self) -> list[str]:
        with self._lock:
            return list(self._datasets.keys())

    def drop(self, name: str) -> bool:
        with self._lock:
            if name in self._datasets:
                del self._datasets[name]
                return True
            return False


store = DatasetStore()