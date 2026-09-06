"""Base classes and shared utilities for forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """A model's forecast output."""

    model_name: str
    horizon: int
    forecast: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    fitted: np.ndarray | None = None
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "method": self.method,
            "forecast": [float(x) for x in np.asarray(self.forecast).ravel()],
            "lower": [float(x) for x in np.asarray(self.lower).ravel()] if self.lower is not None else None,
            "upper": [float(x) for x in np.asarray(self.upper).ravel()] if self.upper is not None else None,
            "metadata": self.metadata,
        }


class BaseForecaster(ABC):
    """Abstract interface implemented by all forecasting models.

    Each model fits on a chronological series and produces a point
    forecast for `horizon` steps ahead, optionally with intervals.
    """

    name: str = "base"

    def __init__(self):
        self.fitted_: Any = None

    @abstractmethod
    def fit(self, series: pd.Series, **kwargs: Any) -> "BaseForecaster":
        """Fit the model on a chronological series (index = dates)."""

    @abstractmethod
    def predict(self, horizon: int, **kwargs: Any) -> ForecastResult:
        """Produce a forecast for `horizon` steps."""

    def name_(self) -> str:
        return self.name

    def requires_series(self) -> bool:
        return True
