"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str


class DatasetRequest(BaseModel):
    dataset_name: str | None = Field(None, description="Name of an uploaded dataset. Defaults to sample data.")
    date_col: str = Field("date", description="Date column name")
    target_col: str = Field("target", description="Target column name")


class ValidateRequest(DatasetRequest):
    pass


class ExploreRequest(DatasetRequest):
    pass


class ModelConfig(DatasetRequest):
    horizon: int = Field(30, ge=1, le=365, description="Forecast horizon in periods")
    backtest_folds: int = Field(4, ge=1, le=10)
    selection_metric: Literal["mae", "rmse", "mape", "smape", "wape"] = Field("smape")
    initial_train_frac: float = Field(0.6, gt=0.1, lt=0.95)
    include_xgboost: bool = True


class TrainRequest(ModelConfig):
    pass


class ForecastRequest(ModelConfig):
    pass


class RecommendRequest(ModelConfig):
    use_llm: bool | None = Field(None, description="Force LLM on/off. Defaults to configured availability.")


class RecommendResponse(BaseModel):
    brief: dict[str, Any]
    structured_input: dict[str, Any]