"""API routes for model training, evaluation, and forecasting."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.routes_data import _resolve_dataset
from app.api.schemas import ForecastRequest, ModelConfig, TrainRequest
from app.data.validator import detect_frequency

router = APIRouter(prefix="/models", tags=["models"])


def _prepare(config: ModelConfig):
    df, name = _resolve_dataset(config)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    from app.data.preprocessing import Preprocessor

    prepped = Preprocessor(date_col=config.date_col, target_col=config.target_col).preprocess(df)
    return prepped, name


@router.post("/train")
async def train_models(req: TrainRequest) -> dict[str, Any]:
    """Train all candidate models and return a summary."""
    prepped, name = _prepare(req)
    series = prepped.series
    freq = prepped.frequency

    if len(series) < 40:
        raise HTTPException(status_code=422, detail="At least 40 observations are required for training")

    from app.models.registry import create_models

    models = create_models(series, frequency=freq, with_xgboost=req.include_xgboost)
    model_info = []
    for m in models:
        try:
            m.fit(series, frequency=freq)
            train_mae = _quick_train_mae(m, series, freq)
            model_info.append(
                {
                    "name": m.name,
                    "class": m.__class__.__name__,
                    "status": "ok",
                    "training_mae": train_mae,
                }
            )
        except Exception as exc:
            model_info.append({"name": m.name, "class": m.__class__.__name__, "status": "error", "error": str(exc)})

    return {
        "dataset_name": name,
        "frequency": freq,
        "n_observations": int(len(series)),
        "models": model_info,
        "note": "Full performance comparison is done in /models/evaluate via time-series backtesting.",
    }


@router.post("/evaluate")
async def evaluate_models(req: TrainRequest) -> dict[str, Any]:
    """Run time-series backtesting over all candidate models."""
    prepped, name = _prepare(req)
    series = prepped.series
    freq = prepped.frequency

    from app.evaluation.backtesting import Backtester
    from app.models.registry import create_models

    if len(series) < 40:
        raise HTTPException(status_code=422, detail="At least 40 observations are required for evaluation")

    backtester = Backtester(
        n_folds=req.backtest_folds,
        selection_metric=req.selection_metric,
        initial_train_frac=req.initial_train_frac,
    )
    models = create_models(series, frequency=freq, with_xgboost=req.include_xgboost)
    try:
        result = backtester.backtest(series, models=models, frequency=freq)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"dataset_name": name, "result": result.to_dict()}


@router.post("/forecast")
async def forecast(req: ForecastRequest) -> dict[str, Any]:
    """Full pipeline: validate, explore, backtest, select best model, forecast."""
    prepped, name = _prepare(req)
    series = prepped.series
    freq = prepped.frequency

    from app.forecasting.pipeline import ForecastingPipeline, PipelineConfig

    if len(series) < 40:
        raise HTTPException(status_code=422, detail="At least 40 observations are required for forecasting")

    config = PipelineConfig(
        date_col=req.date_col,
        target_col=req.target_col,
        horizon=req.horizon,
        backtest_folds=req.backtest_folds,
        selection_metric=req.selection_metric,
        initial_train_frac=req.initial_train_frac,
        include_xgboost=req.include_xgboost,
    )
    pipeline = ForecastingPipeline(config=config)
    try:
        result = pipeline.run(prepped.dataframe, frequency=freq, horizon=req.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"dataset_name": name, "result": result.to_dict()}


@router.get("")
@router.get("/")
async def list_registered_models() -> dict[str, Any]:
    """List the available model families."""
    return {
        "models": [
            {
                "name": "naive",
                "family": "baseline",
                "description": "Forecast equals the most recent observed value.",
            },
            {
                "name": "seasonal_naive",
                "family": "baseline",
                "description": "Forecast equals the value from one season back.",
            },
            {
                "name": "ets",
                "family": "statistical",
                "description": "Exponential smoothing (Holt-Winters) with automatic seasonality.",
            },
            {
                "name": "arima",
                "family": "statistical",
                "description": "ARIMA/SARIMA with automatic order selection via AIC.",
            },
            {
                "name": "xgboost",
                "family": "machine-learning",
                "description": "Gradient boosting on lag, rolling, and calendar features.",
            },
        ]
    }


def _quick_train_mae(model, series, freq: str) -> float | None:
    """Compute a quick within-sample MAE for the train summary."""
    try:
        h = min(7, max(1, int(len(series) * 0.1)))
        train = series.iloc[: -h]
        actual = series.iloc[-h:].values
        fit = model.fit(train, frequency=freq)
        result = fit.predict(h)
        from app.evaluation.metrics import mae

        return float(mae(actual, result.forecast[:h]))
    except Exception:
        return None