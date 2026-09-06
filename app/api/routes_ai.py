"""API routes for explainability and AI decision intelligence."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.routes_data import _resolve_dataset
from app.api.schemas import RecommendRequest

router = APIRouter(tags=["ai"])


@router.post("/recommend")
async def recommend(req: RecommendRequest) -> dict[str, Any]:
    """Generate a decision brief from a full forecast pipeline run.

    The AI layer interprets structured pipeline output. If no LLM API key is
    configured, a deterministic rule-based brief is returned instead.
    """
    from app.forecasting.pipeline import ForecastingPipeline, PipelineConfig

    df, name = _resolve_dataset(req)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    from app.data.preprocessing import Preprocessor

    prepped = Preprocessor(date_col=req.date_col, target_col=req.target_col).preprocess(df)
    if len(prepped.series) < 40:
        raise HTTPException(status_code=422, detail="At least 40 observations are required")

    config = PipelineConfig(
        date_col=req.date_col,
        target_col=req.target_col,
        horizon=req.horizon,
        backtest_folds=req.backtest_folds,
        selection_metric=req.selection_metric,
        initial_train_frac=req.initial_train_frac,
        include_xgboost=req.include_xgboost,
    )
    try:
        result = ForecastingPipeline(config=config).run(
            prepped.dataframe, frequency=prepped.frequency, horizon=req.horizon
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Build AI-layer structured input
    from app.ai.decision_engine import DecisionEngine
    from app.config import settings

    engine = DecisionEngine(
        api_key=settings.openai_api_key,
        model=settings.ai_model,
        temperature=settings.ai_temperature,
        max_tokens=settings.ai_max_tokens,
    )
    structured = engine.build_structured_input(
        forecast_payload=result.forecast,
        eda=result.eda,
        explainability=result.explainability,
        report=result.report,
        backtest=result.backtest,
    )
    try:
        # LLM is used only when explicitly requested AND the application-level
        # AI gate is on (AI_ENABLED=true + key). Otherwise always rule-based.
        use_llm = req.use_llm if req.use_llm is not None else settings.ai_enabled
        brief = engine.generate(structured, use_llm=use_llm)
    except Exception as exc:
        # Graceful degradation: if the LLM call fails, fall back to the
        # deterministic rule-based brief rather than failing the request.
        brief = engine.generate(structured, use_llm=False)
        structured["llm_error"] = str(exc)

    return {
        "dataset_name": name,
        "brief": brief.to_dict(),
        "structured_input": structured,
    }


@router.post("/explain")
async def explain_forecast(req: RecommendRequest) -> dict[str, Any]:
    """Explain the forecast for a dataset (drivers + feature importance)."""
    from app.forecasting.pipeline import ForecastingPipeline, PipelineConfig

    df, name = _resolve_dataset(req)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    from app.data.preprocessing import Preprocessor

    prepped = Preprocessor(date_col=req.date_col, target_col=req.target_col).preprocess(df)
    if len(prepped.series) < 40:
        raise HTTPException(status_code=422, detail="At least 40 observations are required")

    config = PipelineConfig(
        date_col=req.date_col,
        target_col=req.target_col,
        horizon=req.horizon,
        backtest_folds=req.backtest_folds,
        selection_metric=req.selection_metric,
        initial_train_frac=req.initial_train_frac,
        include_xgboost=req.include_xgboost,
    )
    try:
        result = ForecastingPipeline(config=config).run(
            prepped.dataframe, frequency=prepped.frequency, horizon=req.horizon
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "dataset_name": name,
        "explainability": result.explainability,
        "best_model": result.forecast["model"],
    }