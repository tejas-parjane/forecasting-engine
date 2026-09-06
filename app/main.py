"""FastAPI application entry point."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_ai import router as ai_router
from app.api.routes_data import router as data_router
from app.api.routes_models import router as models_router
from app.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("forecasting-engine")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-Powered Forecasting & Decision Intelligence Engine. "
        "Upload time-series data, run validated forecasting with backtesting, "
        "uncertainty, explainability, and AI-assisted decision briefs."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(models_router)
app.include_router(ai_router)


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "ai_enabled": settings.ai_enabled,
    }


@app.on_event("startup")
async def _ensure_sample_data() -> None:
    """Generate the bundled sample dataset on first startup if missing."""
    try:
        path = settings.resolve_sample_data()
        if not Path(path).exists():
            from app.data.sample_data import save_sample_data

            save_sample_data(path, seed=settings.sample_data_seed)
            logger.info("Generated sample dataset at %s", path)
        else:
            logger.info("Sample dataset found at %s", path)
    except Exception as exc:
        logger.warning("Could not prepare sample data: %s", exc)