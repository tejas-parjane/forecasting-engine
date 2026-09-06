"""API routes for data upload, validation, and exploration."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.api.schemas import ExploreRequest, DatasetRequest, ValidateRequest
from app.api.store import store
from app.analysis.eda import EDAAnalyzer
from app.config import settings
from app.data.loader import infer_columns, load_csv
from app.data.validator import DataValidator

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    dataset_name: str | None = Form(None),
) -> dict[str, Any]:
    """Upload a CSV dataset and register it in memory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    try:
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="File contains no rows")

    name = dataset_name or file.filename.rsplit(".", 1)[0] or "dataset"
    store.put(name, df)

    dcol, tcol, numeric = infer_columns(df)
    return {
        "dataset_name": name,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "inferred_date_col": dcol,
        "inferred_target_col": tcol,
    }


@router.post("/validate")
async def validate_data(req: ValidateRequest) -> dict[str, Any]:
    """Validate a dataset and return a structured quality report."""
    df, name = _resolve_dataset(req)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    validator = DataValidator(date_col=req.date_col, target_col=req.target_col)
    report = validator.validate(df).to_dict()
    return {"dataset_name": name, "report": report}


@router.post("/explore")
async def explore_data(req: ExploreRequest) -> dict[str, Any]:
    """Run exploratory analysis on a dataset."""
    df, name = _resolve_dataset(req)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    from app.data.preprocessing import Preprocessor

    prepped = Preprocessor(date_col=req.date_col, target_col=req.target_col).preprocess(df)
    series = prepped.series
    analyzer = EDAAnalyzer()
    eda = analyzer.analyze(series, frequency=prepped.frequency).to_dict()
    return {"dataset_name": name, "eda": eda}


@router.get("/datasets")
async def list_datasets() -> dict[str, Any]:
    """List registered datasets."""
    names = store.list()
    return {"datasets": names, "count": len(names)}


def _resolve_dataset(req: DatasetRequest) -> tuple[pd.DataFrame | None, str]:
    """Resolve the dataset from an explicit upload, else the bundled sample."""
    name = req.dataset_name or "business_demand"

    df = store.get(name)
    if df is not None:
        return df, name

    if req.dataset_name:
        # Explicit name requested but not uploaded
        return None, name

    # Default: bundled sample data
    try:
        path = settings.resolve_sample_data()
        df = load_csv(path)
        return df, "business_demand"
    except Exception:
        return None, "business_demand"