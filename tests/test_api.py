"""API tests using the FastAPI TestClient."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data.sample_data import generate_business_demand
from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def seed_store():
    df = generate_business_demand(n_days=365, seed=99)
    from app.api.store import store

    store.put("test_demand", df)
    yield
    store.drop("test_demand")


class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["app"]


class TestDataRoutes:
    def test_validate_default(self):
        r = client.post("/data/validate", json={"date_col": "date", "target_col": "target"})
        assert r.status_code == 200
        body = r.json()
        assert body["report"]["n_observations"] > 0
        assert body["report"]["quality_score"] >= 0

    def test_validate_uploaded_dataset(self):
        r = client.post(
            "/data/validate",
            json={"dataset_name": "test_demand", "date_col": "date", "target_col": "target"},
        )
        assert r.status_code == 200
        assert r.json()["dataset_name"] == "test_demand"

    def test_validate_missing_dataset(self):
        r = client.post("/data/validate", json={"dataset_name": "nope", "date_col": "date"})
        assert r.status_code == 404

    def test_validate_bad_columns(self):
        r = client.post("/data/validate", json={"dataset_name": "nope"})
        assert r.status_code in (404, 422)

    def test_explore(self):
        r = client.post(
            "/data/explore",
            json={"dataset_name": "test_demand", "date_col": "date", "target_col": "target"},
        )
        assert r.status_code == 200
        eda = r.json()["eda"]
        assert "trend" in eda and "seasonality" in eda and "anomalies" in eda

    def test_upload_csv(self):
        df = generate_business_demand(n_days=30, seed=5)
        buf = df.to_csv(index=False).encode()
        r = client.post(
            "/data/upload",
            files={"file": ("mini.csv", buf, "text/csv")},
            data={"dataset_name": "mini"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["dataset_name"] == "mini"
        assert body["n_rows"] == 30


class TestModelRoutes:
    def test_list_models(self):
        r = client.get("/models")
        assert r.status_code == 200
        names = {m["name"] for m in r.json()["models"]}
        assert "naive" in names and "xgboost" in names

    def test_train(self):
        r = client.post(
            "/models/train",
            json={"dataset_name": "test_demand", "horizon": 7, "backtest_folds": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["models"]
        assert all(m["status"] == "ok" for m in body["models"])

    def test_evaluate(self):
        r = client.post(
            "/models/evaluate",
            json={"dataset_name": "test_demand", "horizon": 7, "backtest_folds": 2},
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert result["best_model"] is not None
        assert len(result["models"]) >= 4

    def test_forecast_full_pipeline(self):
        r = client.post(
            "/models/forecast",
            json={"dataset_name": "test_demand", "horizon": 7, "backtest_folds": 2},
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert len(result["forecast"]["forecast"]) == 7
        assert result["forecast"]["lower"] and result["forecast"]["upper"]
        assert "explainability" in result and "backtest" in result

    def test_forecast_horizon_validation(self):
        r = client.post(
            "/models/forecast",
            json={"dataset_name": "test_demand", "horizon": 0},
        )
        assert r.status_code == 422

    def test_insufficient_data(self):
        r = client.post(
            "/models/forecast",
            json={"dataset_name": "mini", "horizon": 7, "backtest_folds": 2},
        )
        assert r.status_code == 422


class TestAIRoutes:
    def test_recommend_rule_based(self):
        r = client.post(
            "/recommend",
            json={
                "dataset_name": "test_demand",
                "horizon": 7,
                "backtest_folds": 2,
                "use_llm": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["brief"]["forecast_summary"]
        assert body["brief"]["business_recommendation"]
        assert body["brief"]["generated_by"] == "rule-based"
        assert body["structured_input"]["forecast"]["direction"] in (
            "increase", "decrease", "flat"
        )

    def test_explain(self):
        r = client.post(
            "/explain",
            json={"dataset_name": "test_demand", "horizon": 7, "backtest_folds": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["best_model"]
        assert body["explainability"]["drivers"]