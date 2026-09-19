from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthRoute:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModelInfoRoute:
    def test_model_info_returns_feature_counts(self):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_features"] > 0
        assert "threshold" in body


class TestPredictRoute:
    def test_predict_valid_order(self, sample_order):
        resp = client.post("/predict", json=sample_order)
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["probability_late"] <= 1.0
        assert "model_version" in body
        assert "request_id" in body

    def test_predict_rejects_invalid_payload_with_422(self, invalid_order):
        resp = client.post("/predict", json=invalid_order)
        assert resp.status_code == 422
        assert "failed_expectations" in resp.json()["detail"]

    def test_predict_rejects_malformed_json_with_422(self, sample_order):
        bad = dict(sample_order)
        del bad["order_purchase_timestamp"]  # required field missing
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_rejects_wrong_types(self, sample_order):
        bad = dict(sample_order)
        bad["n_items"] = "not-a-number"
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422


class TestBatchPredictRoute:
    def test_batch_predict(self, sample_order):
        second = dict(sample_order)
        second["seller_state"] = "RJ"
        resp = client.post("/predict/batch", json={"orders": [sample_order, second]})
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert len(preds) == 2

    def test_batch_predict_empty_list(self):
        resp = client.post("/predict/batch", json={"orders": []})
        assert resp.status_code == 200
        assert resp.json()["predictions"] == []


class TestMetricsRoute:
    def test_metrics_exposed_after_a_prediction(self, sample_order):
        client.post("/predict", json=sample_order)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "prediction_requests_total" in resp.text


class TestApiDocs:
    def test_openapi_schema_is_valid(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "/predict" in schema["paths"]
        assert "/predict/batch" in schema["paths"]

    def test_docs_page_loads(self):
        resp = client.get("/docs")
        assert resp.status_code == 200
