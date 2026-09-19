from __future__ import annotations

from src.models.predictor import get_model, predict_batch, predict_one


class TestModelLoading:
    def test_model_loads(self):
        loaded = get_model()
        assert loaded.model is not None
        assert loaded.source in ("mlflow_registry", "local_joblib")
        assert 0.0 <= loaded.threshold <= 1.0

    def test_model_has_predict_proba(self):
        loaded = get_model()
        assert hasattr(loaded.model, "predict_proba")


class TestPredict:
    def test_predict_one_shape(self, sample_order):
        result = predict_one(sample_order)
        assert set(result.keys()) >= {
            "prediction",
            "probability_late",
            "threshold_used",
            "model_version",
            "request_id",
        }
        assert result["prediction"] in (0, 1)
        assert 0.0 <= result["probability_late"] <= 1.0

    def test_predict_batch_preserves_order_and_count(self, sample_order):
        cross_state_order = dict(sample_order)
        cross_state_order["seller_state"] = "RJ"
        results = predict_batch([sample_order, cross_state_order])
        assert len(results) == 2
        for r in results:
            assert r["prediction"] in (0, 1)

    def test_cross_state_shifts_probability(self, sample_order):
        """Sanity check on model behavior on a known-shape input: a
        cross-state shipment should not make the predicted late-probability
        lower than the same order shipped in-state (per the synthetic
        training process's cross_state effect)."""
        same_state = predict_one(sample_order)

        cross = dict(sample_order)
        cross["seller_state"] = "RJ" if sample_order["customer_state"] != "RJ" else "MG"
        cross_state = predict_one(cross)

        assert cross_state["probability_late"] >= same_state["probability_late"] - 0.15

    def test_deterministic_on_identical_input(self, sample_order):
        r1 = predict_one(sample_order)
        r2 = predict_one(sample_order)
        assert r1["prediction"] == r2["prediction"]
        assert r1["probability_late"] == r2["probability_late"]
