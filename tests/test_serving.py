"""Tests for the FastAPI serving layer.

Unit tests mock the predictor to avoid loading BERT at test time.
Integration tests require real model artifacts (skipped in CI unless models/ is populated).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MODELS_DIR = Path(__file__).parent.parent / "models"
SERVING_READY = (MODELS_DIR / "xgboost_model.pkl").exists()


class TestServingEndpointsUnit(unittest.TestCase):
    """Unit tests — predictor is mocked, no BERT loading."""

    def setUp(self):
        mock_predictor = MagicMock()
        mock_predictor.is_loaded = True
        mock_predictor.predict.return_value = 91.5

        # Patch at the module level before importing app
        patcher = patch(
            "serving.predictor.CoffeePredictor", return_value=mock_predictor
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        from fastapi.testclient import TestClient

        from serving.app import app

        self.client = TestClient(app)

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_model_info_returns_xgboost(self):
        resp = self.client.get("/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_name"], "xgboost")
        self.assertAlmostEqual(data["r2_score"], 0.9453, places=2)
        self.assertEqual(data["n_features_selected"], 279)

    def test_predict_missing_desc_1_returns_422(self):
        resp = self.client.post("/predict", json={})
        self.assertEqual(resp.status_code, 422)

    def test_predict_valid_request_returns_rating(self):
        resp = self.client.post(
            "/predict",
            json={
                "desc_1": "bright citrus, clean finish",
                "desc_2": "lemon zest and floral notes",
                "desc_3": "exceptional clarity",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("rating", data)
        self.assertIsInstance(data["rating"], float)

    def test_predict_with_optional_sensory(self):
        resp = self.client.post(
            "/predict",
            json={
                "desc_1": "bold and chocolatey",
                "aroma": 9.0,
                "acid": 7.5,
            },
        )
        self.assertEqual(resp.status_code, 200)


@pytest.mark.skipif(not SERVING_READY, reason="model artifacts not trained yet")
@pytest.mark.integration
class TestServingIntegration(unittest.TestCase):
    """Integration tests — loads real model artifacts (slow, ~2 min startup)."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from serving.app import app

        cls.client = TestClient(app)

    def test_real_predict_in_range(self):
        resp = self.client.post(
            "/predict",
            json={
                "desc_1": "jasmine tea, tangerine, brown sugar syrup",
                "desc_2": "Ethiopian natural process, light roast",
                "desc_3": "delicate and complex, exceptional value",
            },
        )
        self.assertEqual(resp.status_code, 200)
        rating = resp.json()["rating"]
        self.assertGreaterEqual(rating, 80)
        self.assertLessEqual(rating, 100)


if __name__ == "__main__":
    unittest.main()
