import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Set environment before importing app
os.environ["PROJECT_ROOT"] = os.getenv(
    "PROJECT_ROOT",
    "/Users/azeemkhalipha/mlops-retail-platform"
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/serving"))

# Mock the model so tests don't need the actual pickle file
mock_model = MagicMock()
mock_model.predict.return_value = np.array([12.5])

with patch("model_loader.ModelLoader.load", return_value=None):
    from app import app
    from model_loader import model_loader
    model_loader.model         = mock_model
    model_loader.model_version = "1"
    model_loader.is_loaded     = True


@pytest.fixture(scope="session")
def client():
    """
    Creates a test client that properly triggers FastAPI lifespan.
    scope="session" means one client is shared across all tests.
    """
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1"


def test_predict_endpoint(client):
    mock_model.predict.return_value = np.array([12.5])
    payload = {
        "qty_lag_1":          10.0,
        "qty_lag_7":          8.0,
        "qty_lag_30":         9.0,
        "qty_rolling_avg_7":  9.5,
        "qty_rolling_avg_30": 9.0,
        "qty_rolling_std_7":  1.2,
        "daily_revenue":      45.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_quantity" in data
    assert data["status"] == "success"
    assert isinstance(data["predicted_quantity"], float)


def test_batch_predict_endpoint(client):
    mock_model.predict.return_value = np.array([12.5, 8.3])
    payload = {
        "inputs": [
            {
                "qty_lag_1": 10.0, "qty_lag_7": 8.0,
                "qty_lag_30": 9.0, "qty_rolling_avg_7": 9.5,
                "qty_rolling_avg_30": 9.0, "qty_rolling_std_7": 1.2,
                "daily_revenue": 45.0
            },
            {
                "qty_lag_1": 5.0, "qty_lag_7": 4.0,
                "qty_lag_30": 6.0, "qty_rolling_avg_7": 5.0,
                "qty_rolling_avg_30": 5.5, "qty_rolling_std_7": 0.8,
                "daily_revenue": 20.0
            }
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_invalid_input_rejected(client):
    """Pydantic should reject non-numeric input with 422."""
    payload = {"qty_lag_1": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_field_rejected(client):
    """Pydantic should reject incomplete input with 422."""
    payload = {"qty_lag_1": 10.0}  # missing 6 required fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
