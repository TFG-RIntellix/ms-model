"""Integration tests for API endpoints.

Tests the full request/response flow through FastAPI using
shared fixtures from ``conftest.py``.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
import numpy as np

from app.main import app


# Note: client, sample_loan_payload are provided by conftest.py


# ==================== Health Check Tests ====================

def test_health_check(client: TestClient):
    """Health endpoint returns 200 with service information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_root_endpoint(client: TestClient):
    """Root endpoint returns API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


# ==================== Validation Tests ====================

def test_predict_loan_missing_field(client: TestClient):
    """Missing required fields return 422 Unprocessable Entity."""
    payload = {
        "edad": 35,
        "genero": "Mujer",
        # Missing other required fields
    }
    response = client.post("/api/v1/risk/predict-loan", json=payload)
    assert response.status_code == 422


def test_predict_loan_invalid_genero(client: TestClient, sample_loan_payload: dict):
    """Invalid enum value for genero returns 422."""
    sample_loan_payload["genero"] = "InvalidValue"
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 422


def test_predict_loan_edad_out_of_range(client: TestClient, sample_loan_payload: dict):
    """Age below 18 or above 80 returns 422."""
    sample_loan_payload["edad"] = 150
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 422

    sample_loan_payload["edad"] = 10
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 422


def test_predict_loan_negative_income(client: TestClient, sample_loan_payload: dict):
    """Negative annual income returns 422."""
    sample_loan_payload["ingresos_anuales"] = -1000
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 422


def test_predict_loan_dti_out_of_range(client: TestClient, sample_loan_payload: dict):
    """DTI ratio above 1 returns 422."""
    sample_loan_payload["dti"] = 1.5
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 422


# ==================== Endpoint Response Tests ====================

def test_predict_loan_valid_request(client: TestClient, sample_loan_payload: dict):
    """Valid loan payload returns 200 with correct response structure."""
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 200

    data = response.json()
    assert "probability_of_default" in data
    assert "risk_segment" in data
    assert "shap_explanations" in data

    assert 0 <= data["probability_of_default"] <= 1
    assert data["risk_segment"] in ["Low", "Medium", "High"]

    assert len(data["shap_explanations"]) > 0
    for explanation in data["shap_explanations"]:
        assert "feature" in explanation
        assert "impact" in explanation
        assert "direction" in explanation
        assert explanation["direction"] in ["increase", "decrease"]


def test_predict_loan_response_schema(client: TestClient, sample_loan_payload: dict):
    """Response fields have correct Python types."""
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["probability_of_default"], (int, float))
    assert isinstance(data["risk_segment"], str)
    assert isinstance(data["shap_explanations"], list)

    for exp in data["shap_explanations"]:
        assert isinstance(exp["feature"], str)
        assert isinstance(exp["impact"], (int, float))
        assert isinstance(exp["direction"], str)


def test_predict_loan_top_5_features(client: TestClient, sample_loan_payload: dict):
    """SHAP explanations contain at most 5 features."""
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 200
    assert len(response.json()["shap_explanations"]) <= 5


def test_predict_loan_features_sorted_by_impact(client: TestClient, sample_loan_payload: dict):
    """SHAP features are sorted by descending absolute impact."""
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert response.status_code == 200

    impacts = [exp["impact"] for exp in response.json()["shap_explanations"]]
    assert impacts == sorted(impacts, reverse=True)


# ==================== Credit Card Endpoint Tests ====================

def test_predict_credit_card_placeholder(client: TestClient):
    """Credit card placeholder returns 501 Not Implemented."""
    payload = {
        "edad": 35,
        "genero": "Mujer",
        "ingresos_anuales": 45000.0,
    }
    response = client.post("/api/v1/risk/predict-credit-card", json=payload)
    assert response.status_code == 501


# ==================== Error Handling Tests ====================

def test_invalid_endpoint(client: TestClient):
    """Unknown endpoints return 404."""
    response = client.get("/api/v1/invalid")
    assert response.status_code == 404


def test_method_not_allowed(client: TestClient):
    """GET on a POST-only endpoint returns 405."""
    response = client.get("/api/v1/risk/predict-loan")
    assert response.status_code == 405


def test_response_contains_request_id_header(client: TestClient, sample_loan_payload: dict):
    """Each response echoes or creates an X-Request-ID header."""
    response = client.post("/api/v1/risk/predict-loan", json=sample_loan_payload)
    assert "x-request-id" in response.headers


# ==================== Model Info Endpoint Tests ====================

def test_model_info_returns_200(client: TestClient):
    """Model-info endpoint returns 200 with expected fields."""
    response = client.get("/api/v1/risk/model-info")
    assert response.status_code == 200

    data = response.json()
    assert "app_version" in data
    assert "model_loaded" in data
    assert "feature_names" in data
    assert "scaler_loaded" in data
    assert "encoder_loaded" in data
    assert "risk_thresholds" in data


def test_model_info_features_count(client: TestClient):
    """Model-info returns exactly 19 feature names."""
    response = client.get("/api/v1/risk/model-info")
    assert response.status_code == 200
    assert len(response.json()["feature_names"]) == 19


def test_model_info_risk_thresholds(client: TestClient):
    """Model-info returns valid risk threshold values."""
    response = client.get("/api/v1/risk/model-info")
    assert response.status_code == 200

    thresholds = response.json()["risk_thresholds"]
    assert "low_below" in thresholds
    assert "high_at_or_above" in thresholds
    assert thresholds["low_below"] < thresholds["high_at_or_above"]


# ==================== Error Body Request-ID Tests ====================

def test_validation_error_contains_request_id(client: TestClient):
    """422 error responses include a request_id in the body."""
    payload = {"edad": 999}  # invalid — triggers validation error
    response = client.post("/api/v1/risk/predict-loan", json=payload)
    assert response.status_code == 422

    data = response.json()
    assert "request_id" in data
