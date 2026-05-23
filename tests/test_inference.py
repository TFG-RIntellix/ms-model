"""Unit tests for the inference service.

Tests encoding correctness, prediction logic, and SHAP value extraction.
Shared fixtures (sample_loan_request, mock_model_manager) come from
``conftest.py``.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from app.schemas.enums import (
    GenderEnum, MaritalStatusEnum, EducationEnum, EmploymentStatusEnum,
    OccupationSectorEnum, HomeOwnershipEnum, HasMortgageEnum, LoanTypeEnum,
    LoanPurposeEnum,
)
from app.services.encoder import CategoricalEncoder
from app.services.inference import InferenceService


# ==================== Encoder Tests ====================

class TestCategoricalEncoder:
    """Tests for ``CategoricalEncoder`` mapping correctness."""

    def test_encode_gender(self):
        """Gender map has correct integer codes."""
        assert CategoricalEncoder.GENDER_MAP[GenderEnum.FEMALE] == 0
        assert CategoricalEncoder.GENDER_MAP[GenderEnum.MALE] == 1
        assert CategoricalEncoder.GENDER_MAP[GenderEnum.OTHER] == 2

    def test_encode_marital_status(self):
        """Marital status map has correct integer codes."""
        assert CategoricalEncoder.MARITAL_STATUS_MAP[MaritalStatusEnum.SINGLE] == 0
        assert CategoricalEncoder.MARITAL_STATUS_MAP[MaritalStatusEnum.MARRIED] == 1
        assert CategoricalEncoder.MARITAL_STATUS_MAP[MaritalStatusEnum.DIVORCED] == 2
        assert CategoricalEncoder.MARITAL_STATUS_MAP[MaritalStatusEnum.WIDOWED] == 3

    def test_encode_education(self):
        """Education level map has correct ordinal codes."""
        assert CategoricalEncoder.EDUCATION_MAP[EducationEnum.NO_STUDIES] == 0
        assert CategoricalEncoder.EDUCATION_MAP[EducationEnum.POSTGRADUATE] == 6

    def test_encode_request_shape(self, sample_loan_request):
        """Encoded request produces shape (1, 19) float32 array."""
        X = CategoricalEncoder.encode_request(sample_loan_request.model_dump())
        assert X.shape == (1, 19), f"Expected shape (1, 19), got {X.shape}"
        assert X.dtype == np.float32

    def test_encode_request_values(self, sample_loan_request):
        """Specific categorical values are encoded to their expected integers."""
        X = CategoricalEncoder.encode_request(sample_loan_request.model_dump())
        encoded_row = X[0]
        assert encoded_row[1] == 0  # gender: Mujer → 0
        assert encoded_row[2] == 1  # maritalStatus: Casado → 1
        assert encoded_row[3] == 5  # education: Grado → 5

    def test_feature_order(self):
        """Feature list contains 19 features starting with age, ending with previousDefaultsCount."""
        features = CategoricalEncoder.get_feature_names()
        assert len(features) == 19
        assert features[0] == "age"
        assert features[-1] == "previousDefaultsCount"


# ==================== InferenceService Unit Tests ====================

class TestInferenceService:
    """Tests for ``InferenceService`` business logic."""

    def test_risk_segment_low(self, mock_model_manager):
        """PD < 0.15 maps to 'Low'."""
        service = InferenceService(mock_model_manager)
        assert service._get_risk_segment(0.10) == "Low"
        assert service._get_risk_segment(0.14999) == "Low"

    def test_risk_segment_medium(self, mock_model_manager):
        """0.15 ≤ PD < 0.35 maps to 'Medium'."""
        service = InferenceService(mock_model_manager)
        assert service._get_risk_segment(0.15) == "Medium"
        assert service._get_risk_segment(0.25) == "Medium"
        assert service._get_risk_segment(0.34999) == "Medium"

    def test_risk_segment_high(self, mock_model_manager):
        """PD ≥ 0.35 maps to 'High'."""
        service = InferenceService(mock_model_manager)
        assert service._get_risk_segment(0.35) == "High"
        assert service._get_risk_segment(0.50) == "High"
        assert service._get_risk_segment(0.95) == "High"

    def test_extract_top_features(self, mock_model_manager):
        """Top-5 SHAP features are sorted by descending absolute value."""
        service = InferenceService(mock_model_manager)
        shap_values = np.array([[
            0.12, -0.08, 0.06, -0.04, 0.03,
            0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01,
        ]])
        explanations = service._extract_top_features(shap_values, base_value=0.2)

        assert len(explanations) == 5
        assert explanations[0].feature == "age"
        assert explanations[0].impact == pytest.approx(0.12)
        assert explanations[0].direction == "increase"

        assert explanations[1].feature == "gender"
        assert explanations[1].impact == pytest.approx(0.08)
        assert explanations[1].direction == "decrease"

    def test_extract_top_features_sorted(self, mock_model_manager):
        """Returned explanations are sorted by impact descending."""
        service = InferenceService(mock_model_manager)
        shap_values = np.array([[0.01, 0.10, -0.05] + [0.0] * 16])
        explanations = service._extract_top_features(shap_values, base_value=0.2)
        impacts = [e.impact for e in explanations]
        assert impacts == sorted(impacts, reverse=True)


# ==================== Integration Tests ====================

@pytest.mark.asyncio
async def test_predict_loan_full_flow(mock_model_manager, sample_loan_request):
    """InferenceService initialises correctly with a mock model manager."""
    mock_model_manager.loan_model.predict.return_value = np.array([0.25])
    service = InferenceService(mock_model_manager)
    assert service.feature_names is not None
    assert len(service.feature_names) == 19
