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
from app.services.inference import LoanInferenceService, BaseInferenceService
from app.core.constants import RiskSegment, SHAPDirection

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

class TestBaseInferenceService:
    """Tests for ``BaseInferenceService`` business logic."""

    def test_risk_segment_low(self):
        """PD < 0.15 maps to 'Low'."""
        service = BaseInferenceService()
        assert service._get_risk_segment(0.10) == RiskSegment.LOW.value
        assert service._get_risk_segment(0.14999) == RiskSegment.LOW.value

    def test_risk_segment_medium(self):
        """0.15 ≤ PD < 0.35 maps to 'Medium'."""
        service = BaseInferenceService()
        assert service._get_risk_segment(0.15) == RiskSegment.MEDIUM.value
        assert service._get_risk_segment(0.25) == RiskSegment.MEDIUM.value
        assert service._get_risk_segment(0.34999) == RiskSegment.MEDIUM.value

    def test_risk_segment_high(self):
        """PD ≥ 0.35 maps to 'High'."""
        service = BaseInferenceService()
        assert service._get_risk_segment(0.35) == RiskSegment.HIGH.value
        assert service._get_risk_segment(0.50) == RiskSegment.HIGH.value
        assert service._get_risk_segment(0.95) == RiskSegment.HIGH.value

    def test_extract_top_features(self):
        """Top-5 SHAP features are sorted by descending absolute value."""
        service = BaseInferenceService()
        shap_values = np.array([[
            0.12, -0.08, 0.06, -0.04, 0.03,
            0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01,
        ]])
        feature_names = CategoricalEncoder.get_feature_names()
        explanations = service._extract_top_features(shap_values, feature_names=feature_names)

        assert len(explanations) == 5
        assert explanations[0].feature == "age"
        assert explanations[0].impact == pytest.approx(0.12)
        assert explanations[0].direction == SHAPDirection.INCREASE.value

        assert explanations[1].feature == "gender"
        assert explanations[1].impact == pytest.approx(-0.08)
        assert explanations[1].direction == SHAPDirection.DECREASE.value

    def test_extract_top_features_sorted(self):
        """Returned explanations are sorted by impact descending."""
        service = BaseInferenceService()
        shap_values = np.array([[0.01, 0.10, -0.05] + [0.0] * 16])
        feature_names = CategoricalEncoder.get_feature_names()
        explanations = service._extract_top_features(shap_values, feature_names=feature_names)
        magnitudes = [abs(e.impact) for e in explanations]
        assert magnitudes == sorted(magnitudes, reverse=True)


# ==================== Integration Tests ====================

@pytest.mark.asyncio
async def test_predict_loan_full_flow(mock_model_manager, sample_loan_request):
    """LoanInferenceService initialises correctly with a mock model manager."""
    mock_model_manager.loan_model.predict.return_value = np.array([0.25])
    mock_model_manager.explainer = Mock()
    mock_model_manager.explainer.shap_values.return_value = np.zeros((1, 19))
    mock_model_manager.explainer.expected_value = 0.0

    service = LoanInferenceService(mock_model_manager)
    response = await service.predict(sample_loan_request)

    assert service.feature_names is not None
    assert len(service.feature_names) == 19
    assert response.probability_of_default == pytest.approx(0.25, rel=1e-4)
    assert response.risk_segment == RiskSegment.MEDIUM.value


@pytest.mark.asyncio
async def test_predict_credit_card_full_flow(mock_model_manager):
    """CreditCardInferenceService initialises correctly with a mock model manager."""
    from app.schemas.models import CreditCardApplicationRequest
    from app.services.inference import CreditCardInferenceService

    # Provide a simple mock request for credit card
    sample_cc_request = CreditCardApplicationRequest(
        edad=35,
        estado_laboral="Empleado",
        antiguedad_laboral=5,
        ingresos_anuales=50000.0,
        tipo_ingreso="Salario",
        propiedad_vivienda="Propia",
        dependientes=1,
        limite_credito=10000.0,
        es_revolving="No",
        tasa_interes=0.15,
        ratio_limite_ingreso=0.2,
        dti=0.3,
        impagos_previos=0
    )

    mock_model_manager.credit_card_model = Mock()
    mock_model_manager.credit_card_model.predict.return_value = np.array([0.4])
    mock_model_manager.credit_card_explainer = Mock()
    mock_model_manager.credit_card_explainer.shap_values.return_value = np.zeros((1, 13))
    mock_model_manager.credit_card_explainer.expected_value = 0.0

    service = CreditCardInferenceService(mock_model_manager)
    response = await service.predict(sample_cc_request)

    assert service.feature_names is not None
    assert len(service.feature_names) == 13
    assert response.probability_of_default == pytest.approx(0.4, rel=1e-4)
    assert response.risk_segment == RiskSegment.HIGH.value
