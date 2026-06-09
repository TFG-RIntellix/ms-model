"""Unit tests to improve coverage for app/services/inference.py.

Targets the credit card prediction path, error branches, and the
get_inference_service dependency — all previously untested (coverage was 62 %).
"""

from unittest.mock import Mock

import numpy as np
import pytest

from app.schemas.enums import (
    CreditCardEmploymentStatusEnum,
    CreditCardHomeOwnershipEnum,
    CreditCardIncomeTypeEnum,
    CreditCardIsRevolvingEnum,
)
from app.schemas.requests import CreditCardApplicationRequest
from app.services.inference import InferenceService, get_inference_service

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_credit_card_request() -> CreditCardApplicationRequest:
    """Sample credit card application for testing."""
    return CreditCardApplicationRequest(
        age=41,
        employmentStatus=CreditCardEmploymentStatusEnum.PERMANENT,
        employmentSeniorityYears=10,
        annualIncome=48000.0,
        incomeType=CreditCardIncomeTypeEnum.SALARY,
        homeOwnership=CreditCardHomeOwnershipEnum.OWNED_MORTGAGED,
        dependents=2,
        creditLimit=5000.0,
        isRevolving=CreditCardIsRevolvingEnum.NO,
        interestRate=0.18,
        lti=1.2,
        dti=0.35,
        previousDefaultsCount=0,
    )


def _make_model_manager(
    *,
    loan_model=None,
    credit_card_model=None,
    credit_card_encoder=None,
    credit_card_scaler=None,
    credit_card_explainer=None,
):
    """Build a mock ModelManager with fine-grained control."""
    mgr = Mock()
    mgr.loan_model = loan_model or Mock()
    mgr.encoder = None
    mgr.scaler = None
    mgr.explainer = None
    mgr.credit_card_model = credit_card_model
    mgr.credit_card_encoder = credit_card_encoder
    mgr.credit_card_scaler = credit_card_scaler
    mgr.credit_card_explainer = credit_card_explainer
    return mgr


# ---------------------------------------------------------------------------
# Risk segment tests (boundary values)
# ---------------------------------------------------------------------------


class TestRiskSegmentBoundaries:
    """Edge-case tests for ``_get_risk_segment``."""

    def test_zero_pd(self):
        """PD of 0 is Low."""
        svc = InferenceService(_make_model_manager())
        assert svc._get_risk_segment(0.0) == "Low"

    def test_one_pd(self):
        """PD of 1 is High."""
        svc = InferenceService(_make_model_manager())
        assert svc._get_risk_segment(1.0) == "High"

    def test_exact_low_threshold(self):
        """PD exactly at low threshold is Medium."""
        svc = InferenceService(_make_model_manager())
        assert svc._get_risk_segment(0.15) == "Medium"

    def test_just_below_low_threshold(self):
        """PD just below low threshold is Low."""
        svc = InferenceService(_make_model_manager())
        assert svc._get_risk_segment(0.1499) == "Low"

    def test_exact_high_threshold(self):
        """PD exactly at high threshold is High."""
        svc = InferenceService(_make_model_manager())
        assert svc._get_risk_segment(0.35) == "High"


# ---------------------------------------------------------------------------
# Credit card extract top features
# ---------------------------------------------------------------------------


class TestExtractTopFeaturesCreditCard:
    """Tests for ``_extract_top_features_credit_card``."""

    def test_top_5_returned(self):
        """At most 5 explanations are returned."""
        svc = InferenceService(_make_model_manager())
        shap_vals = np.zeros((1, 13))
        shap_vals[0, 0] = 0.5
        shap_vals[0, 1] = -0.4
        shap_vals[0, 2] = 0.3
        shap_vals[0, 3] = -0.2
        shap_vals[0, 4] = 0.1
        shap_vals[0, 5] = 0.05

        result = svc._extract_top_features_credit_card(shap_vals, base_value=0.0)
        assert len(result) == 5

    def test_sorted_by_abs_impact(self):
        """Returned explanations are sorted by descending absolute impact."""
        svc = InferenceService(_make_model_manager())
        shap_vals = np.random.randn(1, 13)
        result = svc._extract_top_features_credit_card(shap_vals, base_value=0.0)
        abs_impacts = [abs(e.impact) for e in result]
        assert abs_impacts == sorted(abs_impacts, reverse=True)

    def test_direction_labels(self):
        """Positive SHAP values are 'increase', negative are 'decrease'."""
        svc = InferenceService(_make_model_manager())
        shap_vals = np.zeros((1, 13))
        shap_vals[0, 0] = 0.1
        shap_vals[0, 1] = -0.2

        result = svc._extract_top_features_credit_card(shap_vals, base_value=0.0)
        directions = {e.feature: e.direction for e in result}
        assert directions["age"] == "increase"
        assert directions["employmentStatus"] == "decrease"

    def test_1d_shap_values(self):
        """Handle 1-D shap array (single row, no batch dimension)."""
        svc = InferenceService(_make_model_manager())
        shap_vals = np.array([0.1, -0.2] + [0.0] * 11)
        result = svc._extract_top_features_credit_card(shap_vals, base_value=0.0)
        assert len(result) <= 5


# ---------------------------------------------------------------------------
# Credit card prediction end-to-end (async)
# ---------------------------------------------------------------------------


class TestCreditCardPrediction:
    """End-to-end tests for credit card prediction."""

    @pytest.mark.asyncio
    async def test_predict_credit_card_success(self, sample_credit_card_request):
        """Full predict_credit_card path with mocked model components."""
        cc_model = Mock()
        cc_model.predict.return_value = np.array([0.0])  # logit=0 → sigmoid=0.5

        cc_encoder = Mock()
        cc_encoder.transform.return_value = Mock(
            __getitem__=lambda self, key: Mock(
                values=Mock(astype=lambda dt: np.zeros((1, 13), dtype=np.float32))
            )
        )

        cc_explainer = Mock()
        cc_explainer.shap_values.return_value = np.zeros((1, 13))
        cc_explainer.expected_value = 0.0

        mgr = _make_model_manager(
            credit_card_model=cc_model,
            credit_card_encoder=cc_encoder,
            credit_card_explainer=cc_explainer,
        )
        svc = InferenceService(mgr)
        resp = await svc.predict_credit_card(sample_credit_card_request)

        assert 0 <= resp.probability_of_default <= 1
        assert resp.risk_segment in ("Low", "Medium", "High")
        assert len(resp.shap_explanations) <= 5

    @pytest.mark.asyncio
    async def test_predict_credit_card_with_scaler(self, sample_credit_card_request):
        """Scaler branch is exercised when present."""
        cc_model = Mock()
        cc_model.predict.return_value = np.array([0.0])

        cc_encoder = Mock()
        cc_encoder.transform.return_value = Mock(
            __getitem__=lambda self, key: Mock(
                values=Mock(astype=lambda dt: np.zeros((1, 13), dtype=np.float32))
            )
        )

        cc_scaler = Mock()
        cc_scaler.transform.return_value = np.zeros((1, 13), dtype=np.float32)

        cc_explainer = Mock()
        cc_explainer.shap_values.return_value = np.zeros((1, 13))
        cc_explainer.expected_value = 0.0

        mgr = _make_model_manager(
            credit_card_model=cc_model,
            credit_card_encoder=cc_encoder,
            credit_card_scaler=cc_scaler,
            credit_card_explainer=cc_explainer,
        )
        svc = InferenceService(mgr)
        resp = await svc.predict_credit_card(sample_credit_card_request)

        cc_scaler.transform.assert_called_once()
        assert resp.probability_of_default == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_predict_credit_card_model_not_loaded(self, sample_credit_card_request):
        """RuntimeError when credit card model is None."""
        cc_encoder = Mock()
        cc_encoder.transform.return_value = Mock(
            __getitem__=lambda self, key: Mock(
                values=Mock(astype=lambda dt: np.zeros((1, 13), dtype=np.float32))
            )
        )

        mgr = _make_model_manager(
            credit_card_model=None,
            credit_card_encoder=cc_encoder,
        )
        svc = InferenceService(mgr)

        with pytest.raises(RuntimeError):
            await svc._predict_credit_card(np.zeros((1, 13), dtype=np.float32))

    @pytest.mark.asyncio
    async def test_explain_credit_card_model_not_loaded(self):
        """RuntimeError when credit card model is None for SHAP."""
        mgr = _make_model_manager(credit_card_model=None)
        svc = InferenceService(mgr)

        with pytest.raises(RuntimeError):
            await svc._explain_credit_card(np.zeros((1, 13), dtype=np.float32))

    @pytest.mark.asyncio
    async def test_explain_credit_card_explainer_not_loaded(self):
        """RuntimeError when credit card explainer is None but model exists."""
        mgr = _make_model_manager(
            credit_card_model=Mock(),
            credit_card_explainer=None,
        )
        svc = InferenceService(mgr)

        with pytest.raises(RuntimeError):
            await svc._explain_credit_card(np.zeros((1, 13), dtype=np.float32))

    @pytest.mark.asyncio
    async def test_encode_credit_card_no_encoder_raises(self, sample_credit_card_request):
        """RuntimeError when credit card encoder is None."""
        mgr = _make_model_manager(credit_card_encoder=None)
        svc = InferenceService(mgr)

        with pytest.raises(RuntimeError, match="Credit card encoder not loaded"):
            svc._encode_credit_card_request(sample_credit_card_request)


# ---------------------------------------------------------------------------
# Loan prediction error paths
# ---------------------------------------------------------------------------


class TestLoanPredictionErrors:
    """Error-path tests for loan prediction."""

    @pytest.mark.asyncio
    async def test_predict_model_not_loaded(self):
        """RuntimeError when loan model is None."""
        mgr = _make_model_manager(loan_model=None)
        mgr.loan_model = None
        svc = InferenceService(mgr)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            await svc._predict(np.zeros((1, 19), dtype=np.float32))

    @pytest.mark.asyncio
    async def test_explain_model_not_loaded(self):
        """RuntimeError when loan model is None for SHAP."""
        mgr = _make_model_manager(loan_model=None)
        mgr.loan_model = None
        svc = InferenceService(mgr)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            await svc._explain(np.zeros((1, 19), dtype=np.float32))

    @pytest.mark.asyncio
    async def test_explain_explainer_not_loaded(self):
        """Model is loaded but explainer is None."""
        mgr = _make_model_manager()
        mgr.explainer = None
        svc = InferenceService(mgr)

        with pytest.raises(RuntimeError):
            await svc._explain(np.zeros((1, 19), dtype=np.float32))


# ---------------------------------------------------------------------------
# get_inference_service dependency
# ---------------------------------------------------------------------------


class TestGetInferenceService:
    """Tests for the ``get_inference_service`` FastAPI dependency."""

    @pytest.mark.asyncio
    async def test_returns_inference_service(self):
        """Returns an InferenceService when model_manager is available."""
        mock_request = Mock()
        mock_request.app.state.model_manager = _make_model_manager()

        svc = await get_inference_service(mock_request)
        assert isinstance(svc, InferenceService)

    @pytest.mark.asyncio
    async def test_raises_when_model_manager_missing(self):
        """RuntimeError when model_manager is not on app.state."""
        mock_request = Mock()
        mock_request.app.state = Mock(spec=[])  # no model_manager attribute

        with pytest.raises(RuntimeError, match="Model manager not initialized"):
            await get_inference_service(mock_request)
