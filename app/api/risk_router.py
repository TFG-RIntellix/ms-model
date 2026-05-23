"""Risk assessment API endpoints (v1).

Handles loan and credit card risk prediction requests.
Each endpoint validates the incoming payload against Pydantic schemas
and delegates to :class:`~app.services.inference.InferenceService`.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, status, Body, Request, Depends

from app.core.settings import get_settings
from app.core.features import FEATURE_ORDER
from app.schemas.requests import LoanApplicationRequest, CreditCardApplicationRequest
from app.schemas.responses import PredictionResponse, ModelInfoResponse, ErrorResponse
from app.services.inference import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

_settings = get_settings()

router = APIRouter(
    prefix=_settings.API_V1_PREFIX + "/risk",
    tags=["Risk Assessment"],
)


# ------------------------------------------------------------------
# Model information
# ------------------------------------------------------------------

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Model Information",
    description="Return metadata about the loaded ML model and service configuration.",
)
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the loaded ML model.

    Useful for auditing, monitoring dashboards, and health probes that
    need more detail than the lightweight ``/health`` endpoint.

    Args:
        request: Incoming ``Request`` (used to access ``app.state``).

    Returns:
        ``ModelInfoResponse`` with model status, features, and thresholds.
    """
    model_manager = getattr(request.app.state, "model_manager", None)

    boosting_rounds = None
    model_artifact_date = None

    if model_manager and model_manager.model_loaded and model_manager.loan_model:
        boosting_rounds = model_manager.loan_model.num_boosted_rounds()

        model_path = Path(_settings.MODEL_PATH)
        if model_path.exists():
            mtime = model_path.stat().st_mtime
            model_artifact_date = datetime.fromtimestamp(
                mtime, tz=timezone.utc
            ).isoformat()

    return ModelInfoResponse(
        app_version=_settings.APP_VERSION,
        model_loaded=model_manager.model_loaded if model_manager else False,
        boosting_rounds=boosting_rounds,
        feature_names=list(FEATURE_ORDER),
        scaler_loaded=model_manager.scaler_loaded if model_manager else False,
        encoder_loaded=model_manager.encoder_loaded if model_manager else False,
        model_artifact_date=model_artifact_date,
        risk_thresholds={
            "low_below": _settings.RISK_THRESHOLD_LOW,
            "high_at_or_above": _settings.RISK_THRESHOLD_HIGH,
        },
    )


# ------------------------------------------------------------------
# Loan risk prediction
# ------------------------------------------------------------------

@router.post(
    "/predict-loan",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Predict Loan Risk",
    description=(
        "Calculate probability of default and SHAP explanations "
        "for a loan application."
    ),
)
async def predict_loan(
    request: LoanApplicationRequest,
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Loan risk prediction endpoint.

    Accepts a loan application and returns:

    - Probability of Default (0–1).
    - Risk segment (*Low* / *Medium* / *High*).
    - Top 5 SHAP feature explanations.

    Args:
        request: Validated loan application details.
        service: Injected inference service instance.

    Returns:
        ``PredictionResponse`` with PD, segment, and explanations.
    """
    logger.info("Prediction request for loan application from %s", request.gender)

    response = await service.predict_loan(request)

    logger.info(
        "Prediction completed: PD=%.3f, Segment=%s",
        response.probability_of_default,
        response.risk_segment,
    )
    return response


# ------------------------------------------------------------------
# Credit card risk prediction (placeholder)
# ------------------------------------------------------------------

@router.post(
    "/predict-credit-card",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
    summary="Predict Credit Card Risk (Placeholder)",
    description="Placeholder endpoint for credit card risk assessment. Implementation pending.",
)
async def predict_credit_card(
    request: CreditCardApplicationRequest,
) -> dict:
    """Credit card risk prediction endpoint (PLACEHOLDER).

    This endpoint is reserved for future credit card risk assessment.
    The schema and ML model for credit cards are pending definition.

    Args:
        request: Placeholder credit card application.

    Returns:
        Placeholder response indicating feature set is pending.
    """
    logger.info("Credit card prediction requested — feature set pending")

    return {
        "status": "pending",
        "message": "Credit card risk assessment model is under development",
        "next_update": "To be announced",
    }
