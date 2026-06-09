"""Inference service for loan and credit card risk prediction.

Handles model prediction and SHAP-based explainability.
Integrates with trained XGBoost models, categorical encoders,
and feature scalers.

Typical usage::

    service = InferenceService(model_manager)
    response = await service.predict_loan(request)
    response = await service.predict_credit_card(request)
"""

import asyncio
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import Request

from app.core.features import CREDIT_CARD_FEATURE_ORDER, FEATURE_ORDER
from app.core.settings import get_settings
from app.schemas.models import (
    CreditCardApplicationRequest,
    LoanApplicationRequest,
    PredictionResponse,
    SHAPExplanation,
)
from app.services.encoder import CategoricalEncoder

logger = logging.getLogger(__name__)


class InferenceService:
    """Service for model inference and risk assessment."""

    def __init__(self, model_manager) -> None:
        """Initialise with a loaded model manager.

        Args:
            model_manager: ``ModelManager`` instance with loaded models.

        """
        # Loan model components
        self.model = model_manager.loan_model
        self.encoder = model_manager.encoder
        self.scaler = getattr(model_manager, "scaler", None)
        self.explainer = getattr(model_manager, "explainer", None)
        self.feature_names = CategoricalEncoder.get_feature_names()

        # Credit card model components
        self.credit_card_model = getattr(model_manager, "credit_card_model", None)
        self.credit_card_encoder = getattr(model_manager, "credit_card_encoder", None)
        self.credit_card_scaler = getattr(model_manager, "credit_card_scaler", None)
        self.credit_card_explainer = getattr(model_manager, "credit_card_explainer", None)
        self.credit_card_feature_names = CategoricalEncoder.get_credit_card_feature_names()

        # Risk thresholds from centralised configuration
        settings = get_settings()
        self.threshold_low = settings.RISK_THRESHOLD_LOW
        self.threshold_high = settings.RISK_THRESHOLD_HIGH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict_loan(self, request: LoanApplicationRequest) -> PredictionResponse:
        """Predict loan risk and generate SHAP explanations.

        Args:
            request: Loan application details.

        Returns:
            ``PredictionResponse`` containing the probability of default,
            risk segment, and top-5 SHAP explanations.

        Raises:
            RuntimeError: If the model is not loaded and cannot predict.

        """
        X = self._encode_request(request)

        return await self._predict_risk(
            X=X,
            model=self.model,
            scaler=self.scaler,
            explainer=self.explainer,
            feature_names=self.feature_names,
            label="loan",
        )

    async def predict_credit_card(
        self, request: CreditCardApplicationRequest,
    ) -> PredictionResponse:
        """Predict credit card risk and generate SHAP explanations.

        Args:
            request: Credit card application details.

        Returns:
            ``PredictionResponse`` containing the probability of default,
            risk segment, and top-5 SHAP explanations.

        Raises:
            RuntimeError: If the credit card model is not loaded and cannot predict.

        """
        X = self._encode_credit_card_request(request)

        return await self._predict_risk(
            X=X,
            model=self.credit_card_model,
            scaler=self.credit_card_scaler,
            explainer=self.credit_card_explainer,
            feature_names=self.credit_card_feature_names,
            label="credit card",
        )

    # ------------------------------------------------------------------
    # Unified prediction pipeline
    # ------------------------------------------------------------------

    async def _predict_risk(
        self,
        X: np.ndarray,
        model: xgb.Booster | None,
        scaler,
        explainer,
        feature_names: list[str],
        label: str,
    ) -> PredictionResponse:
        """Run the full prediction pipeline for any model type.

        Applies scaling, model prediction, SHAP explanation, risk
        segmentation, and top-feature extraction in a single flow.

        Args:
            X: Encoded feature array of shape ``(1, n_features)``.
            model: Trained XGBoost ``Booster`` instance.
            scaler: Fitted ``StandardScaler`` (or ``None``).
            explainer: Pre-built SHAP ``TreeExplainer`` (or ``None``).
            feature_names: Ordered feature names matching the model columns.
            label: Human-readable model label for log messages.

        Returns:
            ``PredictionResponse`` with PD, risk segment, and SHAP explanations.

        """
        if scaler is not None:
            X = scaler.transform(X).astype(np.float32)

        pd_value = await self._run_prediction(model, feature_names, X, label)
        shap_values, base_value = await self._run_explanation(
            model, explainer, X, label
        )

        risk_segment = self._get_risk_segment(pd_value)
        explanations = self._extract_top_features(shap_values, base_value, feature_names)

        return PredictionResponse(
            probability_of_default=float(pd_value),
            risk_segment=risk_segment,
            shap_explanations=explanations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_request(self, request: LoanApplicationRequest) -> np.ndarray:
        """Encode loan request data using the training encoder when possible."""
        return self._encode_with_training_encoder(
            request=request,
            encoder=self.encoder,
            feature_order=FEATURE_ORDER,
            fallback_fn=lambda req: CategoricalEncoder.encode_request(req.model_dump()),
        )

    def _encode_credit_card_request(self, request: CreditCardApplicationRequest) -> np.ndarray:
        """Encode credit card request data using the training encoder."""
        return self._encode_with_training_encoder(
            request=request,
            encoder=self.credit_card_encoder,
            feature_order=CREDIT_CARD_FEATURE_ORDER,
            fallback_fn=None,
        )

    @staticmethod
    def _encode_with_training_encoder(
        request, encoder, feature_order, fallback_fn=None,
    ) -> np.ndarray:
        """Encode a request using the persisted training encoder.

        Falls back to ``fallback_fn`` when no encoder is available. If
        neither is available, raises ``RuntimeError``.

        Args:
            request: Pydantic request model instance.
            encoder: Fitted ``TrainingEncoder`` (or ``None``).
            feature_order: Column order expected by the model.
            fallback_fn: Callable accepting the request, returning an ndarray.

        Returns:
            ``np.ndarray`` of shape ``(1, n_features)`` with dtype ``float32``.

        """
        request_df = pd.DataFrame([request.model_dump(mode="json")])

        if encoder is not None:
            encoded_df = encoder.transform(request_df[feature_order])
            return encoded_df[feature_order].values.astype(np.float32)

        if fallback_fn is not None:
            return fallback_fn(request)

        raise RuntimeError("Encoder not loaded")

    async def _run_prediction(
        self,
        model: xgb.Booster | None,
        feature_names: list[str],
        X: np.ndarray,
        label: str,
    ) -> float:
        """Execute ``model.predict()`` in a thread pool.

        Args:
            model: XGBoost Booster to use for prediction.
            feature_names: Feature names for the DMatrix.
            X: Encoded (and optionally scaled) feature array.
            label: Human-readable label for log messages.

        Returns:
            Probability of default clamped to ``[0, 1]``.

        """
        def _model_predict() -> float:
            if model is None:
                raise RuntimeError(f"{label.capitalize()} model not loaded")
            dmatrix = xgb.DMatrix(X, feature_names=feature_names)
            predictions = model.predict(dmatrix)
            pd_value = float(
                1.0 / (1.0 + np.exp(-np.clip(predictions[0], -30.0, 30.0)))
            )
            logger.info("%s prediction: PD=%.6f", label.capitalize(), pd_value)
            return pd_value

        return await asyncio.to_thread(_model_predict)

    async def _run_explanation(
        self,
        model: xgb.Booster | None,
        explainer,
        X: np.ndarray,
        label: str,
    ) -> tuple:
        """Generate SHAP explanations in a thread pool.

        Args:
            model: XGBoost Booster (checked for None).
            explainer: Pre-built ``TreeExplainer``.
            X: Encoded (and optionally scaled) feature array.
            label: Human-readable label for log messages.

        Returns:
            Tuple of ``(shap_values, base_value)``.

        """
        def _generate_shap():
            if model is None:
                raise RuntimeError(f"{label.capitalize()} model not loaded")
            if explainer is None:
                raise RuntimeError(f"{label.capitalize()} explainer not loaded")
            try:
                shap_values = explainer.shap_values(X)
                base_value = explainer.expected_value
                logger.info(
                    "%s SHAP explanations generated. Shape: %s",
                    label.capitalize(),
                    shap_values.shape,
                )
                return shap_values, base_value
            except Exception as e:
                logger.error(
                    "%s SHAP calculation error: %s",
                    label.capitalize(),
                    str(e),
                    exc_info=True,
                )
                raise RuntimeError("SHAP calculation error")

        return await asyncio.to_thread(_generate_shap)

    @staticmethod
    def _extract_top_features(
        shap_values: np.ndarray,
        base_value: float,
        feature_names: list[str],
    ) -> list:
        """Extract top 5 features by absolute SHAP value.

        Args:
            shap_values: SHAP values array with shape ``(1, n_features)``.
            base_value: Base value (expected model output).
            feature_names: Ordered feature names matching the SHAP columns.

        Returns:
            List of ``SHAPExplanation`` objects sorted by absolute impact.

        """
        shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        feature_impacts = [
            (feature_names[i], float(shap_row[i]))
            for i in range(min(len(feature_names), len(shap_row)))
        ]

        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        return [
            SHAPExplanation(
                feature=name,
                impact=float(val),
                direction="increase" if val > 0 else "decrease",
            )
            for name, val in feature_impacts[:5]
        ]

    def _get_risk_segment(self, pd_value: float) -> str:
        """Classify probability of default into risk segment.

        Thresholds are read from ``Settings`` at construction time
        so they can be adjusted via environment variables.

        Args:
            pd_value: Probability of Default (0-1)

        Returns:
            Risk segment: ``"Low"``, ``"Medium"``, or ``"High"``.

        """
        if pd_value < self.threshold_low:
            return "Low"
        elif pd_value < self.threshold_high:
            return "Medium"
        else:
            return "High"


async def get_inference_service(request: Request) -> InferenceService:
    """FastAPI dependency that provides an ``InferenceService``.

    Retrieves the ``ModelManager`` stored on ``app.state`` during startup
    and wraps it in an ``InferenceService``.

    Args:
        request: Incoming FastAPI ``Request`` (injected automatically).

    Returns:
        Ready-to-use ``InferenceService`` instance.

    Raises:
        RuntimeError: If the ``ModelManager`` was not initialised.

    """
    app = request.app
    model_manager = getattr(app.state, "model_manager", None)

    if model_manager is None:
        raise RuntimeError("Model manager not initialized")

    return InferenceService(model_manager)
