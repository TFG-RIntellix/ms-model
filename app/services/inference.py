"""Inference service for loan risk prediction.

Handles model prediction and SHAP-based explainability.
Integrates with trained XGBoost models, categorical encoders,
and feature scalers.

Typical usage::

    service = InferenceService(model_manager)
    response = await service.predict_loan(request)
"""

import logging
import asyncio

import numpy as np
import xgboost as xgb

from app.schemas.models import (
    LoanApplicationRequest, PredictionResponse, SHAPExplanation
)
from app.services.encoder import CategoricalEncoder
from app.core.settings import get_settings

from fastapi import Request

logger = logging.getLogger(__name__)


# This class is the one who makes the predictions and SHAP explanations.
# It's used by the risk router controller method.
class InferenceService:
    """Service for model inference and risk assessment."""
    
    def __init__(self, model_manager) -> None:
        """Initialise with a loaded model manager.

        Args:
            model_manager: ``ModelManager`` instance with loaded models.
        """
        self.model = model_manager.loan_model
        self.encoder = model_manager.encoder
        self.scaler = getattr(model_manager, "scaler", None)
        self.explainer = getattr(model_manager, "explainer", None)
        self.feature_names = CategoricalEncoder.get_feature_names()

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
        # Encode categorical fields of the request into numerical ones 
        X = CategoricalEncoder.encode_request(request.model_dump())

        # It scales the numerical fields of the request to be in the same range as the treated training data in the model. 
        # This is done to improve the model's performance.
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        ## Call to private methods who make the prediction and SHAP explanations.
        pd_value = await self._predict(X)
        shap_values, base_value = await self._explain(X)

        risk_segment = self._get_risk_segment(pd_value)
        explanations = self._extract_top_features(shap_values, base_value)

        return PredictionResponse(
            probability_of_default=float(pd_value),
            risk_segment=risk_segment,
            shap_explanations=explanations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _predict(self, X: np.ndarray) -> float:
        """Execute ``model.predict()`` in a thread pool.

        Args:
            X: Encoded (and optionally scaled) feature array.

        Returns:
            Probability of default clamped to ``[0, 1]``.
        """
        def _model_predict() -> float:
            if self.model is not None:
                dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
                predictions = self.model.predict(dmatrix)
                pd_value = float(predictions[0])
                logger.info("Prediction: PD=%.4f", pd_value)
                return pd_value
            else:
                raise RuntimeError("Model not loaded")
        return await asyncio.to_thread(_model_predict)
    
    async def _explain(self, X: np.ndarray) -> tuple:
        """Generate SHAP explanations using the pre-created explainer.

        Uses the ``TreeExplainer`` pre-created at application startup by
        ``ModelManager`` to avoid the cold-start cost of building it on
        every request.

        Args:
            X: Encoded (and optionally scaled) feature array.

        Returns:
            Tuple of ``(shap_values, base_value)``.
        """
        # Capture for use inside the thread
        explainer = self.explainer

        def _generate_shap():
            if self.model is not None:
                try:
                    active_explainer = explainer
                    if active_explainer is None:
                        raise RuntimeError("Explainer not loaded")
                    shap_values = active_explainer.shap_values(X)
                    base_value = active_explainer.expected_value
                    logger.info("SHAP explanations generated. Shape: %s",
                                shap_values.shape)
                    logger.info ("Shap values: %s", shap_values.tolist().__str__())
                    return shap_values, base_value
                except Exception as e:
                    logger.error("SHAP calculation error: %s", str(e),
                                 exc_info=True)
                    raise RuntimeError("SHAP calculation error")
            else: 
                raise RuntimeError("Model not loaded")
        return await asyncio.to_thread(_generate_shap)
    
    def _extract_top_features(self, shap_values: np.ndarray, base_value: float) -> list:
        """
        Extract top 5 features by absolute SHAP value.
        
        Args:
            shap_values: SHAP values array with shape (1, n_features)
            base_value: Base value (expected model output)
            
        Returns:
            List of SHAPExplanation objects sorted by absolute impact
        """
        # Get first row (single prediction)
        shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Create list of (feature_name, shap_value) tuples
        feature_impacts = []
        for i, feature_name in enumerate(self.feature_names):
            if i < len(shap_row):
                shap_val = float(shap_row[i])
                feature_impacts.append((feature_name, shap_val))
        
        # Sort by absolute value, descending
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Extract top 5 and create SHAPExplanation objects
        explanations = []
        for feature_name, shap_val in feature_impacts[:5]:
            direction = "increase" if shap_val > 0 else "decrease"
            explanations.append(
                SHAPExplanation(
                    feature=feature_name,
                    impact=float(shap_val),
                    direction=direction,
                )
            )
        
        return explanations
    
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

# This method gets the inference service who calls the model to make predictions
# It's used as a dependency in the risk router controller method.
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

