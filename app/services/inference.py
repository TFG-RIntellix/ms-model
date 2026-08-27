"""Inference service for loan and credit card risk prediction.

Handles model prediction and SHAP-based explainability.
Integrates with trained XGBoost models, categorical encoders,
and feature scalers.
"""

import logging
import asyncio

import numpy as np
import pandas as pd
import xgboost as xgb

from app.core.features import FEATURE_ORDER, CREDIT_CARD_FEATURE_ORDER
from app.schemas.models import (
    LoanApplicationRequest, CreditCardApplicationRequest, PredictionResponse, SHAPExplanation
)
from app.services.encoder import CategoricalEncoder
from app.core.settings import get_settings
from app.core.constants import RiskSegment, SHAPDirection, ErrorMessages

from fastapi import Request

logger = logging.getLogger(__name__)


class BaseInferenceService:
    """Base service for model inference and risk assessment."""
    
    def __init__(self) -> None:
        settings = get_settings()
        self.threshold_low = settings.RISK_THRESHOLD_LOW
        self.threshold_high = settings.RISK_THRESHOLD_HIGH

    def _get_risk_segment(self, pd_value: float) -> str:
        """Classify probability of default into risk segment."""
        if pd_value < self.threshold_low:
            return RiskSegment.LOW.value
        elif pd_value < self.threshold_high:
            return RiskSegment.MEDIUM.value
        else:
            return RiskSegment.HIGH.value

    def _extract_top_features(self, shap_values: np.ndarray, feature_names: list[str]) -> list:
        """Extract top 5 features by absolute SHAP value."""
        shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        feature_impacts = []
        for i, feature_name in enumerate(feature_names):
            if i < len(shap_row):
                shap_val = float(shap_row[i])
                feature_impacts.append((feature_name, shap_val))
        
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanations = []
        for feature_name, shap_val in feature_impacts[:5]:
            direction = SHAPDirection.INCREASE.value if shap_val > 0 else SHAPDirection.DECREASE.value
            explanations.append(
                SHAPExplanation(
                    feature=feature_name,
                    impact=shap_val,
                    direction=direction,
                )
            )
        
        return explanations


class LoanInferenceService(BaseInferenceService):
    """Service for loan model inference and risk assessment."""
    
    def __init__(self, model_manager) -> None:
        super().__init__()
        self.model = model_manager.loan_model
        self.encoder = model_manager.encoder
        self.scaler = getattr(model_manager, "scaler", None)
        self.explainer = getattr(model_manager, "explainer", None)
        self.feature_names = CategoricalEncoder.get_feature_names()

    async def predict(self, request: LoanApplicationRequest) -> PredictionResponse:
        X = self._encode_request(request)
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        pd_value = await self._predict(X)
        shap_values, _ = await self._explain(X)

        risk_segment = self._get_risk_segment(pd_value)
        explanations = self._extract_top_features(shap_values, self.feature_names)

        return PredictionResponse(
            probability_of_default=pd_value,
            risk_segment=risk_segment,
            shap_explanations=explanations,
        )

    def _encode_request(self, request: LoanApplicationRequest) -> np.ndarray:
        request_df = pd.DataFrame([request.model_dump(mode="json")])
        if self.encoder is not None:
            encoded_df = self.encoder.transform(request_df[FEATURE_ORDER])
            return encoded_df[FEATURE_ORDER].values.astype(np.float32)
        return CategoricalEncoder.encode_request(request.model_dump())

    async def _predict(self, X: np.ndarray) -> float:
        def _model_predict() -> float:
            if self.model is not None:
                dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
                predictions = self.model.predict(dmatrix)
                pd_value = float(1.0 / (1.0 + np.exp(-np.clip(predictions[0], -30.0, 30.0))))
                logger.info("Prediction: PD=%.6f", pd_value)
                return pd_value
            else:
                raise RuntimeError(ErrorMessages.LOAN_MODEL_NOT_LOADED)
        return await asyncio.to_thread(_model_predict)
    
    async def _explain(self, X: np.ndarray) -> tuple:
        explainer = self.explainer
        def _generate_shap():
            if self.model is not None:
                try:
                    if explainer is None:
                        raise RuntimeError(ErrorMessages.LOAN_EXPLAINER_NOT_LOADED)
                    shap_values = explainer.shap_values(X)
                    base_value = explainer.expected_value
                    return shap_values, base_value
                except Exception as e:
                    logger.error("SHAP calculation error: %s", str(e), exc_info=True)
                    raise RuntimeError(ErrorMessages.SHAP_CALCULATION_ERROR)
            else: 
                raise RuntimeError(ErrorMessages.LOAN_MODEL_NOT_LOADED)
        return await asyncio.to_thread(_generate_shap)


class CreditCardInferenceService(BaseInferenceService):
    """Service for credit card model inference and risk assessment."""
    
    def __init__(self, model_manager) -> None:
        super().__init__()
        self.model = getattr(model_manager, "credit_card_model", None)
        self.encoder = getattr(model_manager, "credit_card_encoder", None)
        self.scaler = getattr(model_manager, "credit_card_scaler", None)
        self.explainer = getattr(model_manager, "credit_card_explainer", None)
        self.feature_names = CategoricalEncoder.get_credit_card_feature_names()

    async def predict(self, request: CreditCardApplicationRequest) -> PredictionResponse:
        X = self._encode_request(request)
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        pd_value = await self._predict(X)
        shap_values, _ = await self._explain(X)

        risk_segment = self._get_risk_segment(pd_value)
        explanations = self._extract_top_features(shap_values, self.feature_names)

        return PredictionResponse(
            probability_of_default=float(pd_value),
            risk_segment=risk_segment,
            shap_explanations=explanations,
        )

    def _encode_request(self, request: CreditCardApplicationRequest) -> np.ndarray:
        request_df = pd.DataFrame([request.model_dump(mode="json")])
        if self.encoder is not None:
            encoded_df = self.encoder.transform(request_df[CREDIT_CARD_FEATURE_ORDER])
            return encoded_df[CREDIT_CARD_FEATURE_ORDER].values.astype(np.float32)
        raise RuntimeError(ErrorMessages.CREDIT_CARD_ENCODER_NOT_LOADED)

    async def _predict(self, X: np.ndarray) -> float:
        def _model_predict() -> float:
            if self.model is not None:
                dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
                predictions = self.model.predict(dmatrix)
                pd_value = float(1.0 / (1.0 + np.exp(-np.clip(predictions[0], -30.0, 30.0))))
                logger.info("Credit card prediction: PD=%.6f", pd_value)
                return pd_value
            else:
                raise RuntimeError(ErrorMessages.CREDIT_CARD_MODEL_NOT_LOADED)
        return await asyncio.to_thread(_model_predict)
    
    async def _explain(self, X: np.ndarray) -> tuple:
        explainer = self.explainer
        def _generate_shap():
            if self.model is not None:
                try:
                    if explainer is None:
                        raise RuntimeError(ErrorMessages.CREDIT_CARD_EXPLAINER_NOT_LOADED)
                    shap_values = explainer.shap_values(X)
                    base_value = explainer.expected_value
                    return shap_values, base_value
                except Exception as e:
                    logger.error("Credit card SHAP calculation error: %s", str(e), exc_info=True)
                    raise RuntimeError(ErrorMessages.SHAP_CALCULATION_ERROR)
            else: 
                raise RuntimeError(ErrorMessages.CREDIT_CARD_MODEL_NOT_LOADED)
        return await asyncio.to_thread(_generate_shap)


async def get_loan_inference_service(request: Request) -> LoanInferenceService:
    app = request.app
    model_manager = getattr(app.state, "model_manager", None)
    if model_manager is None:
        raise RuntimeError(ErrorMessages.MODEL_MANAGER_UNINITIALIZED)
    return LoanInferenceService(model_manager)


async def get_credit_card_inference_service(request: Request) -> CreditCardInferenceService:
    app = request.app
    model_manager = getattr(app.state, "model_manager", None)
    if model_manager is None:
        raise RuntimeError(ErrorMessages.MODEL_MANAGER_UNINITIALIZED)
    return CreditCardInferenceService(model_manager)
