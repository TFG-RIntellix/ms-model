"""Pydantic v2 response models for the Credit Risk Engine API.

Defines the contract for all JSON responses returned by the
prediction and informational endpoints.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SHAPExplanation(BaseModel):
    """Single SHAP feature-importance explanation."""

    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="SHAP value (absolute impact magnitude)")
    direction: str = Field(
        ..., description="'increase' or 'decrease' relative to PD"
    )


class PredictionResponse(BaseModel):
    """Loan risk prediction response.

    Contains the probability of default, a risk-segment label,
    and SHAP-based explanations for the top contributing features.
    """

    probability_of_default: float = Field(
        ..., ge=0, le=1, description="Probability of Default (0-1)"
    )
    risk_segment: str = Field(
        ..., description="Risk category: Low, Medium, or High"
    )
    shap_explanations: List[SHAPExplanation] = Field(
        ...,
        description="Top 5 most impactful features sorted by absolute SHAP value",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "probability_of_default": 0.185,
                    "risk_segment": "Medium",
                    "shap_explanations": [
                        {
                            "feature": "Num_Moras_Previas",
                            "impact": 0.12,
                            "direction": "increase",
                        },
                        {
                            "feature": "Ingresos_Anuales",
                            "impact": 0.08,
                            "direction": "decrease",
                        },
                        {
                            "feature": "DTI",
                            "impact": 0.06,
                            "direction": "increase",
                        },
                        {
                            "feature": "Tipo_Prestamo",
                            "impact": 0.04,
                            "direction": "decrease",
                        },
                        {
                            "feature": "Tasa_Interes",
                            "impact": 0.03,
                            "direction": "increase",
                        },
                    ],
                }
            ]
        }
    }


class ModelInfoResponse(BaseModel):
    """Metadata about the loaded ML model and service configuration."""

    app_version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether the XGBoost model is loaded")
    boosting_rounds: Optional[int] = Field(
        None, description="Number of boosting rounds in the loaded model"
    )
    feature_names: List[str] = Field(
        ..., description="Feature names the model expects, in order"
    )
    scaler_loaded: bool = Field(
        ..., description="Whether the StandardScaler is loaded"
    )
    encoder_loaded: bool = Field(
        ..., description="Whether the categorical encoder is loaded"
    )
    model_artifact_date: Optional[str] = Field(
        None, description="Last modified date of the model file (ISO 8601)"
    )
    risk_thresholds: Dict[str, float] = Field(
        ..., description="PD thresholds for Low/Medium/High classification"
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Error description")
    error_type: Optional[str] = Field(default=None, description="Exception type name")
    request_id: Optional[str] = Field(
        default=None, description="X-Request-ID for log correlation"
    )
