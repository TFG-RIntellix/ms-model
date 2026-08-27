"""Core constants for the application.

Centralises magic strings and fixed values to prevent typos
and improve maintainability.
"""

from enum import Enum


class RiskSegment(str, Enum):
    """Risk segments derived from Probability of Default (PD)."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SHAPDirection(str, Enum):
    """Direction of a feature's impact according to SHAP."""
    INCREASE = "increase"
    DECREASE = "decrease"


class ErrorMessages:
    """Standard internal error messages."""
    MODEL_MANAGER_UNINITIALIZED = "Model manager not initialized"
    
    LOAN_MODEL_NOT_LOADED = "Loan model not loaded"
    LOAN_EXPLAINER_NOT_LOADED = "Explainer not loaded"
    
    CREDIT_CARD_MODEL_NOT_LOADED = "Credit card model not loaded"
    CREDIT_CARD_ENCODER_NOT_LOADED = "Credit card encoder not loaded"
    CREDIT_CARD_EXPLAINER_NOT_LOADED = "Credit card explainer not loaded"
    
    SHAP_CALCULATION_ERROR = "SHAP calculation error"
