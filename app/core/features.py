"""Centralized feature definitions for the Credit Risk Engine.

Single source of truth for feature names, order, and categorical
categories used by both the training pipeline and inference service.
All consumers (training, encoding, inference, tests) MUST import
from this module to avoid drift.

Typical usage example::

    from app.core.features import FEATURE_ORDER, CATEGORICAL_FEATURES, NUMERIC_FEATURES
"""

from typing import Dict, List


# ---------------------------------------------------------------------------
# Feature ordering — CRITICAL: must match model training column order
# ---------------------------------------------------------------------------

FEATURE_ORDER: List[str] = [
    "age",
    "gender",
    "maritalStatus",
    "education",
    "employmentStatus",
    "occupationSector",
    "dependents",
    "homeOwnership",
    "hasMortgage",
    "annualIncome",
    "loanType",
    "purpose",
    "loanAmount",
    "termMonths",
    "interestRate",
    "ltv",
    "dti",
    "previousLoansCount",
    "previousDefaultsCount",
]

# ---------------------------------------------------------------------------
# Categorical features and their allowed values
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "gender": ["Mujer", "Hombre", "Otro"],
    "maritalStatus": ["Soltero", "Casado", "Divorciado", "Viudo"],
    "education": [
        "Sin Estudios",
        "Primaria",
        "Secundaria",
        "Bachillerato",
        "Formacion Profesional",
        "Grado",
        "Posgrado",
    ],
    "employmentStatus": [
        "Indefinido",
        "Temporal",
        "Autonomo",
        "Funcionario",
        "Desempleado",
        "Inactivo",
    ],
    "occupationSector": [
        "Sector Publico",
        "Salud",
        "Educacion",
        "Hosteleria",
        "Ventas",
        "Tecnologia",
        "Construccion",
        "Agropecuario",
        "Otros",
    ],
    "homeOwnership": ["Propia_Pagada", "Propia_Hipoteca", "Alquiler", "Cedida"],
    "hasMortgage": ["Si", "No"],
    "loanType": ["Hipotecario", "Personal", "Auto", "Consumo"],
    "purpose": [
        "Compra_Vivienda",
        "Reforma_Vivienda",
        "Compra_Vehiculo",
        "Reforma_Hogar",
        "Educacion",
        "Salud",
        "Consolidacion_Deuda",
        "Viajes",
        "Otros",
    ],
}

# ---------------------------------------------------------------------------
# Numeric features (all features that are NOT categorical)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: List[str] = [
    f for f in FEATURE_ORDER if f not in CATEGORICAL_FEATURES
]

# ---------------------------------------------------------------------------
# Target column name
# ---------------------------------------------------------------------------

TARGET_COLUMN: str = "target_PD"
