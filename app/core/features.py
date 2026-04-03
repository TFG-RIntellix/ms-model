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
    "Edad",
    "Genero",
    "Estado_Civil",
    "Educacion",
    "Situacion_Laboral",
    "Sector_Trabajo",
    "Dependientes",
    "Vivienda",
    "Tiene_Hipoteca",
    "Ingresos_Anuales",
    "Tipo_Prestamo",
    "Proposito",
    "Monto_Prestamo",
    "Plazo_Meses",
    "Tasa_Interes",
    "LTV",
    "DTI",
    "Num_Prestamos_Previos",
    "Num_Moras_Previas",
]

# ---------------------------------------------------------------------------
# Categorical features and their allowed values
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "Genero": ["Mujer", "Hombre", "Otro"],
    "Estado_Civil": ["Soltero", "Casado", "Divorciado", "Viudo"],
    "Educacion": [
        "Sin Estudios",
        "Primaria",
        "Secundaria",
        "Bachillerato",
        "Formacion Profesional",
        "Grado",
        "Posgrado",
    ],
    "Situacion_Laboral": [
        "Indefinido",
        "Temporal",
        "Autonomo",
        "Funcionario",
        "Desempleado",
        "Inactivo",
    ],
    "Sector_Trabajo": [
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
    "Vivienda": ["Propia_Pagada", "Propia_Hipoteca", "Alquiler", "Cedida"],
    "Tiene_Hipoteca": ["Si", "No"],
    "Tipo_Prestamo": ["Hipotecario", "Personal", "Auto", "Consumo"],
    "Proposito": [
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

TARGET_COLUMN: str = "Target_PD"
