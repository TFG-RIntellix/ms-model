"""Shared pytest fixtures for the Credit Risk Engine test suite.

Centralises common test data and setup so individual test modules
do not duplicate fixture definitions.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.enums import (
    GeneroEnum,
    EstadoCivilEnum,
    EducacionEnum,
    SituacionLaboralEnum,
    SectorTrabajoEnum,
    ViviendaEnum,
    TieneHipotecaEnum,
    TipoPrestamoEnum,
    PropositoEnum,
)
from app.schemas.requests import LoanApplicationRequest


# ------------------------------------------------------------------
# FastAPI TestClient
# ------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a ``TestClient`` for the FastAPI application."""
    return TestClient(app)


# ------------------------------------------------------------------
# Sample payloads / objects
# ------------------------------------------------------------------

@pytest.fixture
def sample_loan_payload() -> dict:
    """Sample loan application payload (JSON-serialisable dict)."""
    return {
        "edad": 35,
        "genero": "Mujer",
        "estado_civil": "Casado",
        "educacion": "Grado",
        "situacion_laboral": "Indefinido",
        "sector_trabajo": "Tecnologia",
        "dependientes": 2,
        "vivienda": "Propia_Hipoteca",
        "tiene_hipoteca": "Si",
        "ingresos_anuales": 45000.0,
        "tipo_prestamo": "Personal",
        "proposito": "Consolidacion_Deuda",
        "monto_prestamo": 15000.0,
        "plazo_meses": 36,
        "tasa_interes": 5.5,
        "ltv": 0.45,
        "dti": 0.35,
        "num_prestamos_previos": 1,
        "num_moras_previas": 0,
    }


@pytest.fixture
def sample_loan_request() -> LoanApplicationRequest:
    """Create a fully-typed ``LoanApplicationRequest`` instance."""
    return LoanApplicationRequest(
        edad=35,
        genero=GeneroEnum.MUJER,
        estado_civil=EstadoCivilEnum.CASADO,
        educacion=EducacionEnum.GRADO,
        situacion_laboral=SituacionLaboralEnum.INDEFINIDO,
        sector_trabajo=SectorTrabajoEnum.TECNOLOGIA,
        dependientes=2,
        vivienda=ViviendaEnum.PROPIA_HIPOTECA,
        tiene_hipoteca=TieneHipotecaEnum.SI,
        ingresos_anuales=45000.0,
        tipo_prestamo=TipoPrestamoEnum.PERSONAL,
        proposito=PropositoEnum.CONSOLIDACION_DEUDA,
        monto_prestamo=15000.0,
        plazo_meses=36,
        tasa_interes=5.5,
        ltv=0.45,
        dti=0.35,
        num_prestamos_previos=1,
        num_moras_previas=0,
    )


@pytest.fixture
def mock_model_manager():
    """Create a mock ``ModelManager`` with a stub model."""
    manager = Mock()
    manager.loan_model = Mock()
    manager.encoder = None
    manager.scaler = None
    return manager
