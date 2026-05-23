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
    GenderEnum,
    MaritalStatusEnum,
    EducationEnum,
    EmploymentStatusEnum,
    OccupationSectorEnum,
    HomeOwnershipEnum,
    HasMortgageEnum,
    LoanTypeEnum,
    LoanPurposeEnum,
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
        "age": 35,
        "gender": "Mujer",
        "maritalStatus": "Casado",
        "education": "Grado",
        "employmentStatus": "Indefinido",
        "occupationSector": "Tecnologia",
        "dependents": 2,
        "homeOwnership": "Propia_Hipoteca",
        "hasMortgage": "Si",
        "annualIncome": 45000.0,
        "loanType": "Personal",
        "purpose": "Consolidacion_Deuda",
        "loanAmount": 15000.0,
        "termMonths": 36,
        "interestRate": 5.5,
        "ltv": 0.45,
        "dti": 0.35,
        "previousLoansCount": 1,
        "previousDefaultsCount": 0,
    }


@pytest.fixture
def sample_loan_request() -> LoanApplicationRequest:
    """Create a fully-typed ``LoanApplicationRequest`` instance."""
    return LoanApplicationRequest(
        age=35,
        gender=GenderEnum.FEMALE,
        maritalStatus=MaritalStatusEnum.MARRIED,
        education=EducationEnum.DEGREE,
        employmentStatus=EmploymentStatusEnum.PERMANENT,
        occupationSector=OccupationSectorEnum.TECHNOLOGY,
        dependents=2,
        homeOwnership=HomeOwnershipEnum.OWNED_MORTGAGED,
        hasMortgage=HasMortgageEnum.YES,
        annualIncome=45000.0,
        loanType=LoanTypeEnum.PERSONAL,
        purpose=LoanPurposeEnum.DEBT_CONSOLIDATION,
        loanAmount=15000.0,
        termMonths=36,
        interestRate=5.5,
        ltv=0.45,
        dti=0.35,
        previousLoansCount=1,
        previousDefaultsCount=0,
    )


@pytest.fixture
def mock_model_manager():
    """Create a mock ``ModelManager`` with a stub model."""
    manager = Mock()
    manager.loan_model = Mock()
    manager.encoder = None
    manager.scaler = None
    return manager
