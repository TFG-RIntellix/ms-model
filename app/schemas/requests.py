"""Pydantic v2 request models for the Credit Risk Engine API.

Each model enforces strict validation on incoming JSON bodies,
matching the constraints of the training dataset.
"""

from pydantic import BaseModel, Field

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


class LoanApplicationRequest(BaseModel):
    """Loan application request model.

    Enforces strict validation matching the training dataset
    feature definitions and value ranges.
    """

    age: int = Field(..., ge=18, le=80, description="Age (18-80)")
    gender: GenderEnum = Field(..., description="Gender")
    maritalStatus: MaritalStatusEnum = Field(..., description="Marital status")
    education: EducationEnum = Field(..., description="Education level")
    employmentStatus: EmploymentStatusEnum = Field(
        ..., description="Employment status"
    )
    occupationSector: OccupationSectorEnum = Field(..., description="Work sector")
    dependents: int = Field(
        ..., ge=0, le=5, description="Number of dependents (0-5)"
    )
    homeOwnership: HomeOwnershipEnum = Field(..., description="Housing type")
    hasMortgage: HasMortgageEnum = Field(..., description="Has mortgage")
    annualIncome: float = Field(
        ..., gt=0, description="Annual income (must be positive)"
    )
    loanType: LoanTypeEnum = Field(..., description="Loan type")
    purpose: LoanPurposeEnum = Field(..., description="Loan purpose")
    loanAmount: float = Field(
        ..., gt=0, description="Loan amount (must be positive)"
    )
    termMonths: int = Field(..., gt=0, description="Loan term in months")
    interestRate: float = Field(..., ge=0, description="Interest rate")
    ltv: float = Field(..., ge=0, le=1, description="Loan-to-Value ratio (0-1)")
    dti: float = Field(..., ge=0, le=1, description="Debt-to-Income ratio (0-1)")
    previousLoansCount: int = Field(
        ..., ge=0, description="Number of previous loans"
    )
    previousDefaultsCount: int = Field(
        ..., ge=0, description="Number of previous defaults"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class CreditCardApplicationRequest(BaseModel):
    """Credit card application request model.

    Placeholder schema — features pending definition.

    Note:
        This schema will be expanded with credit-card-specific features
        such as credit history, spending patterns, and card type
        preferences once the business requirements are finalised.
    """

    age: int = Field(..., ge=18, le=80, description="Age")
    gender: GenderEnum = Field(..., description="Gender")
    annualIncome: float = Field(..., gt=0, description="Annual income")
