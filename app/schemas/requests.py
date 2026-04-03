"""Pydantic v2 request models for the Credit Risk Engine API.

Each model enforces strict validation on incoming JSON bodies,
matching the constraints of the training dataset.
"""

from pydantic import BaseModel, Field

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


class LoanApplicationRequest(BaseModel):
    """Loan application request model.

    Enforces strict validation matching the training dataset
    feature definitions and value ranges.
    """

    edad: int = Field(..., ge=18, le=80, description="Age (18-80)")
    genero: GeneroEnum = Field(..., description="Gender")
    estado_civil: EstadoCivilEnum = Field(..., description="Marital status")
    educacion: EducacionEnum = Field(..., description="Education level")
    situacion_laboral: SituacionLaboralEnum = Field(
        ..., description="Employment status"
    )
    sector_trabajo: SectorTrabajoEnum = Field(..., description="Work sector")
    dependientes: int = Field(
        ..., ge=0, le=5, description="Number of dependents (0-5)"
    )
    vivienda: ViviendaEnum = Field(..., description="Housing type")
    tiene_hipoteca: TieneHipotecaEnum = Field(..., description="Has mortgage")
    ingresos_anuales: float = Field(
        ..., gt=0, description="Annual income (must be positive)"
    )
    tipo_prestamo: TipoPrestamoEnum = Field(..., description="Loan type")
    proposito: PropositoEnum = Field(..., description="Loan purpose")
    monto_prestamo: float = Field(
        ..., gt=0, description="Loan amount (must be positive)"
    )
    plazo_meses: int = Field(..., gt=0, description="Loan term in months")
    tasa_interes: float = Field(..., ge=0, description="Interest rate")
    ltv: float = Field(..., ge=0, le=1, description="Loan-to-Value ratio (0-1)")
    dti: float = Field(..., ge=0, le=1, description="Debt-to-Income ratio (0-1)")
    num_prestamos_previos: int = Field(
        ..., ge=0, description="Number of previous loans"
    )
    num_moras_previas: int = Field(
        ..., ge=0, description="Number of previous defaults"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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

    edad: int = Field(..., ge=18, le=80, description="Age")
    genero: GeneroEnum = Field(..., description="Gender")
    ingresos_anuales: float = Field(..., gt=0, description="Annual income")
