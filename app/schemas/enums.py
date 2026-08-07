"""Enumeration types for categorical credit risk features.

Each enum maps to the allowed values in the training dataset.
API consumers must submit exactly one of the listed values for
each categorical field.
"""

from enum import Enum


class GenderEnum(str, Enum):
    """Gender categories."""

    FEMALE = "Mujer"
    MALE = "Hombre"
    OTHER = "Otro"


class MaritalStatusEnum(str, Enum):
    """Marital status categories."""

    SINGLE = "Soltero"
    MARRIED = "Casado"
    DIVORCED = "Divorciado"
    WIDOWED = "Viudo"


class EducationEnum(str, Enum):
    """Education level categories (ordinal)."""

    NO_STUDIES = "Sin Estudios"
    PRIMARY = "Primaria"
    SECONDARY = "Secundaria"
    HIGH_SCHOOL = "Bachillerato"
    VOCATIONAL = "Formacion Profesional"
    DEGREE = "Grado"
    POSTGRADUATE = "Posgrado"


class EmploymentStatusEnum(str, Enum):
    """Employment situation categories."""

    PERMANENT = "Indefinido"
    TEMPORARY = "Temporal"
    SELF_EMPLOYED = "Autonomo"
    CIVIL_SERVANT = "Funcionario"
    UNEMPLOYED = "Desempleado"
    INACTIVE = "Inactivo"


class OccupationSectorEnum(str, Enum):
    """Work sector categories."""

    PUBLIC_SECTOR = "Sector Publico"
    HEALTHCARE = "Salud"
    EDUCATION = "Educacion"
    HOSPITALITY = "Hosteleria"
    SALES = "Ventas"
    TECHNOLOGY = "Tecnologia"
    CONSTRUCTION = "Construccion"
    AGRICULTURE = "Agropecuario"
    OTHER = "Otros"


class HomeOwnershipEnum(str, Enum):
    """Housing type categories."""

    OWNED_PAID = "Propia_Pagada"
    OWNED_MORTGAGED = "Propia_Hipoteca"
    RENTED = "Alquiler"
    CEDED = "Cedida"


class HasMortgageEnum(str, Enum):
    """Mortgage status (binary)."""

    YES = "Si"
    NO = "No"


class LoanTypeEnum(str, Enum):
    """Loan type categories."""

    MORTGAGE = "Hipotecario"
    PERSONAL = "Personal"
    AUTO = "Auto"
    CONSUMER = "Consumo"


class LoanPurposeEnum(str, Enum):
    """Loan purpose categories."""

    HOME_PURCHASE = "Compra_Vivienda"
    HOME_RENOVATION = "Reforma_Vivienda"
    VEHICLE_PURCHASE = "Compra_Vehiculo"
    HOME_IMPROVEMENT = "Reforma_Hogar"
    EDUCATION = "Educacion"
    HEALTH = "Salud"
    DEBT_CONSOLIDATION = "Consolidacion_Deuda"
    TRAVEL = "Viajes"
    OTHER = "Otros"


# ---------------------------------------------------------------------------
# Credit Card Specific Enums
# ---------------------------------------------------------------------------


class CreditCardEmploymentStatusEnum(str, Enum):
    """Employment situation categories (for credit cards)."""

    PERMANENT = "Indefinido"
    TEMPORARY = "Temporal"
    SELF_EMPLOYED = "Autonomo"
    CIVIL_SERVANT = "Funcionario"
    UNEMPLOYED = "Desempleado"
    INACTIVE = "Inactivo"


class CreditCardIncomeTypeEnum(str, Enum):
    """Income type categories (for credit cards)."""

    SALARY = "Salario"
    SELF_EMPLOYED = "Autonomo"
    PENSION = "Pension"
    BENEFITS = "Ayudas"


class CreditCardHomeOwnershipEnum(str, Enum):
    """Housing type categories (for credit cards)."""

    OWNED_PAID = "Propia_Pagada"
    OWNED_MORTGAGED = "Propia_Hipoteca"
    RENTED = "Alquiler"
    CEDED = "Cedida"


class CreditCardIsRevolvingEnum(str, Enum):
    """Revolving credit status."""

    YES = "Si"
    NO = "No"
