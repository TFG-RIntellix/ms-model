"""Enumeration types for categorical credit risk features.

Each enum maps to the allowed values in the training dataset.
API consumers must submit exactly one of the listed values for
each categorical field.
"""

from enum import Enum


class GeneroEnum(str, Enum):
    """Gender categories."""

    MUJER = "Mujer"
    HOMBRE = "Hombre"
    OTRO = "Otro"


class EstadoCivilEnum(str, Enum):
    """Marital status categories."""

    SOLTERO = "Soltero"
    CASADO = "Casado"
    DIVORCIADO = "Divorciado"
    VIUDO = "Viudo"


class EducacionEnum(str, Enum):
    """Education level categories (ordinal)."""

    SIN_ESTUDIOS = "Sin Estudios"
    PRIMARIA = "Primaria"
    SECUNDARIA = "Secundaria"
    BACHILLERATO = "Bachillerato"
    FORMACION_PROFESIONAL = "Formacion Profesional"
    GRADO = "Grado"
    POSGRADO = "Posgrado"


class SituacionLaboralEnum(str, Enum):
    """Employment situation categories."""

    INDEFINIDO = "Indefinido"
    TEMPORAL = "Temporal"
    AUTONOMO = "Autonomo"
    FUNCIONARIO = "Funcionario"
    DESEMPLEADO = "Desempleado"
    INACTIVO = "Inactivo"


class SectorTrabajoEnum(str, Enum):
    """Work sector categories."""

    SECTOR_PUBLICO = "Sector Publico"
    SALUD = "Salud"
    EDUCACION = "Educacion"
    HOSTELERIA = "Hosteleria"
    VENTAS = "Ventas"
    TECNOLOGIA = "Tecnologia"
    CONSTRUCCION = "Construccion"
    AGROPECUARIO = "Agropecuario"
    OTROS = "Otros"


class ViviendaEnum(str, Enum):
    """Housing type categories."""

    PROPIA_PAGADA = "Propia_Pagada"
    PROPIA_HIPOTECA = "Propia_Hipoteca"
    ALQUILER = "Alquiler"
    CEDIDA = "Cedida"


class TieneHipotecaEnum(str, Enum):
    """Mortgage status (binary)."""

    SI = "Si"
    NO = "No"


class TienePrestamoEnum(str, Enum):
    """Existing loan status (binary)."""

    SI = "Si"
    NO = "No"


class TipoPrestamoEnum(str, Enum):
    """Loan type categories."""

    HIPOTECARIO = "Hipotecario"
    PERSONAL = "Personal"
    AUTO = "Auto"
    CONSUMO = "Consumo"


class PropositoEnum(str, Enum):
    """Loan purpose categories."""

    COMPRA_VIVIENDA = "Compra_Vivienda"
    REFORMA_VIVIENDA = "Reforma_Vivienda"
    COMPRA_VEHICULO = "Compra_Vehiculo"
    REFORMA_HOGAR = "Reforma_Hogar"
    EDUCACION = "Educacion"
    SALUD = "Salud"
    CONSOLIDACION_DEUDA = "Consolidacion_Deuda"
    VIAJES = "Viajes"
    OTROS = "Otros"
