"""Categorical encoding service for inference.

Encodes categorical feature values from the API request into the
integer representation expected by the trained XGBoost model.

The mapping dictionaries **must** remain consistent with the
``LabelEncoder`` order used during training.

Typical usage::

    from app.services.encoder import CategoricalEncoder

    X = CategoricalEncoder.encode_request(request.model_dump())
    # X has shape (1, n_features) and dtype np.float32
"""

import logging
from typing import Any, Dict, List

import numpy as np

from app.core.features import FEATURE_ORDER
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

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """Encodes categorical API request fields to integers for the model.

    Each class-level mapping dictionary converts an enum value to the
    same integer that ``TrainingEncoder`` (via ``LabelEncoder``) assigns
    during training.

    .. note::
        These mappings are hand-coded to mirror the ``LabelEncoder``
        order (alphabetical within each category by training data values).
        If the training data changes, both must be updated together.
    """

    # Mapping: enum value → integer code (MUST match LabelEncoder order)
    GENERO_MAP: Dict[GeneroEnum, int] = {
        GeneroEnum.MUJER: 0,
        GeneroEnum.HOMBRE: 1,
        GeneroEnum.OTRO: 2,
    }

    ESTADO_CIVIL_MAP: Dict[EstadoCivilEnum, int] = {
        EstadoCivilEnum.SOLTERO: 0,
        EstadoCivilEnum.CASADO: 1,
        EstadoCivilEnum.DIVORCIADO: 2,
        EstadoCivilEnum.VIUDO: 3,
    }

    EDUCACION_MAP: Dict[EducacionEnum, int] = {
        EducacionEnum.SIN_ESTUDIOS: 0,
        EducacionEnum.PRIMARIA: 1,
        EducacionEnum.SECUNDARIA: 2,
        EducacionEnum.BACHILLERATO: 3,
        EducacionEnum.FORMACION_PROFESIONAL: 4,
        EducacionEnum.GRADO: 5,
        EducacionEnum.POSGRADO: 6,
    }

    SITUACION_LABORAL_MAP: Dict[SituacionLaboralEnum, int] = {
        SituacionLaboralEnum.INDEFINIDO: 0,
        SituacionLaboralEnum.TEMPORAL: 1,
        SituacionLaboralEnum.AUTONOMO: 2,
        SituacionLaboralEnum.FUNCIONARIO: 3,
        SituacionLaboralEnum.DESEMPLEADO: 4,
        SituacionLaboralEnum.INACTIVO: 5,
    }

    SECTOR_TRABAJO_MAP: Dict[SectorTrabajoEnum, int] = {
        SectorTrabajoEnum.SECTOR_PUBLICO: 0,
        SectorTrabajoEnum.SALUD: 1,
        SectorTrabajoEnum.EDUCACION: 2,
        SectorTrabajoEnum.HOSTELERIA: 3,
        SectorTrabajoEnum.VENTAS: 4,
        SectorTrabajoEnum.TECNOLOGIA: 5,
        SectorTrabajoEnum.CONSTRUCCION: 6,
        SectorTrabajoEnum.AGROPECUARIO: 7,
        SectorTrabajoEnum.OTROS: 8,
    }

    VIVIENDA_MAP: Dict[ViviendaEnum, int] = {
        ViviendaEnum.PROPIA_PAGADA: 0,
        ViviendaEnum.PROPIA_HIPOTECA: 1,
        ViviendaEnum.ALQUILER: 2,
        ViviendaEnum.CEDIDA: 3,
    }

    TIENE_HIPOTECA_MAP: Dict[TieneHipotecaEnum, int] = {
        TieneHipotecaEnum.SI: 1,
        TieneHipotecaEnum.NO: 0,
    }

    TIPO_PRESTAMO_MAP: Dict[TipoPrestamoEnum, int] = {
        TipoPrestamoEnum.HIPOTECARIO: 0,
        TipoPrestamoEnum.PERSONAL: 1,
        TipoPrestamoEnum.AUTO: 2,
        TipoPrestamoEnum.CONSUMO: 3,
    }

    PROPOSITO_MAP: Dict[PropositoEnum, int] = {
        PropositoEnum.COMPRA_VIVIENDA: 0,
        PropositoEnum.REFORMA_VIVIENDA: 1,
        PropositoEnum.COMPRA_VEHICULO: 2,
        PropositoEnum.REFORMA_HOGAR: 3,
        PropositoEnum.EDUCACION: 4,
        PropositoEnum.SALUD: 5,
        PropositoEnum.CONSOLIDACION_DEUDA: 6,
        PropositoEnum.VIAJES: 7,
        PropositoEnum.OTROS: 8,
    }

    @staticmethod
    def encode_request(request_data: Dict[str, Any]) -> np.ndarray:
        """Encode a loan application request dict into a feature array.

        Maps every field in ``request_data`` to its numeric representation
        and assembles the result in ``FEATURE_ORDER`` so it matches the
        column order expected by the XGBoost model.

        Args:
            request_data: Dictionary of request fields as produced by
                ``LoanApplicationRequest.model_dump()``.

        Returns:
            ``np.ndarray`` of shape ``(1, n_features)`` and dtype
            ``np.float32``, ready to be passed to the model.

        Raises:
            KeyError: If a required field is absent from ``request_data``
                or its value is not in the corresponding mapping dict.
        """
        encoded: Dict[str, Any] = {
            "Edad": request_data.get("edad"),
            "Genero": CategoricalEncoder.GENERO_MAP[request_data.get("genero")],
            "Estado_Civil": CategoricalEncoder.ESTADO_CIVIL_MAP[request_data.get("estado_civil")],
            "Educacion": CategoricalEncoder.EDUCACION_MAP[request_data.get("educacion")],
            "Situacion_Laboral": CategoricalEncoder.SITUACION_LABORAL_MAP[
                request_data.get("situacion_laboral")
            ],
            "Sector_Trabajo": CategoricalEncoder.SECTOR_TRABAJO_MAP[
                request_data.get("sector_trabajo")
            ],
            "Dependientes": request_data.get("dependientes"),
            "Vivienda": CategoricalEncoder.VIVIENDA_MAP[request_data.get("vivienda")],
            "Tiene_Hipoteca": CategoricalEncoder.TIENE_HIPOTECA_MAP[
                request_data.get("tiene_hipoteca")
            ],
            "Ingresos_Anuales": request_data.get("ingresos_anuales"),
            "Tipo_Prestamo": CategoricalEncoder.TIPO_PRESTAMO_MAP[
                request_data.get("tipo_prestamo")
            ],
            "Proposito": CategoricalEncoder.PROPOSITO_MAP[request_data.get("proposito")],
            "Monto_Prestamo": request_data.get("monto_prestamo"),
            "Plazo_Meses": request_data.get("plazo_meses"),
            "Tasa_Interes": request_data.get("tasa_interes"),
            "LTV": request_data.get("ltv"),
            "DTI": request_data.get("dti"),
            "Num_Prestamos_Previos": request_data.get("num_prestamos_previos"),
            "Num_Moras_Previas": request_data.get("num_moras_previas"),
        }

        feature_vector = [encoded[feature] for feature in FEATURE_ORDER]
        return np.array([feature_vector], dtype=np.float32)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return feature names in the model's expected column order.

        Returns:
            List of feature name strings matching ``FEATURE_ORDER``.
        """
        return list(FEATURE_ORDER)
