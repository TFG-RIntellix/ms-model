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
    GENDER_MAP: Dict[GenderEnum, int] = {
        GenderEnum.FEMALE: 0,
        GenderEnum.MALE: 1,
        GenderEnum.OTHER: 2,
    }

    MARITAL_STATUS_MAP: Dict[MaritalStatusEnum, int] = {
        MaritalStatusEnum.SINGLE: 0,
        MaritalStatusEnum.MARRIED: 1,
        MaritalStatusEnum.DIVORCED: 2,
        MaritalStatusEnum.WIDOWED: 3,
    }

    EDUCATION_MAP: Dict[EducationEnum, int] = {
        EducationEnum.NO_STUDIES: 0,
        EducationEnum.PRIMARY: 1,
        EducationEnum.SECONDARY: 2,
        EducationEnum.HIGH_SCHOOL: 3,
        EducationEnum.VOCATIONAL: 4,
        EducationEnum.DEGREE: 5,
        EducationEnum.POSTGRADUATE: 6,
    }

    EMPLOYMENT_STATUS_MAP: Dict[EmploymentStatusEnum, int] = {
        EmploymentStatusEnum.PERMANENT: 0,
        EmploymentStatusEnum.TEMPORARY: 1,
        EmploymentStatusEnum.SELF_EMPLOYED: 2,
        EmploymentStatusEnum.CIVIL_SERVANT: 3,
        EmploymentStatusEnum.UNEMPLOYED: 4,
        EmploymentStatusEnum.INACTIVE: 5,
    }

    OCCUPATION_SECTOR_MAP: Dict[OccupationSectorEnum, int] = {
        OccupationSectorEnum.PUBLIC_SECTOR: 0,
        OccupationSectorEnum.HEALTHCARE: 1,
        OccupationSectorEnum.EDUCATION: 2,
        OccupationSectorEnum.HOSPITALITY: 3,
        OccupationSectorEnum.SALES: 4,
        OccupationSectorEnum.TECHNOLOGY: 5,
        OccupationSectorEnum.CONSTRUCTION: 6,
        OccupationSectorEnum.AGRICULTURE: 7,
        OccupationSectorEnum.OTHER: 8,
    }

    HOME_OWNERSHIP_MAP: Dict[HomeOwnershipEnum, int] = {
        HomeOwnershipEnum.OWNED_PAID: 0,
        HomeOwnershipEnum.OWNED_MORTGAGED: 1,
        HomeOwnershipEnum.RENTED: 2,
        HomeOwnershipEnum.CEDED: 3,
    }

    HAS_MORTGAGE_MAP: Dict[HasMortgageEnum, int] = {
        HasMortgageEnum.YES: 1,
        HasMortgageEnum.NO: 0,
    }

    LOAN_TYPE_MAP: Dict[LoanTypeEnum, int] = {
        LoanTypeEnum.MORTGAGE: 0,
        LoanTypeEnum.PERSONAL: 1,
        LoanTypeEnum.AUTO: 2,
        LoanTypeEnum.CONSUMER: 3,
    }

    LOAN_PURPOSE_MAP: Dict[LoanPurposeEnum, int] = {
        LoanPurposeEnum.HOME_PURCHASE: 0,
        LoanPurposeEnum.HOME_RENOVATION: 1,
        LoanPurposeEnum.VEHICLE_PURCHASE: 2,
        LoanPurposeEnum.HOME_IMPROVEMENT: 3,
        LoanPurposeEnum.EDUCATION: 4,
        LoanPurposeEnum.HEALTH: 5,
        LoanPurposeEnum.DEBT_CONSOLIDATION: 6,
        LoanPurposeEnum.TRAVEL: 7,
        LoanPurposeEnum.OTHER: 8,
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
            "age": request_data.get("age"),
            "gender": CategoricalEncoder.GENDER_MAP[request_data.get("gender")],
            "maritalStatus": CategoricalEncoder.MARITAL_STATUS_MAP[request_data.get("maritalStatus")],
            "education": CategoricalEncoder.EDUCATION_MAP[request_data.get("education")],
            "employmentStatus": CategoricalEncoder.EMPLOYMENT_STATUS_MAP[
                request_data.get("employmentStatus")
            ],
            "occupationSector": CategoricalEncoder.OCCUPATION_SECTOR_MAP[
                request_data.get("occupationSector")
            ],
            "dependents": request_data.get("dependents"),
            "homeOwnership": CategoricalEncoder.HOME_OWNERSHIP_MAP[request_data.get("homeOwnership")],
            "hasMortgage": CategoricalEncoder.HAS_MORTGAGE_MAP[
                request_data.get("hasMortgage")
            ],
            "annualIncome": request_data.get("annualIncome"),
            "loanType": CategoricalEncoder.LOAN_TYPE_MAP[
                request_data.get("loanType")
            ],
            "purpose": CategoricalEncoder.LOAN_PURPOSE_MAP[request_data.get("purpose")],
            "loanAmount": request_data.get("loanAmount"),
            "termMonths": request_data.get("termMonths"),
            "interestRate": request_data.get("interestRate"),
            "LTV": request_data.get("LTV"),
            "DTI": request_data.get("DTI"),
            "previousLoansCount": request_data.get("previousLoansCount"),
            "previousDefaultsCount": request_data.get("previousDefaultsCount"),
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
