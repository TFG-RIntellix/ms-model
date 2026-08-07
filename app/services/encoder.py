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

from app.core.features import FEATURE_ORDER, CREDIT_CARD_FEATURE_ORDER
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
    CreditCardEmploymentStatusEnum,
    CreditCardIncomeTypeEnum,
    CreditCardHomeOwnershipEnum,
    CreditCardIsRevolvingEnum,
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

    # Mapping: enum value → integer code (must match the fitted TrainingEncoder)
    GENDER_MAP: Dict[GenderEnum, int] = {
        GenderEnum.MALE: 0,
        GenderEnum.FEMALE: 1,
        GenderEnum.OTHER: 2,
    }

    MARITAL_STATUS_MAP: Dict[MaritalStatusEnum, int] = {
        MaritalStatusEnum.MARRIED: 0,
        MaritalStatusEnum.DIVORCED: 1,
        MaritalStatusEnum.SINGLE: 2,
        MaritalStatusEnum.WIDOWED: 3,
    }

    EDUCATION_MAP: Dict[EducationEnum, int] = {
        EducationEnum.HIGH_SCHOOL: 0,
        EducationEnum.VOCATIONAL: 1,
        EducationEnum.DEGREE: 2,
        EducationEnum.POSTGRADUATE: 3,
        EducationEnum.PRIMARY: 4,
        EducationEnum.SECONDARY: 5,
        EducationEnum.NO_STUDIES: 6,
    }

    EMPLOYMENT_STATUS_MAP: Dict[EmploymentStatusEnum, int] = {
        EmploymentStatusEnum.SELF_EMPLOYED: 0,
        EmploymentStatusEnum.UNEMPLOYED: 1,
        EmploymentStatusEnum.CIVIL_SERVANT: 2,
        EmploymentStatusEnum.INACTIVE: 3,
        EmploymentStatusEnum.PERMANENT: 4,
        EmploymentStatusEnum.TEMPORARY: 5,
    }

    OCCUPATION_SECTOR_MAP: Dict[OccupationSectorEnum, int] = {
        OccupationSectorEnum.AGRICULTURE: 0,
        OccupationSectorEnum.CONSTRUCTION: 1,
        OccupationSectorEnum.EDUCATION: 2,
        OccupationSectorEnum.HOSPITALITY: 3,
        OccupationSectorEnum.OTHER: 4,
        OccupationSectorEnum.HEALTHCARE: 5,
        OccupationSectorEnum.PUBLIC_SECTOR: 6,
        OccupationSectorEnum.TECHNOLOGY: 7,
        OccupationSectorEnum.SALES: 8,
    }

    HOME_OWNERSHIP_MAP: Dict[HomeOwnershipEnum, int] = {
        HomeOwnershipEnum.RENTED: 0,
        HomeOwnershipEnum.CEDED: 1,
        HomeOwnershipEnum.OWNED_MORTGAGED: 2,
        HomeOwnershipEnum.OWNED_PAID: 3,
    }

    HAS_MORTGAGE_MAP: Dict[HasMortgageEnum, int] = {
        HasMortgageEnum.YES: 1,
        HasMortgageEnum.NO: 0,
    }

    LOAN_TYPE_MAP: Dict[LoanTypeEnum, int] = {
        LoanTypeEnum.AUTO: 0,
        LoanTypeEnum.CONSUMER: 1,
        LoanTypeEnum.MORTGAGE: 2,
        LoanTypeEnum.PERSONAL: 3,
    }

    LOAN_PURPOSE_MAP: Dict[LoanPurposeEnum, int] = {
        LoanPurposeEnum.VEHICLE_PURCHASE: 0,
        LoanPurposeEnum.HOME_PURCHASE: 1,
        LoanPurposeEnum.DEBT_CONSOLIDATION: 2,
        LoanPurposeEnum.EDUCATION: 3,
        LoanPurposeEnum.HOME_IMPROVEMENT: 6,
        LoanPurposeEnum.OTHER: 5,
        LoanPurposeEnum.HOME_RENOVATION: 7,
        LoanPurposeEnum.HEALTH: 8,
        LoanPurposeEnum.TRAVEL: 10,
    }

    # =====================================================================
    # Credit Card Specific Mappings
    # =====================================================================

    CREDIT_CARD_EMPLOYMENT_STATUS_MAP: Dict[CreditCardEmploymentStatusEnum, int] = {
        CreditCardEmploymentStatusEnum.SELF_EMPLOYED: 0,
        CreditCardEmploymentStatusEnum.UNEMPLOYED: 1,
        CreditCardEmploymentStatusEnum.CIVIL_SERVANT: 2,
        CreditCardEmploymentStatusEnum.INACTIVE: 3,
        CreditCardEmploymentStatusEnum.PERMANENT: 4,
        CreditCardEmploymentStatusEnum.TEMPORARY: 5,
    }

    CREDIT_CARD_INCOME_TYPE_MAP: Dict[CreditCardIncomeTypeEnum, int] = {
        CreditCardIncomeTypeEnum.SALARY: 0,
        CreditCardIncomeTypeEnum.SELF_EMPLOYED: 1,
        CreditCardIncomeTypeEnum.PENSION: 2,
        CreditCardIncomeTypeEnum.BENEFITS: 3,
    }

    CREDIT_CARD_HOME_OWNERSHIP_MAP: Dict[CreditCardHomeOwnershipEnum, int] = {
        CreditCardHomeOwnershipEnum.RENTED: 0,
        CreditCardHomeOwnershipEnum.CEDED: 1,
        CreditCardHomeOwnershipEnum.OWNED_MORTGAGED: 2,
        CreditCardHomeOwnershipEnum.OWNED_PAID: 3,
    }

    CREDIT_CARD_IS_REVOLVING_MAP: Dict[CreditCardIsRevolvingEnum, int] = {
        CreditCardIsRevolvingEnum.NO: 0,
        CreditCardIsRevolvingEnum.YES: 1,
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
            "ltv": request_data.get("ltv"),
            "dti": request_data.get("dti"),
            "previousLoansCount": request_data.get("previousLoansCount"),
            "previousDefaultsCount": request_data.get("previousDefaultsCount"),
        }

        feature_vector = [encoded[feature] for feature in FEATURE_ORDER]
        return np.array([feature_vector], dtype=np.float32)

    @staticmethod
    def encode_credit_card_request(request_data: Dict[str, Any]) -> np.ndarray:
        """Encode a credit card application request dict into a feature array.

        Maps every field in ``request_data`` to its numeric representation
        and assembles the result in ``CREDIT_CARD_FEATURE_ORDER`` so it matches
        the column order expected by the XGBoost model.

        Args:
            request_data: Dictionary of request fields as produced by
                ``CreditCardApplicationRequest.model_dump()``.

        Returns:
            ``np.ndarray`` of shape ``(1, n_features)`` and dtype
            ``np.float32``, ready to be passed to the model.

        Raises:
            KeyError: If a required field is absent from ``request_data``
                or its value is not in the corresponding mapping dict.
        """
        encoded: Dict[str, Any] = {
            "age": request_data.get("age"),
            "employmentStatus": CategoricalEncoder.CREDIT_CARD_EMPLOYMENT_STATUS_MAP[
                request_data.get("employmentStatus")
            ],
            "workSeniority": request_data.get("workSeniority"),
            "annualIncome": request_data.get("annualIncome"),
            "incomeType": CategoricalEncoder.CREDIT_CARD_INCOME_TYPE_MAP[
                request_data.get("incomeType")
            ],
            "homeOwnership": CategoricalEncoder.CREDIT_CARD_HOME_OWNERSHIP_MAP[
                request_data.get("homeOwnership")
            ],
            "dependents": request_data.get("dependents"),
            "creditLimit": request_data.get("creditLimit"),
            "isRevolving": CategoricalEncoder.CREDIT_CARD_IS_REVOLVING_MAP[
                request_data.get("isRevolving")
            ],
            "interestRate": request_data.get("interestRate"),
            "creditLimitToIncomeRatio": request_data.get("creditLimitToIncomeRatio"),
            "dti": request_data.get("dti"),
            "previousDefaults": request_data.get("previousDefaults"),
        }

        feature_vector = [encoded[feature] for feature in CREDIT_CARD_FEATURE_ORDER]
        return np.array([feature_vector], dtype=np.float32)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return feature names in the model's expected column order.

        Returns:
            List of feature name strings matching ``FEATURE_ORDER``.
        """
        return list(FEATURE_ORDER)

    @staticmethod
    def get_credit_card_feature_names() -> List[str]:
        """Return credit card feature names in the model's expected column order.

        Returns:
            List of feature name strings matching ``CREDIT_CARD_FEATURE_ORDER``.
        """
        return list(CREDIT_CARD_FEATURE_ORDER)
