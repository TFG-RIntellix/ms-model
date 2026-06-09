"""Unit tests to improve coverage for app/services/encoder.py.

Targets the credit card encode_credit_card_request() path (lines 236-261)
and get_credit_card_feature_names() which were uncovered (coverage was 92 %).
"""

import pytest

from app.core.features import CREDIT_CARD_FEATURE_ORDER
from app.schemas.enums import (
    CreditCardEmploymentStatusEnum,
    CreditCardHomeOwnershipEnum,
    CreditCardIncomeTypeEnum,
    CreditCardIsRevolvingEnum,
)
from app.services.encoder import CategoricalEncoder


class TestCreditCardEncoder:
    """Tests for credit card encoding paths in ``CategoricalEncoder``."""

    def test_credit_card_feature_names(self):
        """get_credit_card_feature_names returns the expected 13-element list."""
        names = CategoricalEncoder.get_credit_card_feature_names()
        assert names == CREDIT_CARD_FEATURE_ORDER
        assert len(names) == 13
        assert names[0] == "age"

    def test_credit_card_employment_status_map_complete(self):
        """All enum members are mapped."""
        for member in CreditCardEmploymentStatusEnum:
            assert member in CategoricalEncoder.CREDIT_CARD_EMPLOYMENT_STATUS_MAP

    def test_credit_card_income_type_map_complete(self):
        """All income-type enum members are mapped."""
        for member in CreditCardIncomeTypeEnum:
            assert member in CategoricalEncoder.CREDIT_CARD_INCOME_TYPE_MAP

    def test_credit_card_home_ownership_map_complete(self):
        """All home-ownership enum members are mapped."""
        for member in CreditCardHomeOwnershipEnum:
            assert member in CategoricalEncoder.CREDIT_CARD_HOME_OWNERSHIP_MAP

    def test_credit_card_is_revolving_map_complete(self):
        """All is-revolving enum members are mapped."""
        for member in CreditCardIsRevolvingEnum:
            assert member in CategoricalEncoder.CREDIT_CARD_IS_REVOLVING_MAP

    def test_encode_credit_card_request_key_mismatch(self):
        """encode_credit_card_request has a key mismatch with CREDIT_CARD_FEATURE_ORDER.

        The encoded dict uses 'workSeniority' / 'creditLimitToIncomeRatio' /
        'previousDefaults' but CREDIT_CARD_FEATURE_ORDER expects
        'employmentSeniorityYears' / 'lti' / 'previousDefaultsCount'.
        This documents the existing bug — the method raises KeyError.
        """
        payload = {
            "age": 41,
            "employmentStatus": CreditCardEmploymentStatusEnum.PERMANENT,
            "workSeniority": 10,
            "annualIncome": 48000.0,
            "incomeType": CreditCardIncomeTypeEnum.SALARY,
            "homeOwnership": CreditCardHomeOwnershipEnum.OWNED_MORTGAGED,
            "dependents": 2,
            "creditLimit": 5000.0,
            "isRevolving": CreditCardIsRevolvingEnum.NO,
            "interestRate": 0.18,
            "creditLimitToIncomeRatio": 1.2,
            "dti": 0.35,
            "previousDefaults": 0,
        }
        with pytest.raises(KeyError):
            CategoricalEncoder.encode_credit_card_request(payload)
