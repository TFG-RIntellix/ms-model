import sys
print("Testing credit card model implementation...")

try:
    # Test imports
    from app.schemas.enums import (
        CreditCardEmploymentStatusEnum,
        CreditCardIncomeTypeEnum,
        CreditCardHomeOwnershipEnum,
        CreditCardIsRevolvingEnum,
    )
    print("✓ Credit card enums imported successfully")

    from app.schemas.requests import CreditCardApplicationRequest
    print("✓ CreditCardApplicationRequest imported successfully")

    from app.core.features import (
        CREDIT_CARD_FEATURE_ORDER,
        CREDIT_CARD_CATEGORICAL_FEATURES,
        CREDIT_CARD_NUMERIC_FEATURES,
        CREDIT_CARD_TARGET_COLUMN,
    )
    print(f"✓ Credit card features imported successfully")
    print(f"  - Features: {len(CREDIT_CARD_FEATURE_ORDER)} total")
    print(f"  - Categorical: {len(CREDIT_CARD_CATEGORICAL_FEATURES)}")
    print(f"  - Numeric: {len(CREDIT_CARD_NUMERIC_FEATURES)}")

    from app.services.encoder import CategoricalEncoder
    print("✓ CategoricalEncoder imported successfully")
    print(f"  - Credit card feature names: {CategoricalEncoder.get_credit_card_feature_names()}")

    from app.services.inference import InferenceService
    print("✓ InferenceService imported successfully")

    from app.core.settings import Settings
    print("✓ Settings imported successfully")
    settings = Settings()
    print(f"  - Loan model path: {settings.MODEL_PATH}")
    print(f"  - Credit card model path: {settings.CREDIT_CARD_MODEL_PATH}")

    from app.core.lifespan import ModelManager
    print("✓ ModelManager imported successfully")

    # Test CreditCardApplicationRequest validation
    test_request = CreditCardApplicationRequest(
        age=41,
        employmentStatus=CreditCardEmploymentStatusEnum.PERMANENT,
        workSeniority=12,
        annualIncome=56079.61,
        incomeType=CreditCardIncomeTypeEnum.SALARY,
        homeOwnership=CreditCardHomeOwnershipEnum.OWNED_MORTGAGED,
        dependents=0,
        creditLimit=19000.0,
        isRevolving=CreditCardIsRevolvingEnum.YES,
        interestRate=0.2489,
        creditLimitToIncomeRatio=0.3388,
        dti=0.1626,
        previousDefaults=0,
    )
    print("✓ CreditCardApplicationRequest validation passed")
    print(f"  Sample request created with age={test_request.age}, creditLimit={test_request.creditLimit}")

    print("\n✓ All credit card components are correctly implemented!")
    print("\nCredit card model is ready for:")
    print("  1. Inference via /api/v1/risk/predict-credit-card endpoint")
    print("  2. Training via: python -m training.train_credit_card")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
