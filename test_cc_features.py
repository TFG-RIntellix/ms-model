import pandas as pd
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test the dataset and feature mapping
print("Testing credit card dataset and feature mapping...")

df = pd.read_csv('resources/dataset_tarjeta_credito_with_noise.csv', sep=';')
print(f"Original columns: {list(df.columns)}")

# Test column mapping
column_mapping = {
    "Edad": "age",
    "Situacion_Laboral": "employmentStatus",
    "Antiguedad_Laboral": "workSeniority",
    "Ingresos_Anuales": "annualIncome",
    "Tipo_Ingreso": "incomeType",
    "Vivienda": "housing",
    "Dependientes": "dependents",
    "Limite_Credito": "creditLimit",
    "Es_Revolving": "isRevolving",
    "Tasa_Interes": "interestRate",
    "Ratio_Limite_Ingreso": "creditLimitToIncomeRatio",
    "DTI": "dti",
    "Impagos_Previos": "previousDefaults",
    "PD_Estimada": "estimatedPD",
    "Default": "Default",
}

df_mapped = df.rename(columns=column_mapping)
print(f"Mapped columns: {list(df_mapped.columns)}")

# Check if all required features are present
from app.core.features import CREDIT_CARD_FEATURE_ORDER, CREDIT_CARD_TARGET_COLUMN

required = CREDIT_CARD_FEATURE_ORDER + [CREDIT_CARD_TARGET_COLUMN]
missing = [col for col in required if col not in df_mapped.columns]

if missing:
    print(f"Missing columns: {missing}")
else:
    print(f"✓ All required columns present!")

print(f"\nDataset shape: {df_mapped.shape}")
print(f"Target distribution:\n{df_mapped['Default'].value_counts()}")
