import pandas as pd

# Load credit card dataset
cc_df = pd.read_csv('resources/dataset_tarjeta_credito_with_noise.csv', sep=';')

print('=== CREDIT CARD DATASET ANALYSIS ===')
print(f'\nShape: {cc_df.shape}')
print(f'\nColumns: {list(cc_df.columns)}')
print(f'\nData types:\n{cc_df.dtypes}')

# Analyze categorical features
categorical_cols = ['Situacion_Laboral', 'Tipo_Ingreso', 'Vivienda', 'Es_Revolving']
print('\n=== CATEGORICAL VALUES ===')
for col in categorical_cols:
    unique_vals = cc_df[col].unique()
    print(f'{col}: {sorted(unique_vals.tolist())}')

# Analyze numeric features
print('\n=== NUMERIC FEATURES STATISTICS ===')
numeric_cols = ['Edad', 'Antiguedad_Laboral', 'Ingresos_Anuales', 'Limite_Credito', 'Tasa_Interes', 'Ratio_Limite_Ingreso', 'DTI', 'Impagos_Previos', 'PD_Estimada']
print(cc_df[numeric_cols].describe())

print(f'\n=== TARGET DISTRIBUTION ===')
print(cc_df['Default'].value_counts())
