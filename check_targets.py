import pandas as pd

# Check loans dataset target
df_loans = pd.read_csv('resources/dataset_credito_with_noise_default.csv', sep=';')
print("Loans target column (default):")
print(f"  Type: {df_loans['default'].dtype}")
print(f"  Unique values: {df_loans['default'].unique()[:5]}")
print(f"  Sample:\n{df_loans[['default']].head()}")

# Check credit card dataset target
df_cc = pd.read_csv('resources/dataset_tarjeta_credito_with_noise.csv', sep=';')
print("\n\nCredit Card target column (Default):")
print(f"  Type: {df_cc['Default'].dtype}")
print(f"  Unique values: {df_cc['Default'].unique()}")
print(f"  Value counts:\n{df_cc['Default'].value_counts()}")
