import pandas as pd

# Load train and test data (correct path and extension)
train_df = pd.read_excel('credit_risk_dataset.csv/train-FIN_ANA_DATA .xls')
test_df  = pd.read_excel('credit_risk_dataset.csv/test-FIN_ANA_DATA .xls')

# Shape
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# Column names
print("\nColumns:")
print(train_df.columns.tolist())

# First 5 rows
print("\nFirst 5 rows:")
print(train_df.head())

# Missing values
print("\nMissing values:")
print(train_df.isnull().sum())

# Data types
print("\nData types:")
print(train_df.dtypes)

# Target column distribution
print("\nTarget value counts:")
print(train_df.iloc[:, -1].value_counts())
