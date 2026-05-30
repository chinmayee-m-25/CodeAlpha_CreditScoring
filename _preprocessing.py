import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── 1. Load data ──────────────────────────────────────────────
train_df = pd.read_excel('credit_risk_dataset.csv/train-FIN_ANA_DATA .xls')
test_df  = pd.read_excel('credit_risk_dataset.csv/test-FIN_ANA_DATA .xls')

# ── 2. Drop ACC_NO ────────────────────────────────────────────
train_df = train_df.drop(columns=['ACC_NO'])
test_df  = test_df.drop(columns=['ACC_NO'])

# ── 3. Handle missing values ──────────────────────────────────
train_df['INSTALL_SIZE'] = train_df['INSTALL_SIZE'].fillna(train_df['INSTALL_SIZE'].median())
test_df['INSTALL_SIZE']  = test_df['INSTALL_SIZE'].fillna(test_df['INSTALL_SIZE'].median())

train_df['CLIENT_TYPE'] = train_df['CLIENT_TYPE'].fillna(train_df['CLIENT_TYPE'].mode()[0])
test_df['CLIENT_TYPE']  = test_df['CLIENT_TYPE'].fillna(test_df['CLIENT_TYPE'].mode()[0])

for col in ['INF_MARITAL_STATUS', 'INF_GENDER', 'COMPENSATION_CHARGED']:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col]  = test_df[col].fillna(test_df[col].mode()[0])

print("Missing values after fix:")
print(train_df.isnull().sum())

# ── 4. Convert all categorical columns to string first ────────
cat_cols = ['INF_MARITAL_STATUS', 'INF_GENDER',
            'COMPENSATION_CHARGED', 'CLIENT_TYPE',
            'QUALITY_OF_LOAN']

for col in cat_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col]  = test_df[col].astype(str)

# ── 5. Encode categorical columns ─────────────────────────────
le = LabelEncoder()
for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col]  = le.fit_transform(test_df[col])

# Encode target
train_df['REPAY_MODE'] = train_df['REPAY_MODE'].map({'N': 0, 'I': 1})
test_df['REPAY_MODE']  = test_df['REPAY_MODE'].map({'N': 0, 'I': 1})

print("\nAfter encoding:")
print(train_df.head())

# ── 6. Split features and target ──────────────────────────────
X = train_df.drop(columns=['REPAY_MODE'])
y = train_df['REPAY_MODE']

print("\nClass distribution before SMOTE:")
print(y.value_counts())

# ── 7. Handle class imbalance using SMOTE ─────────────────────
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# ── 8. Train-Validation split ─────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)

print("\nFinal shapes:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)

print("\nStep 2 - Preprocessing Complete!")
