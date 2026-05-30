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

# ── 3. Fill missing values ────────────────────────────────────
train_df['INSTALL_SIZE'] = train_df['INSTALL_SIZE'].fillna(train_df['INSTALL_SIZE'].median())
test_df['INSTALL_SIZE']  = test_df['INSTALL_SIZE'].fillna(test_df['INSTALL_SIZE'].median())

train_df['CLIENT_TYPE'] = train_df['CLIENT_TYPE'].fillna(train_df['CLIENT_TYPE'].mode()[0])
test_df['CLIENT_TYPE']  = test_df['CLIENT_TYPE'].fillna(test_df['CLIENT_TYPE'].mode()[0])

for col in ['INF_MARITAL_STATUS', 'INF_GENDER', 'COMPENSATION_CHARGED']:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col]  = test_df[col].fillna(test_df[col].mode()[0])

# ── 4. Feature Engineering (new columns) ─────────────────────
# Balance to investment ratio
train_df['BALANCE_TO_INVEST'] = train_df['ACCCURRENTBALANCE'] / (train_df['INVESTMENT_TOTAL'] + 1)
test_df['BALANCE_TO_INVEST']  = test_df['ACCCURRENTBALANCE']  / (test_df['INVESTMENT_TOTAL'] + 1)

# Payment burden: instalment vs due payment
train_df['PAYMENT_BURDEN'] = train_df['INSTALL_SIZE'] / (train_df['DUE_PAYMENT'] + 1)
test_df['PAYMENT_BURDEN']  = test_df['INSTALL_SIZE']  / (test_df['DUE_PAYMENT'] + 1)

# Is high balance flag
train_df['IS_HIGH_BALANCE'] = (train_df['ACCCURRENTBALANCE'] > train_df['ACCCURRENTBALANCE'].median()).astype(int)
test_df['IS_HIGH_BALANCE']  = (test_df['ACCCURRENTBALANCE']  > test_df['ACCCURRENTBALANCE'].median()).astype(int)

print("New features added:")
print(train_df[['BALANCE_TO_INVEST', 'PAYMENT_BURDEN', 'IS_HIGH_BALANCE']].head())
print("\nTotal columns now:", train_df.shape[1])

# ── 5. Encode categorical columns ─────────────────────────────
cat_cols = ['INF_MARITAL_STATUS', 'INF_GENDER',
            'COMPENSATION_CHARGED', 'CLIENT_TYPE',
            'QUALITY_OF_LOAN']

for col in cat_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col]  = test_df[col].astype(str)

le = LabelEncoder()
for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col]  = le.fit_transform(test_df[col])

# Encode target
train_df['REPAY_MODE'] = train_df['REPAY_MODE'].map({'N': 0, 'I': 1})
test_df['REPAY_MODE']  = test_df['REPAY_MODE'].map({'N': 0, 'I': 1})

# ── 6. Split features and target ──────────────────────────────
X = train_df.drop(columns=['REPAY_MODE'])
y = train_df['REPAY_MODE']

# ── 7. SMOTE ──────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ── 8. Train-Validation split ─────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)

print("\nFinal shapes after feature engineering:")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)

print("\nAll features:")
print(X_train.columns.tolist())

print("\nStep 3 - Feature Engineering Complete!")
