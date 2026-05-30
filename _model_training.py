import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

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

# ── 4. Feature Engineering ────────────────────────────────────
train_df['BALANCE_TO_INVEST'] = train_df['ACCCURRENTBALANCE'] / (train_df['INVESTMENT_TOTAL'] + 1)
test_df['BALANCE_TO_INVEST']  = test_df['ACCCURRENTBALANCE']  / (test_df['INVESTMENT_TOTAL'] + 1)

train_df['PAYMENT_BURDEN'] = train_df['INSTALL_SIZE'] / (train_df['DUE_PAYMENT'] + 1)
test_df['PAYMENT_BURDEN']  = test_df['INSTALL_SIZE']  / (test_df['DUE_PAYMENT'] + 1)

train_df['IS_HIGH_BALANCE'] = (train_df['ACCCURRENTBALANCE'] > train_df['ACCCURRENTBALANCE'].median()).astype(int)
test_df['IS_HIGH_BALANCE']  = (test_df['ACCCURRENTBALANCE']  > test_df['ACCCURRENTBALANCE'].median()).astype(int)

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

# ── 9. Train 3 models ─────────────────────────────────────────
print("Training models...\n")

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
lr_acc  = accuracy_score(y_val, lr_pred)
print("Logistic Regression  Accuracy:", round(lr_acc * 100, 2), "%")

# Model 2: Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_val)
dt_acc  = accuracy_score(y_val, dt_pred)
print("Decision Tree        Accuracy:", round(dt_acc * 100, 2), "%")

# Model 3: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_acc  = accuracy_score(y_val, rf_pred)
print("Random Forest        Accuracy:", round(rf_acc * 100, 2), "%")

# ── 10. Save best model ───────────────────────────────────────
best_acc   = max(lr_acc, dt_acc, rf_acc)
best_name  = ['Logistic Regression','Decision Tree','Random Forest'][[lr_acc, dt_acc, rf_acc].index(best_acc)]
best_model = [lr, dt, rf][[lr_acc, dt_acc, rf_acc].index(best_acc)]

joblib.dump(best_model, 'best_model.pkl')
print("\nBest model:", best_name, "→ saved as best_model.pkl")
print("\nStep 4 - Model Training Complete!")
