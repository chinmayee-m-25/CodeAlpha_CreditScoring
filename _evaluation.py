import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix)
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

# ── 5. Encode ─────────────────────────────────────────────────
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

# ── 6. Split ──────────────────────────────────────────────────
X = train_df.drop(columns=['REPAY_MODE'])
y = train_df['REPAY_MODE']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2, random_state=42
)

# ── 7. Train all 3 models ─────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)

# ── 8. Evaluate all models ────────────────────────────────────
print("=" * 60)
print("        MODEL EVALUATION REPORT")
print("=" * 60)

for name, model in models.items():
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print(f"\n--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_val, y_pred) * 100, 2), "%")
    print("Precision:", round(precision_score(y_val, y_pred) * 100, 2), "%")
    print("Recall   :", round(recall_score(y_val, y_pred) * 100, 2), "%")
    print("F1 Score :", round(f1_score(y_val, y_pred) * 100, 2), "%")
    print("ROC-AUC  :", round(roc_auc_score(y_val, y_prob), 4))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred,
          target_names=['No Default', 'Default']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

print("=" * 60)
print("\nStep 5 - Evaluation Complete!")
