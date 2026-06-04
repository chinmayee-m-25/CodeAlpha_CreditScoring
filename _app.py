import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Credit Scoring Model", layout="wide")
st.title("Credit Scoring Model")
st.markdown("---")

@st.cache_data
def load_and_train():
    train_df = pd.read_excel('credit_risk_dataset.csv/train-FIN_ANA_DATA .xls')

    train_df = train_df.drop(columns=['ACC_NO'])
    train_df['INSTALL_SIZE']  = train_df['INSTALL_SIZE'].fillna(train_df['INSTALL_SIZE'].median())
    train_df['CLIENT_TYPE']   = train_df['CLIENT_TYPE'].fillna(train_df['CLIENT_TYPE'].mode()[0])
    for col in ['INF_MARITAL_STATUS', 'INF_GENDER', 'COMPENSATION_CHARGED']:
        train_df[col] = train_df[col].fillna(train_df[col].mode()[0])

    train_df['BALANCE_TO_INVEST'] = train_df['ACCCURRENTBALANCE'] / (train_df['INVESTMENT_TOTAL'] + 1)
    train_df['PAYMENT_BURDEN']    = train_df['INSTALL_SIZE'] / (train_df['DUE_PAYMENT'] + 1)
    train_df['IS_HIGH_BALANCE']   = (train_df['ACCCURRENTBALANCE'] > 68348).astype(int)

    cat_cols = ['INF_MARITAL_STATUS', 'INF_GENDER',
                'COMPENSATION_CHARGED', 'CLIENT_TYPE', 'QUALITY_OF_LOAN']
    le = LabelEncoder()
    for col in cat_cols:
        train_df[col] = le.fit_transform(train_df[col].astype(str))

    train_df['REPAY_MODE'] = train_df['REPAY_MODE'].map({'N': 0, 'I': 1})

    X = train_df.drop(columns=['REPAY_MODE'])
    y = train_df['REPAY_MODE']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree'      : DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for m in models.values():
        m.fit(X_train, y_train)

    return models, X_train, X_val, y_train, y_val

with st.spinner("Loading model and data..."):
    models, X_train, X_val, y_train, y_val = load_and_train()
    best_model = models['Random Forest']

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction",
    "Feature Importance",
    "Model Comparison",
    "ROC Curve"
])

with tab1:
    st.subheader("Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:
        INVESTMENT_TOTAL     = st.number_input("Investment Total",    value=10000)
        ACCCURRENTBALANCE    = st.number_input("Current Balance",     value=5000)
        INF_MARITAL_STATUS   = st.selectbox("Marital Status",         [0, 1], format_func=lambda x: "Single" if x==0 else "Married")
        INF_GENDER           = st.selectbox("Gender",                 [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        INSTALL_SIZE         = st.number_input("Instalment Size",     value=500)

    with col2:
        DUE_PAYMENT          = st.number_input("Due Payment",         value=200)
        COMPENSATION_CHARGED = st.selectbox("Compensation Charged",   [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        CLIENT_TYPE          = st.number_input("Client Type",         value=1)
        QUALITY_OF_LOAN      = st.selectbox("Quality of Loan",        [0, 1], format_func=lambda x: "Bad" if x==0 else "Good")

    if st.button("Check Credit Risk", use_container_width=True):
        input_data = {
            'INVESTMENT_TOTAL'    : INVESTMENT_TOTAL,
            'ACCCURRENTBALANCE'   : ACCCURRENTBALANCE,
            'INF_MARITAL_STATUS'  : INF_MARITAL_STATUS,
            'INF_GENDER'          : INF_GENDER,
            'INSTALL_SIZE'        : INSTALL_SIZE,
            'DUE_PAYMENT'         : DUE_PAYMENT,
            'COMPENSATION_CHARGED': COMPENSATION_CHARGED,
            'CLIENT_TYPE'         : CLIENT_TYPE,
            'QUALITY_OF_LOAN'     : QUALITY_OF_LOAN
        }
        input_df = pd.DataFrame([input_data])
        input_df['BALANCE_TO_INVEST'] = input_df['ACCCURRENTBALANCE'] / (input_df['INVESTMENT_TOTAL'] + 1)
        input_df['PAYMENT_BURDEN']    = input_df['INSTALL_SIZE'] / (input_df['DUE_PAYMENT'] + 1)
        input_df['IS_HIGH_BALANCE']   = (input_df['ACCCURRENTBALANCE'] > 68348).astype(int)

        prediction  = best_model.predict(input_df)[0]
        probability = best_model.predict_proba(input_df)[0][1] * 100

        st.markdown("---")
        if prediction == 0:
            st.success("LOW RISK — NO DEFAULT")
        else:
            st.error("HIGH RISK — DEFAULT LIKELY")
        st.metric("Default Probability", f"{round(probability, 2)}%")

with tab2:
    st.subheader("Feature Importance — Random Forest")
    st.write("Shows which features most influence the credit default prediction.")

    importance = best_model.feature_importances_
    features   = X_train.columns.tolist()
    fi_df      = pd.DataFrame({'Feature': features, 'Importance': importance})
    fi_df      = fi_df.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ['#2196F3' if i < len(fi_df)-3 else '#F44336'
               for i in range(len(fi_df))]
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    st.caption("Red bars = top 3 most important features")

with tab3:
    st.subheader("Model Comparison")
    st.write("Comparison of all 3 models across key metrics.")

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        results[name] = {
            'Accuracy' : round(accuracy_score(y_val, y_pred)  * 100, 2),
            'Precision': round(precision_score(y_val, y_pred) * 100, 2),
            'Recall'   : round(recall_score(y_val, y_pred)    * 100, 2),
            'F1 Score' : round(f1_score(y_val, y_pred)        * 100, 2),
            'ROC-AUC'  : round(roc_auc_score(y_val, y_prob)   * 100, 2)
        }

    results_df = pd.DataFrame(results).T
    st.dataframe(results_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(results_df.columns))
    width  = 0.25
    colors = ['#2196F3', '#4CAF50', '#F44336']

    for i, (name, row) in enumerate(results_df.iterrows()):
        ax.bar(x + i * width, row.values, width,
               label=name, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df.columns)
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.set_ylim(90, 101)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with tab4:
    st.subheader("ROC Curve")
    st.write("Higher the curve, better the model at distinguishing default vs no default.")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ['#2196F3', '#4CAF50', '#F44336']

    for (name, model), color in zip(models.items(), colors):
        y_prob      = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc         = round(roc_auc_score(y_val, y_prob), 4)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc})', color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — All Models')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    st.caption("AUC closer to 1.0 = better model performance")
