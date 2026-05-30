# Credit Scoring Model — CodeAlpha Internship

## Project Overview
An end-to-end Machine Learning project to predict 
whether a loan applicant will default or not — 
just like real banks do!

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib & Seaborn
- Streamlit

## Project Structure
| File | Description |
|---|---|
| _data_collection.py | Load and inspect data |
| _preprocessing.py | Clean and balance data |
| _feature_engineering.py | Create new features |
| _model_training.py | Train 3 ML models |
| _evaluation.py | Evaluate model performance |
| _app.py | Streamlit web app |

## Model Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 97.25% | 0.9947 |
| Decision Tree | 99.58% | 0.9995 |
| Random Forest | 99.83% | 0.9999 |

## How to Run
pip install pandas scikit-learn imbalanced-learn streamlit matplotlib seaborn xlrd
streamlit run _app.py

## Internship
CodeAlpha Machine Learning Internship — Task 1
