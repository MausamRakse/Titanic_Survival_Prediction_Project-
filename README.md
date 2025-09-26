# Titanic Survival Prediction

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)  

---

## 🚢 Project Overview

This project builds a complete machine learning pipeline to **predict passenger survival on the Titanic**, using the well-known Titanic dataset. Starting from data cleaning and exploratory data analysis (EDA), the project trains and evaluates classification models (Logistic Regression and Support Vector Machine), performs hyperparameter tuning, and finally wraps the prediction logic into a simple web application using Streamlit (or Flask).

The goal is to illustrate end-to-end workflow:  
- Data understanding & preprocessing  
- Feature engineering & encoding  
- Model building, tuning & evaluation  
- Deployment of a predictive interface  

---

## 📁 Repository Structure

Titanic_Survival_Prediction_Project/
│
├── README.md # Project overview (this file)
├── Titanic_Survival_Prediction.ipynb # Main analysis & modeling notebook
├── project_report.md # Markdown report summarizing findings
├── project_report.pdf # PDF version of the report
├── app.py # Web app for prediction (Streamlit / Flask)
├── requirements.txt # Required Python packages
├── summary.txt # Short summary / notes
├── models/ # (Optional) Saved model files & scalers
│ ├── logistic_model.pkl
│ ├── svm_model.pkl
│ └── scaler.pkl
└── images/ # (Optional) Visualization assets
└── ...


---

## 📦 Installation & Setup

### Prerequisites

- Python 3.8 or newer  
- `pip` (Python package manager)  

### Setup Steps

1. Clone the repository:  
   ```bash
   git clone https://github.com/MausamRakse/Titanic_Survival_Prediction_Project-.git
   cd Titanic_Survival_Prediction_Project-
   
pip install -r requirements.txt


streamlit run app.py


📊 Key Findings & Results
Metric	Logistic Regression	SVM	Tuned LR	Tuned SVM
Accuracy	(your value)	(your value)	(your value)	(your value)
Precision	...	...	...	...
Recall	...	...	...	...
F1-Score	...	...	...	...

Overall survival rate: ~ 38.38%

Gender effect: Females had a significantly higher survival rate than males

Class effect: 1st class passengers had higher survival probability

Age effect: Younger passengers (children) had improved survival

After hyperparameter tuning, improvements of ~1–2% in accuracy were observed

🔧 Methodology & Technical Details

Missing Value Treatment: Impute Age using median or predictive methods; Embarked mode imputation

Feature Engineering: Derived features such as Title (Mr/Mrs/Miss/Master), FamilySize, IsAlone

Encoding: One-hot encoding for categorical variables; label encoding for target if needed

Scaling: StandardScaler applied to numerical features

Models:

Logistic Regression (with regularization)

Support Vector Machine (linear / RBF kernel)

Hyperparameter Tuning: GridSearchCV or RandomizedSearchCV

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, (optional ROC-AUC)

🚀 Future Enhancements & Improvements

Try additional algorithms (Random Forest, XGBoost, Gradient Boosting)

More advanced feature engineering (interaction terms, polynomial features)

Cross-validation and ensemble methods

Dockerize the application for easy deployment

Add unit tests, logging, and CI/CD pipelines

Web UI improvements (better input validation, UX)
