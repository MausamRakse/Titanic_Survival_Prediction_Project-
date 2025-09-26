# Titanic_Survival_Prediction_Project-
Titanic Survival Prediction 🚢
A comprehensive machine learning project that predicts passenger survival on the Titanic using Logistic Regression and Support Vector Machine (SVM) algorithms.

https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Scikit--learn-1.0%252B-orange
https://img.shields.io/badge/Streamlit-1.12%252B-red
https://img.shields.io/badge/License-MIT-green

📊 Project Overview
This project demonstrates a complete machine learning pipeline from data preprocessing to model deployment. The analysis compares two classification algorithms to predict whether a passenger survived the Titanic disaster based on various features like passenger class, age, gender, and fare.

Key Features:

Data preprocessing and feature engineering

Exploratory Data Analysis (EDA) with visualizations

Model training and hyperparameter tuning

Interactive web application for predictions

Comprehensive model evaluation

🎯 Results Summary
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	81.56%	78.95%	75.47%	77.17%
SVM	82.12%	80.36%	75.47%	77.85%
Tuned Logistic Regression	83.24%	80.36%	79.25%	79.80%
Tuned SVM	84.36%	82.08%	79.25%	80.65%
📁 Project Structure
text
titanic-survival-prediction/
│
├── 📓 titanic_survival_prediction.ipynb    # Main Jupyter Notebook
├── 📄 project_report.md                    # Detailed analysis report
├── 🚀 app.py                               # Streamlit web application
├── 📋 requirements.txt                     # Python dependencies
├── 📖 README.md                            # This file
│
├── 🤖 models/                              # Trained model files
│   ├── logistic_regression_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl
│
└── 📊 images/                              # Visualization assets
    └── eda_visualizations.png
🚀 Quick Start
Prerequisites
Python 3.8+

pip (Python package manager)

Installation
Clone the repository

bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
Install dependencies

bash
pip install -r requirements.txt
Usage
Option 1: Run the Jupyter Notebook

bash
jupyter notebook titanic_survival_prediction.ipynb
Option 2: Launch the Web Application

bash
streamlit run app.py
Option 3: Run as Python Script

bash
python -c "
import titanic_survival_prediction as tsp
tsp.run_analysis()
"
📈 Key Findings
Data Insights
Overall survival rate: 38.38%

Gender disparity: 74.2% females vs 18.9% males survived

Class impact: 62.96% 1st class vs 24.24% 3rd class survived

Age factor: Children had significantly higher survival rates

Model Performance
SVM slightly outperforms Logistic Regression after tuning

Both models achieve >80% accuracy

Hyperparameter tuning improved accuracy by 1.5-2.5%

Important Features
Gender (strongest predictor)

Passenger Class

Fare

Title (Mr, Mrs, Miss, Master)

Age

🛠️ Technical Details
Algorithms Used
Logistic Regression: Linear model with L1/L2 regularization

Support Vector Machine (SVM): With linear and RBF kernels

Preprocessing Steps
Missing value imputation (Age, Embarked)

Feature engineering (FamilySize, IsAlone, Title encoding)

Categorical variable encoding (One-hot, Label encoding)

Feature scaling (StandardScaler)

Evaluation Metrics
Accuracy, Precision, Recall, F1-Score

Confusion Matrix

ROC-AUC Curve

🌐 Web Application
The Streamlit app provides an interactive interface to:

Input passenger details

Select between Logistic Regression and SVM models

View survival probability with confidence scores

Understand feature importance

Access the app locally:

bash
streamlit run app.py
Then open http://localhost:8501 in your browser.

📊 Sample Usage
Using the Prediction Function
python
from models import load_models
import numpy as np

# Load trained models
lr_model, svm_model, scaler = load_models()

# Prepare passenger data
passenger_data = {
    'pclass': 1,
    'sex': 'female', 
    'age': 25,
    'sibsp': 0,
    'parch': 0,
    'fare': 50,
    'embarked': 'C'
}

# Make prediction
survival_prob = predict_survival(passenger_data, lr_model, scaler)
print(f"Survival probability: {survival_prob:.2%}")
🤝 Contributing
Contributions are welcome! Here's how you can help:

Fork the repository

Create a feature branch

bash
git checkout -b feature/amazing-feature
Commit your changes

bash
git commit -m 'Add some amazing feature'
Push to the branch

bash
git push origin feature/amazing-feature
Open a Pull Request

Areas for Improvement
Implement additional algorithms (Random Forest, XGBoost)

Add more sophisticated feature engineering

Create Docker container for easy deployment

Add unit tests and CI/CD pipeline

📚 References
Data Source
Titanic Dataset from Kaggle

Libraries Used
pandas: Data manipulation and analysis

scikit-learn: Machine learning algorithms

matplotlib/seaborn: Data visualization

streamlit: Web application framework

joblib: Model serialization

Related Research
Titanic Survival Analysis

Machine Learning Interpretation

🏆 Acknowledgments
Titanic dataset providers and Kaggle community

Scikit-learn development team for excellent ML tools

Streamlit team for easy web app deployment

Open-source contributors who made this project possible

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👥 Authors
Your Name - GitHub Profile

Contributors - List of contributors

🔗 Links
Live Demo: Coming Soon

Issue Tracker: GitHub Issues

Discussion Forum: GitHub Discussions

<div align="center">
⭐️ Don't forget to star this repository if you find it helpful!
"Women and children first" - The unwritten law of the sea that influenced survival patterns

</div>
🎯 Next Steps
Ready to explore? Here's what you can do next:

Run the notebook to understand the analysis process

Launch the web app to make interactive predictions

Experiment with different models by modifying the code

Contribute to make this project even better!

For questions or support, please open an issue or discussion in the repository.
