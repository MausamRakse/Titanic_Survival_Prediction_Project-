# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained models and preprocessing objects
lr_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Titanic Survival Prediction')
st.write('This app predicts whether a passenger would have survived the Titanic disaster using Logistic Regression and SVM models.')

# User input
st.sidebar.header('Passenger Information')

pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0, 100, 30)
sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
fare = st.sidebar.slider('Fare', 0, 100, 50)
embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Feature engineering
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Encode sex
sex_encoded = 1 if sex == 'female' else 0

# Encode title based on sex and age
if sex == 'male':
    if age < 18:
        title_encoded = 4  # Master
    else:
        title_encoded = 1  # Mr
else:
    if age < 18:
        title_encoded = 2  # Miss
    else:
        title_encoded = 3  # Mrs

# Encode age group
if age <= 12:
    age_group_encoded = 0  # Child
elif age <= 18:
    age_group_encoded = 1  # Teen
elif age <= 35:
    age_group_encoded = 2  # Adult
elif age <= 60:
    age_group_encoded = 3  # Middle
else:
    age_group_encoded = 4  # Senior

# One-hot encode embarked
embarked_c = 1 if embarked == 'C' else 0
embarked_q = 1 if embarked == 'Q' else 0
embarked_s = 1 if embarked == 'S' else 0

# Create feature array in correct order
features = np.array([[pclass, age, sibsp, parch, fare, sex_encoded, 
                      embarked_c, embarked_q, embarked_s, title_encoded, 
                      family_size, is_alone, age_group_encoded]])

# Scale numerical features
numerical_features = features[:, [1, 2, 3, 4, 10]]  # Age, SibSp, Parch, Fare, FamilySize
scaled_numerical = scaler.transform(numerical_features)

# Replace original numerical features with scaled ones
features[:, [1, 2, 3, 4, 10]] = scaled_numerical

# Model selection
model_choice = st.sidebar.selectbox('Select Model', ['Logistic Regression', 'SVM'])

if st.sidebar.button('Predict Survival'):
    if model_choice == 'Logistic Regression':
        model = lr_model
        model_name = 'Logistic Regression'
    else:
        model = svm_model
        model_name = 'SVM'
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    st.subheader(f'Prediction Result ({model_name})')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction[0] == 1:
            st.success('✅ The passenger would have survived!')
        else:
            st.error('❌ The passenger would not have survived.')
    
    with col2:
        st.metric("Probability of Survival", f"{probability[0][1]:.2%}")
        st.metric("Probability of Not Surviving", f"{probability[0][0]:.2%}")
    
    # Show confidence level
    confidence = max(probability[0])
    st.progress(confidence)
    st.write(f'Model Confidence: {confidence:.2%}')

# Display model comparison
st.sidebar.header('Model Comparison')
if st.sidebar.button('Compare Models'):
    st.subheader('Model Comparison on Test Data')
    
    # Create comparison data (you would normally load this from saved results)
    comparison_data = {
        'Model': ['Logistic Regression', 'SVM'],
        'Accuracy': [0.8212, 0.8324],  # Example values
        'Precision': [0.7895, 0.8036],
        'Recall': [0.7547, 0.7736],
        'F1-Score': [0.7717, 0.7883]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.highlight_max(axis=0))

# Feature explanation
st.sidebar.header('About Features')
st.sidebar.info("""
**Key Features:**
- **Sex**: Female passengers had higher survival rates
- **Pclass**: Higher classes (1st) had better survival chances
- **Age**: Children had priority in lifeboats
- **Fare**: Higher fare correlates with higher class
- **Family Size**: Traveling with family affected survival
""")

# Main area additional info
st.header('How to Use')
st.write("""
1. Adjust the passenger information in the sidebar
2. Select which model to use for prediction
3. Click 'Predict Survival' to see the result
4. Use 'Compare Models' to see performance metrics
""")

st.header('Model Information')
st.write("""
**Logistic Regression:**
- Linear model that estimates probability
- Fast training and prediction
- Good interpretability

**Support Vector Machine (SVM):**
- Finds optimal decision boundary
- Handles non-linear relationships well
- Can be more accurate but slower
""")