import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained objects
model = pickle.load(open('churn_logistic_model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler-2.pkl', 'rb'))

st.title("Logistic Regression for Churn Prediction")

gender = st.selectbox('Select Gender:', options=['Female', 'Male'])
SeniorCitizen = st.selectbox('Are you a Senior Citizen?', options=['Yes', 'No'])
Partner = st.selectbox('Do you have a partner?', options=['Yes', 'No'])
Dependent = st.selectbox('Are you dependent on others?', options=['Yes', 'No'])
tenure = st.text_input('Enter tenure (in months):')
PhoneService = st.selectbox('Do you have Phone Service?', options=['Yes', 'No'])
MultipleLines = st.selectbox('Do you have Multiple Lines?', options=['Yes', 'No', 'No phone service'])
Contract = st.selectbox('Your contract type:', options=['One year', 'Two year', 'Month-to-month'])
TotalCharges = st.text_input('Enter your total charges:')

try:
    tenure = float(tenure)
    TotalCharges = float(TotalCharges)
    AvgMonthlyCharges = TotalCharges / (tenure + 1)  # to avoid division by zero
except:
    st.error("Please enter valid numeric values for tenure and total charges.")
    st.stop()

st.write('Average Monthly Charges:', round(AvgMonthlyCharges, 2))

# Derive is_long_term feature
is_long_term = 0 if Contract == 'Month-to-month' else 1


def predictive(gender, SeniorCitizen, Partner, Dependent, tenure, PhoneService, MultipleLines, Contract,
               TotalCharges, AvgMonthlyCharges, is_long_term):
    # Map SeniorCitizen from Yes/No to 1/0
    senior_citizen_num = 1 if SeniorCitizen == 'Yes' else 0

    data = {
        'gender': [gender],
        'SeniorCitizen': [senior_citizen_num],
        'Partner': [Partner],
        'Dependents': [Dependent],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'Contract': [Contract],
        'TotalCharges': [TotalCharges],
        'AvgMonthlyCharges': [AvgMonthlyCharges],
        'is_long_term': [is_long_term]
    }

    df = pd.DataFrame(data)

    # Encode categorical variables with your pre-trained label encoders
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col])

    # Scale numerical columns
    numeric_columns = ['tenure', 'TotalCharges', 'AvgMonthlyCharges', 'is_long_term']
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    prediction = model.predict(df)
    return prediction[0]


if st.button('Predict'):
    result = predictive(gender, SeniorCitizen, Partner, Dependent, tenure, PhoneService, MultipleLines, Contract,
                        TotalCharges, AvgMonthlyCharges, is_long_term)
    if result == 0:
        st.write("No churn predicted")
    else:
        st.write("Churn predicted")
