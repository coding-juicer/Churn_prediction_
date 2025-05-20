import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
le = LabelEncoder()
scaler = StandardScaler()

model = pickle.load(open('churn_logistic_model.pkl', 'rb'))
st.title("Logistic Regression for churn Prediction")



gender = st.selectbox('Select Gender:',options=['Female','Male'])
SeniorCitizen = st.selectbox('Select Citizen:',options=['Yes','No'])
Partner = st.selectbox('Do you have partner:',options=['Yes','No'])
Dependent = st.selectbox('Are you dependents on other:',options=['Yes','No'])
tenure = st.text_input('Enter tenure:')
PhoneService=st.selectbox('Do you have PhoneService:',options=['Yes','No'])
MultipleLines=st.selectbox('Do you have MultiLineService:',options=['Yes','No','No phone service'])
Contract=st.selectbox('Your contracts:',options=['One year','Two year','Month-to-month'])
TotalCharges=st.text_input('Enter your total charges:')

try:
    tenure = float(tenure)
    TotalCharges = float(TotalCharges)
    AvgMonthlyCharges = TotalCharges / (tenure + 1)
except:
    st.error("Please enter numeric values for Tenure and Total Charges.")
    st.stop()

st.write('The Average Monthly Charges is:', round(AvgMonthlyCharges, 2))

# Derive new feature
is_long_term = 0 if Contract == 'Month-to-month' else 1


def predictive(gender, SeniorCitizen, Partner, Dependent, tenure, PhoneService, MultipleLines, Contract, TotalCharges,
               AvgMonthlyCharges, is_long_term):
    data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependent': [Dependent],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'Contract': [Contract],
        'TotalCharges': [TotalCharges],
        'AvgMonthlyCharges': [AvgMonthlyCharges],
        'is_long_term': [is_long_term]

    }
    df1 = pd.DataFrame(data)

    categroical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependent', 'tenure', 'PhoneService',
                           'MultipleLines', 'Contract', 'TotalCharges', 'AvgMonthlyCharges', 'is_long_term']
    for columns in categroical_columns:
        df1[columns] = le.fit_transform(df1[columns])
    df1 = scaler.fit_transform(df1)

    result = model.predict(df1).reshape(1, -1)
    return result[0]

if st.button('Predict'):
    result =predictive(gender,SeniorCitizen,Partner, Dependent,tenure,PhoneService,MultipleLines,Contract,TotalCharges,AvgMonthlyCharges,is_long_term)
    if result == 0:
        st.write("Not churn")
    else:
        st.write("Churn")