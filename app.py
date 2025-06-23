import streamlit as st
import pandas as pd
import numpy as np
from utils import clean_total_charges
import pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Streamlit UI
st.title("📊 Customer Churn Predictor")
st.write("This app predicts whether a customer will churn based on their demographics and account information.")

st.header("Enter Customer Details")

# Input widgets
gender = st.selectbox("Gender", ['Male', 'Female'])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ['Yes', 'No'])
dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone = st.selectbox("Phone Service", ['Yes', 'No'])
multiple = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])

internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_sec = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])

contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

monthly_charge = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charge = st.number_input("Total Charges", min_value=0.0, value=2000.0)

# Create input DataFrame
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone],
    "MultipleLines": [multiple],
    "InternetService": [internet],
    "OnlineSecurity": [online_sec],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charge],
    "TotalCharges": [total_charge]
})

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Customer will Churn ❌" if prediction == 1 else "Customer will Stay ✅"
    st.subheader("Prediction:")
    st.success(result)
