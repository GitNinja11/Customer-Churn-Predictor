import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from utils import clean_total_charges

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Load model and data
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Title
st.title("📊 Customer Churn Predictor")
st.write("This app predicts whether a customer will churn based on their profile and subscription information.")

# Inputs
st.header("Enter Customer Details")
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
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charge = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charge = st.number_input("Total Charges", min_value=0.0, value=2000.0)

# Create DataFrame
input_df = pd.DataFrame({
    "gender": [gender], "SeniorCitizen": [senior], "Partner": [partner], "Dependents": [dependents],
    "tenure": [tenure], "PhoneService": [phone], "MultipleLines": [multiple], "InternetService": [internet],
    "OnlineSecurity": [online_sec], "OnlineBackup": [online_backup], "DeviceProtection": [device_protection],
    "TechSupport": [tech_support], "StreamingTV": [streaming_tv], "StreamingMovies": [streaming_movies],
    "Contract": [contract], "PaperlessBilling": [paperless], "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charge], "TotalCharges": [total_charge]
})

# Predict button
if st.button("🔍 Predict Churn"):
    # Prediction
    y_prob = model.predict_proba(input_df)[0]
    pred = model.predict(input_df)[0]
    label = "Customer will Churn ❌" if pred == 1 else "Customer will Stay ✅"
    color = "red" if pred == 1 else "green"

    # Animated Progress Bar
    st.subheader("🔄 Churn Probability")
    my_bar = st.progress(0, text="Calculating...")
    for i in range(0, int(y_prob[1] * 100) + 1):
        my_bar.progress(i / 100.0, text=f"Churn Probability: {i}%")
        time.sleep(0.01)
    st.success(f"Final Churn Probability: {y_prob[1]:.2%}")

    # Classification result
    st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)

    # Static bar plot
    st.subheader("Live Probability Chart")
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(['Stay', 'Churn'], y_prob, color=['green', 'red'])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Probability")
    st.pyplot(fig_bar)

    
# Evaluation toggle checkboxes
st.markdown("---")
show_roc = st.checkbox("📈 Show ROC Curve")
show_confusion = st.checkbox("📊 Show Confusion Matrix")
show_report = st.checkbox("📄 Show Classification Report")

if show_roc:
    st.subheader("ROC Curve on Test Set")
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='orange')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

if show_confusion:
    st.subheader("Confusion Matrix on Test Set")
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

if show_report:
    st.subheader("Classification Report")
    y_pred_test = model.predict(X_test)
    report = classification_report(y_test, y_pred_test, target_names=["Stayed", "Churned"])
    st.text(report)

st.caption("✨ Use checkboxes to toggle evaluation results.")