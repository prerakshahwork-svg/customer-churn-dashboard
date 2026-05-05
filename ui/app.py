import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")   # adjust filename if needed

st.title("Customer Churn Prediction Dashboard")

# Input fields
customer_id = st.number_input("Customer ID", value=12345)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", value=35)
tenure = st.number_input("Tenure", value=5)
balance = st.number_input("Balance", value=60000)
products = st.number_input("Products", value=2)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", value=60000)

if st.button("Predict"):
    gender_num = 1 if gender == "Male" else 0
    
    # ✅ Create DataFrame properly
    input_df = pd.DataFrame([{
        "CustomerID": customer_id,
        "Gender": gender_num,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "Products": products,
        "HasCreditCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }])
    
    # ✅ Direct prediction (no response.json)
    prediction = model.predict(input_df)[0]
    st.write("Prediction:", "Exited" if prediction == 1 else "Retained")
