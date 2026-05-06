import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport

# -----------------------------
# Load dataset and model
# -----------------------------
df = pd.read_csv("data/customer_churn.csv")
model = joblib.load("models/churn_model.pkl")

# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title("Customer Churn Prediction Dashboard")

# -----------------------------
# Automated EDA Section
# -----------------------------
st.header("Automated EDA Report")
profile = ProfileReport(df, title="Customer Churn EDA", explorative=True)
st.components.v1.html(profile.to_html(), height=800, scrolling=True)

# -----------------------------
# Custom EDA Visuals
# -----------------------------
st.header("Custom EDA Visuals")

# Churn by Gender
fig, ax = plt.subplots()
sns.countplot(x="Gender", hue="Exited", data=df, ax=ax)
st.pyplot(fig)

# Age distribution
fig, ax = plt.subplots()
sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Correlation heatmap
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------
# Prediction Form
# -----------------------------
st.header("Predict Customer Churn")

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
    
    prediction = model.predict(input_df)[0]
    st.write("Prediction:", "Exited" if prediction == 1 else "Retained")
