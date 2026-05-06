import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Profiling temporarily disabled
# from ydata_profiling import ProfileReport

# -----------------------------
# Load dataset and model
# -----------------------------
df = pd.read_csv("data/customer_churn.csv")
model = joblib.load("models/churn_model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="VanPre Customer Churn Dashboard", layout="wide")

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #1E90FF;
    }
    .stMetric {
        background-color: #eaf2f8;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Logo / Banner
# -----------------------------
st.image("assets/vanpre_logo.png", width=180)
st.title("📊 VanPre Customer Churn Prediction Dashboard")

# -----------------------------
# Metrics Cards
# -----------------------------
churn_rate = df["Exited"].mean() * 100
avg_balance = df["Balance"].mean()
avg_age = df["Age"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Churn Rate", f"{churn_rate:.2f}%")
col2.metric("Average Balance", f"{avg_balance:,.0f}")
col3.metric("Average Age", f"{avg_age:.1f}")

st.markdown("---")

# -----------------------------
# Sidebar Prediction Form
# -----------------------------
st.sidebar.image("assets/vanpre_logo.png", width=120)
st.sidebar.header("🔮 Predict Customer Churn")

customer_id = st.sidebar.number_input("Customer ID", value=12345)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", value=35)
tenure = st.sidebar.number_input("Tenure", value=5)
balance = st.sidebar.number_input("Balance", value=60000)
products = st.sidebar.number_input("Products", value=2)
has_credit_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", value=60000)

if st.sidebar.button("Predict"):
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
    st.success(f"Prediction: {'Exited' if prediction == 1 else 'Retained'}")

st.markdown("---")

# -----------------------------
# Automated EDA Section (commented out)
# -----------------------------
st.header("📑 Automated EDA Report")
# profile = ProfileReport(df, title="Customer Churn EDA", explorative=True)
# st.components.v1.html(profile.to_html(), height=800, scrolling=True)
st.info("ℹ️ Automated profiling (ydata-profiling) is disabled due to environment issues.")

st.markdown("---")

# -----------------------------
# Custom EDA Visuals
# -----------------------------
st.header("📈 Exploratory Data Analysis")

sns.set_theme(style="whitegrid", palette="crest")

with st.expander("Gender vs Churn"):
    fig, ax = plt.subplots()
    sns.countplot(x="Gender", hue="Exited", data=df, ax=ax)
    st.pyplot(fig)

with st.expander("Age Distribution"):
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

with st.expander("Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with st.expander("Balance vs Churn"):
    fig, ax = plt.subplots()
    sns.boxplot(x="Exited", y="Balance", data=df, ax=ax)
    ax.set_xticklabels(["Retained", "Exited"])
    st.pyplot(fig)

with st.expander("Churn Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(x="Exited", data=df, ax=ax)
    ax.set_xticklabels(["Retained", "Exited"])
    st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("© 2026 VanPre | Dashboard built with Streamlit. Profiling disabled for stability.")
