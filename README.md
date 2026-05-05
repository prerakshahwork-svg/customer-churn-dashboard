# Customer Churn Prediction Dashboard

## 📌 Project Overview
This project predicts whether a customer will churn (exit) or stay (retain) based on their profile and account details.  
It combines:
- **Machine Learning (Logistic Regression)** for prediction.
- **Flask API** for serving the model.
- **Streamlit UI** for user interaction and visualization.

---

## 📂 Repository Structure
customer-churn-dashboard/
├── api/                # Flask API backend
│   └── api.py
├── models/             # Saved ML models
│   └── churn_model.pkl
├── scripts/            # Helper/testing scripts
│   └── test_api.py
├── ui/                 # Streamlit frontend
│   └── app.py
├── customer_churn.csv  # Dataset
└── README.md           # Documentation

Code

---

## 📊 Dataset
- File: `customer_churn.csv`
- Features:
  - `CustomerID` → Unique identifier
  - `Gender` → Male/Female
  - `Age` → Customer age
  - `Tenure` → Years with the bank
  - `Balance` → Account balance
  - `Products` → Number of products used
  - `HasCreditCard` → 1 if yes, 0 if no
  - `IsActiveMember` → 1 if active, 0 if not
  - `EstimatedSalary` → Salary estimate
- Target:
  - `Exited` → 1 if churned, 0 if retained

---

## 🤖 Model Training
- Algorithm: **Logistic Regression** (scikit-learn).
- Features used:  
  `CustomerID, Gender, Age, Tenure, Balance, Products, HasCreditCard, IsActiveMember, EstimatedSalary`
- Target: `Exited`
- Saved model: `models/churn_model.pkl`

---

## ⚙️ Flask API
- File: `api/api.py`
- Endpoint:  
POST /predict

Code
- Input JSON format:
```json
{
  "CustomerID": 15647311,
  "Gender": 1,
  "Age": 35,
  "Tenure": 5,
  "Balance": 60000,
  "Products": 2,
  "HasCreditCard": 0,
  "IsActiveMember": 1,
  "EstimatedSalary": 60000
}
Output JSON:

json
{"Exited": 0}
🖥️ Streamlit UI
File: ui/app.py

Run with:

bash
streamlit run ui/app.py
Features:

Form inputs for customer details.

Sends JSON to Flask API.

Displays prediction (Exited or Retained).

🧪 Testing
Run API:

bash
python api/api.py
Run test script:

bash
python scripts/test_api.py
Run UI:

bash
streamlit run ui/app.py
👥 Team Contributions
Vanitha → Backend (Flask API, debugging, feature alignment).

Prerak → Frontend (Streamlit UI, integration, testing).

Together → Dataset preparation, model training, end-to-end testing.
