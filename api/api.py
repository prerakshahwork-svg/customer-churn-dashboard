# api/api.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("models/churn_model.pkl")

# Define the expected features in the exact order used during training
expected_features = [
    "CustomerID", "Gender", "Age", "Tenure", "Balance",
    "Products", "HasCreditCard", "IsActiveMember", "EstimatedSalary"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Build DataFrame with correct order of features
        df = pd.DataFrame([[data.get(f) for f in expected_features]], columns=expected_features)
        print("Incoming Data:", df)  # Debug line to check input
        prediction = model.predict(df)[0]
        return jsonify({"Exited": int(prediction)})
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
