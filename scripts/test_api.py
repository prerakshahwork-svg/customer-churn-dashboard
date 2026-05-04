import requests

url = "http://127.0.0.1:5000/predict"

data = {
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

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
