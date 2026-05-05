# scripts/eda.py
# Exploratory Data Analysis for Customer Churn Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure plots folder exists
os.makedirs("data/eda_plots", exist_ok=True)

# 1. Load dataset (correct path)
df = pd.read_csv("data/customer_churn.csv")

# 2. Basic inspection
print("Shape:", df.shape)
print(df.info())
print(df.head())

# 3. Missing values & duplicates
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# 4. Descriptive statistics
print(df.describe())
print("Gender counts:\n", df['Gender'].value_counts())
print("Exited counts:\n", df['Exited'].value_counts())

# 5. Univariate Analysis
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("data/eda_plots/age_distribution.png", dpi=300)
plt.close()

sns.countplot(x='Exited', data=df)
plt.title("Churn Distribution")
plt.savefig("data/eda_plots/churn_distribution.png", dpi=300)
plt.close()

# 6. Bivariate Analysis
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title("Balance vs Churn")
plt.savefig("data/eda_plots/balance_vs_churn.png", dpi=300)
plt.close()

sns.countplot(x='Products', hue='Exited', data=df)
plt.title("Products vs Churn")
plt.savefig("data/eda_plots/products_vs_churn.png", dpi=300)
plt.close()

# 7. Correlation Analysis (numeric only)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("data/eda_plots/correlation_heatmap.png", dpi=300)
plt.close()


print("✅ EDA completed. Plots saved in data/eda_plots/")
