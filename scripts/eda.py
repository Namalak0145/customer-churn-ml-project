# scripts/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data(path):
    try:
        df = pd.read_csv(r"F:\churn_prediction_project\churn_prediction_project\data\telco_churn.csv")
        print("✅ Data loaded successfully!")
        return df
    except FileNotFoundError:
        print("❌ File not found. Please check the path.")
        return None

# Basic EDA
def explore_data(df):
    print("\n📌 Data Shape:", df.shape)
    print("\n📌 Columns:\n", df.columns.tolist())
    print("\n📌 Missing Values:\n", df.isnull().sum())
    print("\n📌 Data Types:\n", df.dtypes)
    print("\n📌 First 5 Rows:\n", df.head())

if __name__ == "__main__":
    df = load_data(r"F:\churn_prediction_project\churn_prediction_project\data\telco_churn.csv")
    if df is not None:
        explore_data(df)


def plot_churn_distribution(df):
    sns.countplot(x='Churn', data=df)
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_numerical_distributions(df):
    num_cols = ['tenure', 'MonthlyCharges']
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

def plot_categorical_vs_churn(df):
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'SeniorCitizen']
    for col in cat_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, hue='Churn', data=df)
        plt.title(f"{col} vs Churn")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    df = load_data("data/telco_churn.csv")
    if df is not None:
        explore_data(df)
        plot_churn_distribution(df)
        plot_numerical_distributions(df)
        plot_categorical_vs_churn(df)
