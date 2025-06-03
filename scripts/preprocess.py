# scripts/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_raw_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows where TotalCharges is NaN
    df = df.dropna(subset=['TotalCharges'])

    # Drop customerID (not useful)
    df.drop('customerID', axis=1, inplace=True)

    return df

def encode_features(df):
    # Binary features (Yes/No)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Encode gender (Male=1, Female=0)
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode other categoricals
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

if __name__ == "__main__":
    df = load_raw_data(r"F:\churn_prediction_project\churn_prediction_project\data\telco_churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)

    print("✅ Preprocessing complete.")
    print("🎯 Final dataset shape:", df.shape)
    print(df.head())
