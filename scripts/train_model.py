# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

from scripts.preprocess import load_raw_data, clean_data, encode_features, scale_features

def prepare_data():
    df = load_raw_data(r"F:\churn_prediction_project\churn_prediction_project\data\telco_churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n✅ Model trained successfully!")
    print("\n🎯 Accuracy:", accuracy_score(y_test, predictions))
    print("\n📊 Classification Report:\n", classification_report(y_test, predictions))

    # Save model
    joblib.dump(model, "models/logistic_model.pkl")
    print("\n💾 Model saved to models/logistic_model.pkl")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    train_and_evaluate(X_train, X_test, y_train, y_test)
