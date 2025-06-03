# scripts/evaluate.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

from preprocess import load_raw_data, clean_data, encode_features, scale_features
from sklearn.model_selection import train_test_split

def get_test_data():
    df = load_raw_data("data/telco_churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:\n", classification_report(y_test, predictions))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("📥 Loading model...")
    model = joblib.load("models/logistic_model.pkl")
    X_train, X_test, y_train, y_test = get_test_data()
    evaluate_model(model, X_test, y_test)
