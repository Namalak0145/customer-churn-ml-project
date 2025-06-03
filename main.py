# main.py

from scripts.preprocess import load_raw_data, clean_data, encode_features, scale_features
from scripts.train_model import train_and_evaluate
from sklearn.model_selection import train_test_split
import joblib

def run_pipeline():
    print("🔄 Loading and preprocessing data...")
    df = load_raw_data(r"F:\churn_prediction_project\churn_prediction_project\data\telco_churn.csv")
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 Training model and evaluating...")
    model = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("💾 Saving model...")
    joblib.dump(model, r"F:\churn_prediction_project\churn_prediction_project\models\logistic_model.pkl")
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()
