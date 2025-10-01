"""
Customer Churn Prediction
Business Goal: Predict which customers are likely to churn
Approach: Logistic Regression + Scaled Features
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

MODEL_PATH = "models/churn_model.pkl"

def train_model():
    """Train churn prediction model and save it."""
    df = pd.read_csv("data/churn_data.csv")

    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df[['Age', 'Tenure', 'Usage', 'SupportCalls']]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: Scaling + Logistic Regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")


def predict_churn(age, tenure, usage, support_calls):
    """Load trained model and predict churn for a new customer."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found! Train it first using --train")

    model = joblib.load(MODEL_PATH)
    new_customer = pd.DataFrame([[age, tenure, usage, support_calls]],
                                columns=['Age', 'Tenure', 'Usage', 'SupportCalls'])
    prediction = model.predict(new_customer)[0]
    print("\n🔮 Prediction:", "Yes (Will Churn)" if prediction else "No (Will Stay)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Prediction")

    parser.add_argument("--train", action="store_true",
                        help="Train the churn prediction model")
    parser.add_argument("--predict", nargs=4, type=int, metavar=("AGE", "TENURE", "USAGE", "CALLS"),
                        help="Predict churn for new customer (e.g., --predict 40 15 250 2)")

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.predict:
        age, tenure, usage, calls = args.predict
        predict_churn(age, tenure, usage, calls)
    else:
        print("No arguments provided. Example usage:")
        print("  python churn_predictor.py --train")
        print("  python churn_predictor.py --predict 40 15 250 2")
