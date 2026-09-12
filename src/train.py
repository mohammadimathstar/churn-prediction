import os

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocess import build_pipeline, clean_data, load_data


def train_and_save_model():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    X, y = clean_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building pipeline...")
    pipeline = build_pipeline(X)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    pred_probs = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, pred_probs)

    print(f"Model Accuracy: {acc:.4f}")
    print(f"Model ROC-AUC: {roc:.4f}")

    # Save the pipeline
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
