"""
run_pipeline.py

Main pipeline for song popularity prediction project.
"""

import joblib
import os

from src.data_load import load_data
from src.preprocessing import clean_data

from src.feature_engineering import (
    split_features_target,
    train_test_split_data,
    scale_features
)

from src.train_model import (
    train_linear_regression,
    train_random_forest,
    train_gradient_boosting
)

from src.evaluation_model import evaluate_model


def main():

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    print("\nLoading dataset...")
    df = load_data()

    print("\nCleaning data...")
    df = clean_data(df)

    print("\nPreparing features...")
    X, y = split_features_target(df)

    # 🔹 Save feature column names (VERY IMPORTANT FOR STREAMLIT)
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    print("Feature columns saved to models/feature_columns.pkl")

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 🔹 Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")

    print("\nTraining models...")

    models = {
        "Linear Regression": train_linear_regression(X_train_scaled, y_train),
        "Random Forest": train_random_forest(X_train_scaled, y_train),
        "Gradient Boosting": train_gradient_boosting(X_train_scaled, y_train)
    }

    print("\nEvaluating models...")

    results = []

    for name, model in models.items():

        result = evaluate_model(model, X_test_scaled, y_test, name)

        results.append((name, model, result["R2"]))

    # Select best model
    best_model_name, best_model, best_score = max(results, key=lambda x: x[2])

    print("\n==============================")
    print("BEST MODEL SELECTED")
    print("==============================")

    print(f"Model: {best_model_name}")
    print(f"R2 Score: {best_score:.3f}")

    # Save best model
    joblib.dump(best_model, "models/best_model.pkl")

    print("\nBest model saved to models/best_model.pkl")


if __name__ == "__main__":
    main()