"""
train_model.py

Train machine learning models for song popularity prediction.
"""
import os 
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.config import MODEL_PATH

def save_feature_columns(X_train):

    os.makedirs("models", exist_ok=True)

    feature_path = "models/feature_columns.pkl"

    joblib.dump(X_train.columns.tolist(), feature_path)

    print(f"Feature columns saved to {feature_path}")



def train_linear_regression(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("Linear Regression trained")

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Random Forest trained")

    return model


def train_gradient_boosting(X_train, y_train):

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Gradient Boosting trained")

    return model


def save_model(model):

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")