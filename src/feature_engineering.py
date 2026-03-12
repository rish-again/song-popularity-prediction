"""
feature_engineering.py

Prepare dataset for machine learning models.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def split_features_target(df):

    # Target variable
    y = df["popularity"]

    # Drop target
    X = df.drop(columns=["popularity"])

    # Keep only numeric columns
    X = X.select_dtypes(include=["int64", "float64"])

    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    return X, y


def train_test_split_data(X, y):
    """
    Split dataset into training and testing sets
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed")

    return X_train_scaled, X_test_scaled, scaler