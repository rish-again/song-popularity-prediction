"""
preprocessing.py

This module cleans and prepares the dataset
for machine learning.
"""

import pandas as pd
from src.config import TARGET_COLUMN


def remove_duplicates(df):
    """Remove duplicate rows."""

    initial_rows = len(df)

    df = df.drop_duplicates()

    final_rows = len(df)

    print(f"Duplicates removed: {initial_rows - final_rows}")

    return df


def handle_missing_values(df):
    """Handle missing values."""

    missing_values = df.isnull().sum().sum()

    print(f"Total missing values: {missing_values}")

    # Simple strategy: drop rows with missing values
    df = df.dropna()

    return df


def drop_irrelevant_columns(df):
    """
    Drop columns that are not useful for prediction.
    """

    columns_to_drop = ["track_id", "track_name", "artists"]

    existing_cols = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_cols)

    print(f"Dropped columns: {existing_cols}")

    return df

"""
preprocessing.py

Clean the dataset before training.
"""

def clean_data(df):

    print("Initial dataset shape:", df.shape)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Dropped column: Unnamed: 0")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    print("Dataset shape after cleaning:", df.shape)

    return df


def split_features_target(df):
    """
    Split dataset into features (X) and target (y)
    """

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y