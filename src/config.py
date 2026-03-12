"""
config.py

This file contains configuration variables used across the
Song Popularity Prediction ML project.
"""

# -----------------------------
# Data Paths
# -----------------------------

DATA_PATH = "data/songs.csv"


# -----------------------------
# Model Paths
# -----------------------------

MODEL_PATH = "models/song_popularity_model.pkl"


# -----------------------------
# Target Variable
# -----------------------------

TARGET_COLUMN = "popularity"


# -----------------------------
# Train-Test Split
# -----------------------------

TEST_SIZE = 0.2


# -----------------------------
# Random State
# -----------------------------

RANDOM_STATE = 42


# -----------------------------
# Model Parameters
# -----------------------------

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_STATE
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": RANDOM_STATE
}