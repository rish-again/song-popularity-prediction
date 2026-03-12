"""
eda.py

Exploratory Data Analysis for Song Popularity Dataset.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def create_reports_folder():
    """Create reports directory if it doesn't exist."""
    os.makedirs("reports", exist_ok=True)


def popularity_distribution(df):
    """Plot popularity distribution."""

    plt.figure()

    sns.histplot(df["popularity"], bins=30, kde=True)

    plt.title("Song Popularity Distribution")
    plt.xlabel("Popularity Score")
    plt.ylabel("Frequency")

    plt.savefig("reports/popularity_distribution.png")

    plt.close()

    print("Saved popularity distribution plot")


def feature_correlation(df):
    """Create correlation heatmap."""

    plt.figure(figsize=(12, 8))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(correlation, cmap="coolwarm")

    plt.title("Feature Correlation Heatmap")

    plt.savefig("reports/correlation_heatmap.png")

    plt.close()

    print("Saved correlation heatmap")


def feature_vs_popularity(df):
    """Plot relationships between key features and popularity."""

    important_features = [
        "danceability",
        "energy",
        "loudness",
        "tempo",
        "valence"
    ]

    for feature in important_features:

        if feature in df.columns:

            plt.figure()

            sns.scatterplot(x=df[feature], y=df["popularity"])

            plt.title(f"{feature} vs Popularity")

            plt.xlabel(feature)
            plt.ylabel("Popularity")

            filename = f"reports/{feature}_vs_popularity.png"

            plt.savefig(filename)

            plt.close()

            print(f"Saved {feature} vs popularity plot")


def run_eda(df):
    """Run all EDA functions."""

    create_reports_folder()

    popularity_distribution(df)

    feature_correlation(df)

    feature_vs_popularity(df)

if __name__ == "__main__":

    from src.data_load import load_data

    df = load_data()

    run_eda(df)