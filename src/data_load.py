import pandas as pd
import os
from src.config import DATA_PATH


def load_data(n_rows=5000):
    """
    Load dataset with optional row limit.

    Parameters
    ----------
    n_rows : int
        Number of rows to load from dataset

    Returns
    -------
    df : pandas.DataFrame
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, nrows=n_rows)

    print("Dataset loaded successfully")
    print(f"Rows Loaded: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df

def dataset_overview(df):
    """
    Display basic dataset information.
    """

    print("\nDataset Info")
    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nBasic Statistics")
    print(df.describe())

if __name__ == "__main__":
    df = load_data()
    dataset_overview(df)