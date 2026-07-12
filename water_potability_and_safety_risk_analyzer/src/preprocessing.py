"""
preprocessing.py

Data loading and cleaning utilities for the Water Potability project.

Responsibilities:
- Load the raw CSV
- Report / handle missing values
- Produce a clean, model-ready DataFrame
- Split into train/test sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from reference_ranges import FEATURE_COLUMNS, TARGET_COLUMN


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw water potability CSV."""
    df = pd.read_csv(csv_path)
    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small table summarising missing values per column."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def clean_data(df: pd.DataFrame, strategy: str = "median_by_class") -> pd.DataFrame:
    """
    Clean the raw dataframe.

    strategy:
        "median_by_class" (default) -> fill NaNs using the median of that
            column computed *within each Potability class*. This keeps
            the imputation realistic since safe vs unsafe water tends to
            have different typical parameter values.
        "median" -> fill NaNs with the overall column median.
        "drop"   -> drop rows containing any NaN.
    """
    df = df.copy()

    # Drop fully empty rows/duplicates first
    df = df.drop_duplicates()

    if strategy == "drop":
        df = df.dropna()

    elif strategy == "median":
        for col in FEATURE_COLUMNS:
            df[col] = df[col].fillna(df[col].median())

    elif strategy == "median_by_class":
        for col in FEATURE_COLUMNS:
            df[col] = df.groupby(TARGET_COLUMN)[col].transform(
                lambda s: s.fillna(s.median())
            )
            # safety net in case a class-median is itself NaN
            df[col] = df[col].fillna(df[col].median())

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df = df.reset_index(drop=True)
    return df


def get_feature_target_split(df: pd.DataFrame):
    """Return X (features) and y (target) from a cleaned dataframe."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split cleaned data into train/test sets (stratified on target)."""
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw = load_raw_data("../data/water_potability.csv")
    print("Raw shape:", raw.shape)
    print("\nMissing value report:\n", missing_value_report(raw))

    clean = clean_data(raw, strategy="median_by_class")
    print("\nClean shape:", clean.shape)
    print("Remaining NaNs:", clean.isnull().sum().sum())
