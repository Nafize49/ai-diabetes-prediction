"""
preprocessing.py
----------------
Data loading, inspection, biological zero treatment, and preprocessing pipeline.

Student : Nafize Ali
Branch  : Artificial Intelligence & Data Science
Project : AI-Based Diabetes Prediction System (B.Tech Minor Project)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Core features directly present in the dataset and aligned with project specification
CORE_FEATURES = ["Age", "BMI", "BloodPressure", "Glucose"]


def load_dataset(csv_path: str = None) -> pd.DataFrame:
    """Loads the real diabetes dataset from disk."""
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(base_dir, "dataset", "diabetes_binary_health.csv"),
            os.path.join(base_dir, "diabetes_binary_health.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                csv_path = p
                break
        if csv_path is None:
            csv_path = candidates[0]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    return pd.read_csv(csv_path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Returns verified dataset statistics without fabricating any data."""
    biological_zeros = {}
    for col in ["Glucose", "BloodPressure", "BMI"]:
        if col in df.columns:
            biological_zeros[col] = int((df[col] == 0).sum())

    target_col = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    class_dist = df[target_col].value_counts().to_dict()

    return {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_explicit": df.isnull().sum().to_dict(),
        "biological_zeros": biological_zeros,
        "duplicates": int(df.duplicated().sum()),
        "class_distribution": class_dist,
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Biological Zero Treatment:
    Glucose, BloodPressure, and BMI cannot physiologically be zero in living persons.
    Zeros in these columns represent unrecorded values and are replaced with NaN.
    """
    df_clean = df.copy()

    # Deduplicate if exact duplicate rows exist
    df_clean = df_clean.drop_duplicates()

    # Biological zero replacement
    physiological_columns = ["Glucose", "BloodPressure", "BMI"]
    for col in physiological_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)

    return df_clean


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separates the feature matrix X and target vector y.
    Only uses the 4 core clinically verified features.
    """
    target_col = "Outcome" if "Outcome" in df.columns else df.columns[-1]

    # Verify core features exist in dataset
    available_features = [f for f in CORE_FEATURES if f in df.columns]
    X = df[available_features].copy()
    y = df[target_col].astype(int).copy()

    return X, y


def build_preprocessor() -> Pipeline:
    """
    Encapsulates preprocessing in a Scikit-Learn Pipeline:
    1. SimpleImputer with median strategy (handles biological zero NaNs)
    2. StandardScaler (normalizes features to mean=0, std=1)

    Fitting strictly on X_train guarantees zero data leakage.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified split to preserve the 62.3% / 37.7% class balance in both partitions.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
