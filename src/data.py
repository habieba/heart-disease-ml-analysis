# data.py
"""
Always fetch the UCI Heart Disease dataset programmatically (no manual files).
- Source: UCI Machine Learning Repository via `ucimlrepo` (id=45)
- Target standardized to binary `presence` (1: disease, 0: no disease)
- Returns cleaned pandas DataFrame via `load_dataset()`
- Provides `get_X_y(df)` for modeling
"""

from __future__ import annotations
import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
except Exception as e:
    raise ImportError(
        "ucimlrepo is required to fetch the dataset automatically. "
        "Install with: pip install ucimlrepo"
    ) from e


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the combined UCI heart dataset to Cleveland-like schema."""
    # unify target
    if "presence" not in df.columns:
        if "num" in df.columns:
            df = df.rename(columns={"num": "presence"})
        else:
            raise KeyError("Expected 'num' (or 'presence') column in UCI dataset.")

    # Remove rows with any missing values and coerce numerics
    df = df.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # Binary target: >0 -> 1
    df["presence"] = df["presence"].apply(lambda x: 1 if x > 0 else 0)

    # Drop duplicates, reset index
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_dataset() -> pd.DataFrame:
    """
    Fetch UCI id=45 via ucimlrepo and return a cleaned DataFrame.
    """
    heart = fetch_ucirepo(id=45)
    X = heart.data.features
    y = heart.data.targets
    df = pd.concat([X, y], axis=1)
    df = _clean_df(df)
    return df


def get_X_y(df: pd.DataFrame):
    """
    Return (X, y) for classification. Drop 'thalach' to mirror previous runs, if present.
    """
    target = "presence"
    drop_cols = [target, "thalach"] if "thalach" in df.columns else [target]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target]
    return X, y
