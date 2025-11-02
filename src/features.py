# src/features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_preprocessor(task="classification"):
    """
    Create a reusable preprocessor pipeline for numeric and categorical features.
    Returns a ColumnTransformer object.
    """
    if task == "classification":
        numeric_features = ['age', 'trestbps', 'chol', 'oldpeak']
    else:  # regression includes thalach as input target
        numeric_features = ['age', 'trestbps', 'chol', 'oldpeak']

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor


def load_data(data_path="data/cleaned_heart.csv"):
    """Load cleaned heart dataset."""
    df = pd.read_csv(data_path)
    return df
