
# features.py
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

# New: use data1.py as the single source of truth for loading data
try:
    import data as data_src
except ImportError:
    # Fallback to local import if the module name/environment differs
    import data as data_src


def preprocessor(kind: str = "linear"):
    """
    Create preprocessing pipeline for numeric and categorical features.
    Valid kinds: 'polynomial', 'sqrt', 'linear' (default).
    """
    if kind == "polynomial":
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('features', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler())
        ])
    elif kind == "sqrt":
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('features', FunctionTransformer(lambda X: np.sign(X) * np.abs(X)**0.2, validate=False)),
            ('scaler', StandardScaler())
        ])
    else:
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Keep a robust core feature set found in UCI Cleveland-like schema
    numeric_features = ['age', 'trestbps', 'chol', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    transformer = ColumnTransformer(transformers=[
        ('numeric_features', numeric_pipeline, numeric_features),
        ('categorical_features', categorical_pipeline, categorical_features)
    ])
    return transformer


def build_preprocessor(task: str = "classification"):
    """
    Thin wrapper for compatibility with existing code.
    """
    return preprocessor("linear")


def load_data() -> pd.DataFrame:
    """
    Load the cleaned UCI Heart Disease dataset through data1.py.
    """
    return data_src.load_dataset()
