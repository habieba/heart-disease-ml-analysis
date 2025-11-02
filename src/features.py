from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

def preprocessor(type):
    """
        creates preprocessing pipeline for numeric and categorical features.

    :param type:
            model_type (str, optional): The type of numeric pipeline to create.
            Valid options are 'polynomial', 'sqrt', or 'linear'.
            Defaults to 'linear'.
    :return:
            ColumnTransformer: The configured preprocessing object

    """
    if type == "polynomial":
        numeric_pipeline = Pipeline(steps=[
            ('imputer' , SimpleImputer(strategy='mean')),
            ('features', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler' , StandardScaler())]
)
    elif type == 'sqrt':
        numeric_pipeline = Pipeline(steps=[
            ('imputer' , SimpleImputer(strategy='mean')),
            ('features', FunctionTransformer(lambda X: np.sign(X) * np.abs(X)**0.2, validate=False)),
            ('scaler' , StandardScaler())
        ])

    else:
        numeric_pipeline = Pipeline(steps=[
            ('imputer' , SimpleImputer(strategy='mean')),
            ('scaler' , StandardScaler())
            ]
        )

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))]
    )
    transformer = ColumnTransformer(transformers=
            [
            ('numeric_features', numeric_pipeline, ['age', 'trestbps', 'thalach', 'oldpeak']),
            ('categorical_features',categorical_pipeline, ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
            ]
    )
    return transformer
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
