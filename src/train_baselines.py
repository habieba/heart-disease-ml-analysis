from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.evaluate import evaluate_classification_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score, roc_auc_score

import joblib

from src.features import build_preprocessor, load_data

def train_gaussian_nb(data_path="data/cleaned_cleveland.csv", save_model=True):
    df = load_data(data_path)
    X = df.drop(columns=['presence', 'thalach'])
    y = df['presence']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(task="classification")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GaussianNB())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

   # Evaluate metrics
    acc, f1, auc, y_pred, y_proba = evaluate_classification_model(
        pipeline, X_test, y_test, model_name="GaussianNB"
    )
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    if save_model:
        joblib.dump(pipeline, "models/gaussian_nb_pipeline.pkl")
        print("GaussianNB pipeline saved!")

    return pipeline, X_test, y_test, y_pred, acc, f1, auc


# --------------------------- #
# 2. Bernoulli Naive Bayes
# --------------------------- #
def train_bernoulli_nb(data_path="data/cleaned_cleveland.csv", save_model=True):
    df = load_data(data_path)
    X = df.drop(columns=['presence', 'thalach'])
    y = df['presence']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(task="classification")

    # After preprocessing, binarize features for BernoulliNB
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('binarizer', Binarizer(threshold=0.0)),  # turn scaled features into 0/1
        ('model', BernoulliNB())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate metrics
    acc, f1, auc, y_pred, y_proba = evaluate_classification_model(
        pipeline, X_test, y_test, model_name="BernoulliNB"
    )
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # if save_model:
    #     os.makedirs("../models", exist_ok=True)
    #     joblib.dump(pipeline, "../models/bernoulli_nb_pipeline.pkl")
    return pipeline, X_test, y_test, y_pred, acc, f1, auc


# --------------------------- #
# 3. Multinomial Naive Bayes
# --------------------------- #
def train_multinomial_nb(data_path="data/cleaned_cleveland.csv", save_model=True):
    df = load_data(data_path)
    X = df.drop(columns=['presence', 'thalach'])
    y = df['presence']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MultinomialNB requires non-negative values
    preprocessor = build_preprocessor(task="classification")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', MinMaxScaler()),  # ensures all values are >= 0
        ('model', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate metrics
    acc, f1, auc, y_pred, y_proba = evaluate_classification_model(
        pipeline, X_test, y_test, model_name="MultinomialNB"
    )
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # if save_model:
    #     os.makedirs("../models", exist_ok=True)
    #     joblib.dump(pipeline, "../models/multinomial_nb_pipeline.pkl")
    return pipeline, X_test, y_test, y_pred, acc, f1, auc



def train_logistic_regression(data_path="data/cleaned_cleveland.csv", save_model=True):
    """
    Train a Logistic Regression model for heart disease classification.
    """
    # Load dataset
    df = load_data(data_path)

    # Separate features and target
    X = df.drop(columns=['presence', 'thalach'])
    y = df['presence']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build preprocessing pipeline (classification version)
    preprocessor = build_preprocessor(task="classification")

    # Define Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000, solver='liblinear')  # 'liblinear' is good for small datasets

    # Combine preprocessor + model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', log_reg)
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate metrics
    acc, f1, auc, y_pred, y_proba = evaluate_classification_model(
        pipeline, X_test, y_test, model_name="LogisticRegression"
    )
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    print(f"Logistic Regression Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    # if save_model:
    #     os.makedirs("../models", exist_ok=True)
    #     joblib.dump(pipeline, "../models/logistic_regression_pipeline.pkl")
    #     print("✅ Logistic Regression pipeline saved to /models/")

    return pipeline, X_test, y_test, y_pred, acc, f1, auc


if __name__ == "__main__":
    train_gaussian_nb()
    train_bernoulli_nb()
    train_multinomial_nb()
    train_logistic_regression()


