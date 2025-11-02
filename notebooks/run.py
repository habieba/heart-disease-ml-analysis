"""
Run all baseline models (Naive Bayes + Logistic Regression),
evaluate them, log to MLflow, and generate comparison plots.
"""

import sys, os
import pandas as pd

# --- Ensure src package is importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# --- Imports ---
from src.train_baselines import (
    train_gaussian_nb,
    train_bernoulli_nb,
    train_multinomial_nb,
    train_logistic_regression
)
from src.evaluate import evaluate_classification_model


# -------------------------------
# 🚀 Run and Compare Models
# -------------------------------

def run_all_models():
    results = []

    print("🚀 Starting baseline model training and comparison...\n")

    # 1️⃣ Gaussian Naive Bayes
    print("Training Gaussian Naive Bayes...")
    _, X_test, y_test, y_pred, acc, f1, auc = train_gaussian_nb()
    results.append({
        "model": "GaussianNB",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc
    })

    # 2️⃣ Bernoulli Naive Bayes
    print("\nTraining Bernoulli Naive Bayes...")
    _, X_test, y_test, y_pred, acc, f1, auc = train_bernoulli_nb()
    results.append({
        "model": "BernoulliNB",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc
    })

    # 3️⃣ Multinomial Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")
    _, X_test, y_test, y_pred, acc, f1, auc = train_multinomial_nb()
    results.append({
        "model": "MultinomialNB",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc
    })

    # 4️⃣ Logistic Regression
    print("\nTraining Logistic Regression...")
    _, X_test, y_test, y_pred, acc, f1, auc = train_logistic_regression()
    results.append({
        "model": "Logistic Regression",
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc
    })

    # -------------------------------
    # 📊 Create Comparison Table
    # -------------------------------
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    print("\n✅ Final Model Comparison:")
    print(df_results)


    return df_results


if __name__ == "__main__":
    df_results = run_all_models()
