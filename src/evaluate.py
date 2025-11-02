# src/evaluate.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)


def evaluate_classification_model(model, X_test, y_test, model_name="Model", log_to_mlflow=True):
    """
    Universal evaluation function for classification models.
    Computes metrics, plots confusion matrix and ROC curve,
    and logs everything to MLflow (optional).

    Returns: acc, f1, auc, y_pred, y_proba
    """

    # --- Predictions ---
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n📊 {model_name} Evaluation:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    if auc is not None:
        print(f"  ROC-AUC  : {auc:.4f}")
    else:
        print(f"  ROC-AUC  : N/A (no predict_proba)")

    # --- Create directories for plots ---
    os.makedirs("models/plots", exist_ok=True)

    # --- Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix – {model_name}")
    cm_path = f"models/plots/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # --- ROC Curve Plot ---
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0,1],[0,1],'k--')
        plt.title(f"ROC Curve – {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_path = f"models/plots/{model_name}_roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
    else:
        roc_path = None

    # --- MLflow logging ---
    if log_to_mlflow:
        mlflow.set_experiment("Heart Disease Baselines")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            if auc is not None:
                mlflow.log_metric("roc_auc", auc)

            # log plots if available
            mlflow.log_artifact(cm_path)
            if roc_path is not None:
                mlflow.log_artifact(roc_path)

            # log model
            mlflow.sklearn.log_model(model, name=f"{model_name}_model")

    return acc, f1, auc, y_pred, y_proba
