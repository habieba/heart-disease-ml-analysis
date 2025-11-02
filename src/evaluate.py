# evaluate.py

import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, mean_squared_error, mean_absolute_error, r2_score
)


def _ensure_plot_dir():
    os.makedirs("models/plots", exist_ok=True)


def evaluate_classification_model(model, X_test, y_test, model_name="Model", log_to_mlflow=True):
    """
    Universal evaluation for classification models.
    Computes metrics, confusion matrix, ROC curve, and logs to MLflow.
    Returns: acc, f1, auc, y_pred, y_proba
    """
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\\n📊 {model_name} Evaluation:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}" if auc is not None else "  ROC-AUC  : N/A (no predict_proba)")

    _ensure_plot_dir()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix – {model_name}")
    cm_path = f"models/plots/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # ROC Curve
    roc_path = None
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f"ROC Curve – {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_path = f"models/plots/{model_name}_roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

    if log_to_mlflow:
        mlflow.set_experiment("Heart Disease Baselines")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            if auc is not None:
                mlflow.log_metric("roc_auc", auc)
            mlflow.log_artifact(cm_path)
            if roc_path is not None:
                mlflow.log_artifact(roc_path)
            mlflow.sklearn.log_model(model, name=f"{model_name}_model")

    return acc, f1, auc, y_pred, y_proba


def evaluate_regression_model(model, X_test, y_test, model_name="Regressor", log_to_mlflow=True):
    """
    Universal evaluation for regression models.
    Computes RMSE/MAE/R2, creates parity plot, residual histogram,
    and residuals-vs-predicted scatter; logs all to MLflow.
    Returns: rmse, mae, r2, y_pred
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\\n📈 {model_name} Evaluation:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R^2  : {r2:.4f}")

    _ensure_plot_dir()

    # Parity plot
    import numpy as np
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.7)
    line_min = float(np.min([y_test.min(), y_pred.min()]))
    line_max = float(np.max([y_test.max(), y_pred.max()]))
    plt.plot([line_min, line_max], [line_min, line_max], 'k--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Parity Plot – {model_name}")
    parity_path = f"models/plots/{model_name}_parity.png"
    plt.savefig(parity_path)
    plt.close()

    # Residuals
    residuals = y_test - y_pred

    # Residuals vs Predicted
    res_vs_pred_path = None
    try:
        plt.figure()
        plt.scatter(y_pred, residuals, alpha=0.7)
        plt.axhline(0, linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residual (True - Pred)")
        plt.title(f"Residuals vs Predicted – {model_name}")
        res_vs_pred_path = f"models/plots/{model_name}_residuals_vs_pred.png"
        plt.savefig(res_vs_pred_path)
        plt.close()
    except Exception:
        res_vs_pred_path = None

    # Residuals histogram
    plt.figure()
    plt.hist(residuals, bins=20)
    plt.title(f"Residuals – {model_name}")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    resid_path = f"models/plots/{model_name}_residuals.png"
    plt.savefig(resid_path)
    plt.close()

    if log_to_mlflow:
        mlflow.set_experiment("Heart Disease Baselines")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_artifact(parity_path)
            mlflow.log_artifact(resid_path)
            if res_vs_pred_path is not None:
                mlflow.log_artifact(res_vs_pred_path)
            mlflow.sklearn.log_model(model, name=f"{model_name}_model")

    return rmse, mae, r2, y_pred
