
# train_baselines.py
# Baselines for classification + regression using MLflow + centralized evaluation

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import Binarizer, MinMaxScaler

import joblib

# Project-local utilities
import data as data_src
from features import preprocessor, build_preprocessor
from evaluate import evaluate_classification_model, evaluate_regression_model


# ------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------
df = data_src.load_dataset()

# Classification target
X_cls = df.drop(columns=['presence', 'thalach'], errors='ignore')
y_cls = df['presence']

# Regression target: predict 'thalach' (max heart rate) when present
if 'thalach' in df.columns:
    X_reg = df.drop(columns=['thalach'], errors='ignore')
    y_reg = df['thalach']
else:
    X_reg, y_reg = None, None


# ------------------------------------------------------------------------------------
# Classification Baselines
# ------------------------------------------------------------------------------------
def run_classification_baselines(save_models: bool = True):
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    results = {}

    prep = build_preprocessor(task="classification")

    # 1) GaussianNB
    pipe_gnb = Pipeline([('preprocessor', prep), ('model', GaussianNB())])
    pipe_gnb.fit(X_train, y_train)
    acc, f1, auc, _, _ = evaluate_classification_model(pipe_gnb, X_test, y_test, model_name="GaussianNB")
    results['GaussianNB'] = (acc, f1, auc)
    if save_models: joblib.dump(pipe_gnb, "models/gaussian_nb_pipeline.pkl")

    # 2) BernoulliNB (with binarizer on preprocessed features)
    pipe_bnb = Pipeline([('preprocessor', prep), ('binarizer', Binarizer(threshold=0.0)), ('model', BernoulliNB())])
    pipe_bnb.fit(X_train, y_train)
    acc, f1, auc, _, _ = evaluate_classification_model(pipe_bnb, X_test, y_test, model_name="BernoulliNB")
    results['BernoulliNB'] = (acc, f1, auc)

    # 3) MultinomialNB (needs non-negative)
    pipe_mnb = Pipeline([('preprocessor', prep), ('scaler', MinMaxScaler()), ('model', MultinomialNB())])
    pipe_mnb.fit(X_train, y_train)
    acc, f1, auc, _, _ = evaluate_classification_model(pipe_mnb, X_test, y_test, model_name="MultinomialNB")
    results['MultinomialNB'] = (acc, f1, auc)

    # 4) Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
    pipe_lr = Pipeline([('preprocessor', prep), ('model', log_reg)])
    pipe_lr.fit(X_train, y_train)
    acc, f1, auc, _, _ = evaluate_classification_model(pipe_lr, X_test, y_test, model_name="LogisticRegression")
    results['LogisticRegression'] = (acc, f1, auc)

    # 5) Decision Tree variants
    base_dt = Pipeline([('preprocessor', prep), ('model', DecisionTreeClassifier(random_state=42, criterion='entropy'))])
    base_dt.fit(X_train, y_train)
    acc, f1, auc, _, _ = evaluate_classification_model(base_dt, X_test, y_test, model_name="DecisionTree_entropy")
    results['DecisionTree_entropy'] = (acc, f1, auc)

    gini_dt = Pipeline([('preprocessor', prep), ('model', DecisionTreeClassifier(random_state=2, criterion='gini'))])
    gini_dt.fit(X_train, y_train)
    acc, f1, auc, _, _ = evaluate_classification_model(gini_dt, X_test, y_test, model_name="DecisionTree_gini")
    results['DecisionTree_gini'] = (acc, f1, auc)

    grid_dt = Pipeline([('preprocessor', prep), ('model', DecisionTreeClassifier(random_state=42))])
    grid_params = {
        "model__criterion": ['entropy', 'gini'],
        "model__min_samples_leaf": [2, 3, 4, 5, 7],
        "model__max_depth": [1, 2, 3, 5, 7, None],
    }
    gs = GridSearchCV(grid_dt, grid_params, scoring='f1_macro', cv=7)
    gs.fit(X_train, y_train)
    best_dt = gs.best_estimator_
    acc, f1, auc, _, _ = evaluate_classification_model(best_dt, X_test, y_test, model_name="DecisionTree_tuned")
    results['DecisionTree_tuned'] = (acc, f1, auc)

    # 6) Random Forest + tuned
    rf_base = Pipeline([('preprocessor', prep), ('model', RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced_subsample'))])
    rf_base.fit(X_train, y_train)
    acc, f1, auc, _, _ = evaluate_classification_model(rf_base, X_test, y_test, model_name="RandomForest")
    results['RandomForest'] = (acc, f1, auc)

    rf_grid = {
        'model__n_estimators': [100, 150, 200, 300],
        'model__max_depth': [None, 10, 15, 20],
        'model__min_samples_split': [2, 5, 10]
    }
    rf_gs = GridSearchCV(rf_base, rf_grid, scoring='f1_macro', cv=7)
    rf_gs.fit(X_train, y_train)
    rf_best = rf_gs.best_estimator_
    acc, f1, auc, _, _ = evaluate_classification_model(rf_best, X_test, y_test, model_name="RandomForest_tuned")
    results['RandomForest_tuned'] = (acc, f1, auc)

    # Optional: 7-fold CV summary on F1
    cv = KFold(n_splits=7, shuffle=True, random_state=42)
    for name, pipe in {
        "GaussianNB": pipe_gnb, "BernoulliNB": pipe_bnb, "MultinomialNB": pipe_mnb,
        "LogisticRegression": pipe_lr, "DecisionTree_entropy": base_dt,
        "DecisionTree_gini": gini_dt, "DecisionTree_tuned": best_dt,
        "RandomForest": rf_base, "RandomForest_tuned": rf_best
    }.items():
        scores = cross_val_score(pipe, X_cls, y_cls, cv=cv, scoring='f1_macro')
        print(f"{name}: mean F1 = {scores.mean():.3f} ± {scores.std():.3f}")

    return results


# ------------------------------------------------------------------------------------
# Regression Baselines (predict 'thalach' if available)
# ------------------------------------------------------------------------------------
def run_regression_baselines(save_models: bool = False):
    if X_reg is None:
        print("Skipping regression: 'thalach' not present in dataset.")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    results = {}

    # Linear
    lin_pipe = Pipeline([('processor', preprocessor("linear")), ('regressor', LinearRegression())])
    lin_pipe.fit(X_train, y_train)
    rmse, mae, r2, _ = evaluate_regression_model(lin_pipe, X_test, y_test, model_name="LinearRegression")
    results['LinearRegression'] = (rmse, mae, r2)

    # Polynomial
    poly_pipe = Pipeline([('processor', preprocessor("polynomial")), ('regressor', LinearRegression())])
    poly_pipe.fit(X_train, y_train)
    rmse, mae, r2, _ = evaluate_regression_model(poly_pipe, X_test, y_test, model_name="PolynomialRegression")
    results['PolynomialRegression'] = (rmse, mae, r2)

    # Root (sqrt-like) transform
    sqrt_pipe = Pipeline([('processor', preprocessor("sqrt")), ('regressor', LinearRegression())])
    sqrt_pipe.fit(X_train, y_train)
    rmse, mae, r2, _ = evaluate_regression_model(sqrt_pipe, X_test, y_test, model_name="SqrtTransformRegression")
    results['SqrtTransformRegression'] = (rmse, mae, r2)

    # Decision Tree
    dt_pipe = Pipeline([('processor', preprocessor("linear")), ('regressor', DecisionTreeRegressor(random_state=3))])
    dt_pipe.fit(X_train, y_train)
    rmse, mae, r2, _ = evaluate_regression_model(dt_pipe, X_test, y_test, model_name="DecisionTreeRegressor")
    results['DecisionTreeRegressor'] = (rmse, mae, r2)

    # Tuned Decision Tree
    dt_grid = Pipeline([('processor', preprocessor("linear")), ('regressor', DecisionTreeRegressor(random_state=3))])
    hyp_params = {
        'regressor__max_depth': [2, 3, 4, 5, 6, 7, 8, None],
        'regressor__min_samples_leaf': [1, 2, 3, 4, 5, 8],
    }
    gs = GridSearchCV(dt_grid, hyp_params, scoring='neg_mean_squared_error', cv=7)
    gs.fit(X_train, y_train)
    best_dt = gs.best_estimator_
    rmse, mae, r2, _ = evaluate_regression_model(best_dt, X_test, y_test, model_name="DecisionTreeRegressor_tuned")
    results['DecisionTreeRegressor_tuned'] = (rmse, mae, r2)

    # 7-fold CV on RMSE
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv = KFold(n_splits=7, shuffle=True, random_state=3)
    for name, model in {
        "LinearRegression": lin_pipe,
        "PolynomialRegression": poly_pipe,
        "SqrtTransformRegression": sqrt_pipe,
        "DecisionTreeRegressor": dt_pipe,
        "DecisionTreeRegressor_tuned": best_dt
    }.items():
        neg_mse_scores = cross_val_score(model, X_reg, y_reg, scoring=scorer, cv=cv)
        rmse_scores = np.sqrt(-neg_mse_scores)
        print(f"{name}: Mean RMSE = {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")

    return results


if __name__ == "__main__":
    print("=== Running Classification Baselines ===")
    cls_results = run_classification_baselines()
    print("\n=== Running Regression Baselines ===")
    reg_results = run_regression_baselines()
    print("\nDone.")
