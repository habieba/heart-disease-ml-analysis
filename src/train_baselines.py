#Necessary imports
from features import *
import numpy as np
from data import X, y_regression, y_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""Trains, evaluates, and compares several baseline regression, and classification models.

This script performs a comparative analysis of different regression and classification models to
establish a performance baseline for the given dataset. It splits the data,
then builds and trains the following models:
- Standard Linear Regression
- Polynomial Regression (degree 2)
- Linear Regression with root-transformed features (power 0.2)
- A default Decision Tree Regressor
- An optimized Decision Tree Regressor (tuned via GridSearchCV)
- Two Decision Tree classifiers with gini and entropy criterion respectively
- An optimized Decision Tree classifier (tuned via GridSearchCV)

The script evaluates each Regression model using Root Mean Squared Error (RMSE) on the
test set, and Accuracy score for the Classification model with a confusion matrix for more
insight on the model's performance

Output:
    - The best hyperparameters found for the Decision Tree Regressor.
    - A final summary of the RMSE scores for all tested regression models.
    - The best hyperparameters found for the Decision Trees.
    - A confusion matrix plot describing the model's performance.
"""

dashes = 100 #for pretty print

#splitting data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y_regression, test_size=0.2, random_state=42)

#---------------------------------------------------------------------------------------------------------------
# Regression Models

#base Linear Regression Model
linear_transformer = preprocessor("linear")
baseline_linear_regressor = Pipeline(
    steps=[('processor', linear_transformer),
           ('regression', LinearRegression())]
)

baseline_linear_regressor.fit(X_train, y_train)
y_pred_base = baseline_linear_regressor.predict(X_test)
mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = np.sqrt(mse_base)


#LinearRegression with polynomial degrees
poly_transformer = preprocessor("polynomial")
poly_regressor = Pipeline(steps=[
            ('processor', poly_transformer),
            ('regression', LinearRegression())]
)

#training and testing poly_regressor
poly_regressor.fit(X_train, y_train)
y_pred_poly = poly_regressor.predict(X_test)
mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)


#LinearRegression with root polynomial degrees
rt_transformer = preprocessor("sqrt")
root_poly_regressor = Pipeline(steps=[
            ('processor', rt_transformer),
            ('regression', LinearRegression(s))]
)

#training and testing root_poly_regressor
root_poly_regressor.fit(X_train, y_train)
y_pred_rt = root_poly_regressor.predict(X_test)
mse_rt = mean_squared_error(y_test, y_pred_rt)
rmse_rt = np.sqrt(mse_rt)


#Decision Tree Regressor to see performance improves
dt_transformer = preprocessor("")
dt_regressor = Pipeline(steps=[
                ('processor', dt_transformer),
                ('regressor', DecisionTreeRegressor(random_state=3))]
)

#training and testing dt_regressor
dt_regressor.fit(X_train, y_train)
y_pred_dt = dt_regressor.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)


#Decision Tree Regressor with HyperParameters
hyp_param_dt_transformer = preprocessor("")
hyp_param_dt_regressor = Pipeline(steps=[
                ('processor', hyp_param_dt_transformer),
                ('regressor', DecisionTreeRegressor(random_state=3))]
)
hyp_params = {
    'regressor__max_depth': [2, 3, 4, 5, 6, 7, 8, None],
    'regressor__min_samples_leaf': [1, 2, 3, 4, 5, 8],
}

#Cross validation to find best hyperparameters
grid_search = GridSearchCV(
    estimator= hyp_param_dt_regressor,
    param_grid= hyp_params,
    scoring= 'neg_mean_squared_error',
    cv=7
)

grid_search.fit(X_train, y_train)
best_dt_regressor = grid_search.best_estimator_
y_pred_dt_hyp = best_dt_regressor.predict(X_test)
mse_hyp = mean_squared_error(y_test, y_pred_dt_hyp)
rmse_hyp = np.sqrt(mse_hyp)

#prints the best hyperparameters
print(f"\nBest Hyperparameters: {grid_search.best_params_}\n")

#Organised view of Regression Models for comparison
model_scores= {
    "Linear Regression": rmse_base,
    "Polynomial Regression": rmse_poly,
    "Square Root Regression": rmse_rt,
    "Decision Tree Regressor": rmse_dt,
    "Optimized Decision Tree Regressor": rmse_hyp
}

print("-"*dashes)

for model_name, score in model_scores.items():
    print(f"{model_name} RMSE: {score}")

print("-"*dashes)

#Cross-Validation
scorer = make_scorer(mean_squared_error, greater_is_better=False)
cv = KFold(n_splits=7, shuffle=True, random_state=3)

print("\nCross-Validation Results (7-fold):")
cv_results = {}

models = {
    "Linear Regression": baseline_linear_regressor,
    "Polynomial Regression": poly_regressor,
    "Square Root Regression": root_poly_regressor,
    "Decision Tree Regressor": dt_regressor,
    "Optimized Decision Tree Regressor": best_dt_regressor
}

print("-"*dashes)

for model_name, model in models.items():
    neg_mse_scores = cross_val_score(model, X, y_regression, scoring=scorer, cv=cv)
    rmse_scores = np.sqrt(-neg_mse_scores)
    cv_results[model_name] = (rmse_scores.mean(), rmse_scores.std())
    print(f"{model_name}: Mean RMSE = {rmse_scores.mean():.3f} ± {rmse_scores.std():.3f}")

print("-"*dashes)

#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------

#splitting the dataset
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)

#Classification Models

#Decision Trees

#decision tree with entropy criterion
baseline_classification = preprocessor('')
baseline_dt = Pipeline(steps=[
    ('processor', baseline_classification),
    ('classification', DecisionTreeClassifier(random_state=42, criterion='entropy'))
])

#training
baseline_dt.fit(X_train_class, y_train_class)
y_pred_base_dt = baseline_dt.predict(X_test_class)
accuracy_score_dt_entropy = accuracy_score(y_test_class, y_pred_base_dt)


#decision tree with gini criterion
dt_gini = Pipeline(steps=[
    ('processor', baseline_classification),
    ('classification', DecisionTreeClassifier(random_state=2, criterion='gini'))
])

#training
dt_gini.fit(X_train_class, y_train_class)
y_pred_gini = dt_gini.predict(X_test_class)
accuracy_score_dt_gini = accuracy_score(y_test_class, y_pred_gini)

#adding hyperparameters using cross-validation to try and improve efficiency

dt_hyper = Pipeline(steps=[
    ('processor',  baseline_classification),
    ('classification', DecisionTreeClassifier(random_state=42))
]
)

hyp_params_classification = {
    "classification__criterion": ['entropy', 'gini'],
    "classification__min_samples_leaf": [2, 3, 4, 5, 7],
    "classification__max_depth": [1, 2, 3, 5, 7, None],
}

grid_search_classification = GridSearchCV(
    estimator= dt_hyper,
    scoring= 'f1_macro',
    param_grid= hyp_params_classification,
    cv= 7
)

#training
grid_search_classification.fit(X_train_class, y_train_class)
best_dt_classifier = grid_search_classification.best_estimator_
y_pred_dtc_opt = best_dt_classifier.predict(X_test_class)
accuracy_score_dtc_opt = accuracy_score(y_test_class, y_pred_dtc_opt)


#Random Forest
rf = Pipeline([
    ('processor', baseline_classification),
    ('classification', RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced_subsample'))
])

#training
rf.fit(X_train_class, y_train_class)
y_pred_rf = rf.predict(X_test_class)
accuracy_score_rf = accuracy_score(y_test_class, y_pred_rf)

#optimizing Random Forest
param_grid = {
    'classification__n_estimators': [100, 125, 150, 200, 225, 250, 300],
    'classification__max_depth': [None, 10, 12, 15, 20],
    'classification__min_samples_split': [2, 5, 7, 10]
}

rf_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=7
)

#training
rf_search.fit(X_train_class, y_train_class)
opt_rf = rf_search.best_estimator_
y_pred_rf_opt = opt_rf.predict(X_test_class)
accuracy_score_rf_opt = accuracy_score(y_test_class, y_pred_rf_opt)

model_scores = {
    "Decision Trees (criterion: entropy)": accuracy_score_dt_entropy,
    "Decision Trees (criterion: gini)": accuracy_score_dt_gini,
    "Optimized Decision Trees": accuracy_score_dtc_opt,
    "Random Forest": accuracy_score_rf,
    "Optimized Random Forest": accuracy_score_rf_opt,
}

print("-"*dashes)
for model_name, score in model_scores.items():
    print(f"{model_name} Accuracy: {score:.2%}")


#confusion matrix, and classification report per model
models = {
    "Decision Trees (criterion: entropy)": baseline_dt,
    "Decision Trees (criterion: gini)": dt_gini,
    "Optimized Decision Trees": best_dt_classifier,
    "Random Forest": rf,
    "Optimized Random Forest": opt_rf,

}

for name, model in models.items():
    print(f"{name} Accuracy: {model.score(X_test_class, y_test_class):.2%}")
    y_pred = model.predict(X_test_class)
    print("Classification Report:")
    print(classification_report(y_test_class, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_class, y_pred))
    print("-" * dashes)

#Cross-validation for classification models
cv = KFold(n_splits=7, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_val_score(model, X, y_classification, cv=cv, scoring='f1_macro')
    print(f"{name}: mean F1 = {scores.mean():.3f} ± {scores.std():.3f}")
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


