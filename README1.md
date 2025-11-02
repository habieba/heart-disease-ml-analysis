# heart-disease-ml-analysis
Machine learning analysis of the UCI Heart Disease dataset featuring classification (disease presence) and regression models (cholesterol/max heart rate prediction). Includes data preprocessing, model comparison, and performance evaluation.

##  What's inside
- `data1.py` — single source of truth for loading/cleaning the dataset.
- `features.py` — preprocessing pipelines (numeric/categorical, polynomial/sqrt/linear).
- `train_baselines.py` — runs a suite of **classification** and **regression** baselines and logs everything to MLflow.
- `evaluate.py` — central evaluation functions (metrics + plots) for both tasks.
- `models/` — models and plots output directory (created at runtime).
- `requirements.txt` — Python dependencies.
- `README.md` — you are here.

> Classification target: `presence` (heart disease).  
> Optional regression target: `thalach` (max heart rate) if available.

## Quickstart


### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run baselines
```bash
python train_baselines.py
```
This will:
- Load data via `data.py`
- Fit multiple models (GaussianNB, BernoulliNB, MultinomialNB, Logistic Regression, Decision Trees, Random Forests; plus regressors if `thalach` exists)
- Log metrics, parameters, plots, and models to MLflow
- Save pipelines and plots under `models/`

### 4) Open the MLflow UI
```bash
mlflow ui
```
Then open the printed URL in your browser (default: http://127.0.0.1:5000).  
You’ll see the **Heart Disease Baselines** experiment with runs, metrics, and artifacts (confusion matrix, ROC, parity/residuals).

---

## Key modules

### `data.py`
- Provides `load_dataset()` to fetch and clean the UCI dataset into a Cleveland-like schema.
- Acts as the *single source of truth* for data across the repo.

### `features.py`
- Exposes `preprocessor(kind="linear"|"polynomial"|"sqrt")` and `build_preprocessor()`.
- Handles numeric imputation/scaling and categorical encoding.

### `evaluate.py`
- `evaluate_classification_model(...)`: accuracy, F1, ROC-AUC, confusion matrix, ROC curve, MLflow logging.
- `evaluate_regression_model(...)`: RMSE, MAE, R², parity & residual plots, MLflow logging.
- Plots are written to `models/plots/` and logged as MLflow artifacts.

### `train_baselines.py`
- Splits data, builds pipelines, trains baselines, and delegates evaluation to `evaluate.py`.
- Saves fitted pipelines (e.g., `models/gaussian_nb_pipeline.pkl`).

---


---

## License
MIT (or adjust to your preferred license).
