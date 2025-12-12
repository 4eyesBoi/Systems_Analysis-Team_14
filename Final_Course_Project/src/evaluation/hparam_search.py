#!/usr/bin/env python3
"""
Randomized hyperparameter search for RandomForestRegressor.
Saves best model as: results/models/rf_best_randomized.pkl
And a CSV summary of search results in results/ml/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import joblib

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("hparam_search")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "psychopathy_DOOM_DATA.csv"
RESULTS_MODELS = ROOT / "results" / "models"
RESULTS_MODELS.mkdir(parents=True, exist_ok=True)
RESULTS_ML = ROOT / "results" / "ml"
RESULTS_ML.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"DOOM DATA not found: {DATA}")

df = pd.read_csv(DATA)
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"

X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(-1)
y = df[target].values

# small train/val split to speed up search
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["auto", "sqrt", 0.2, 0.5, 0.8]
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rs = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

logger.info("Starting RandomizedSearchCV (this may take a while)...")
rs.fit(X_train, y_train)

best = rs.best_estimator_
logger.info("Best params: %s", rs.best_params_)
logger.info("Best score (neg MSE): %s", rs.best_score_)

# Save best model
joblib.dump(best, RESULTS_MODELS / "rf_best_randomized.pkl")
logger.info("Saved best model to %s", RESULTS_MODELS / "rf_best_randomized.pkl")

# Save CV results
cvres = pd.DataFrame(rs.cv_results_)
cvres.to_csv(RESULTS_ML / "rf_randomized_search_results.csv", index=False)
logger.info("Saved CV results to %s", RESULTS_ML / "rf_randomized_search_results.csv")
