#!/usr/bin/env python3
"""
Bootstrap-based uncertainty estimation.
Trains N bootstrap RF models and computes 5-95% prediction intervals.
Saves:
 - results/ml/bootstrap_intervals.csv
 - results/ml/bootstrap_summary.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("bootstrap_uncertainty")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "psychopathy_DOOM_DATA.csv"
RESULTS_ML = ROOT / "results" / "ml"
RESULTS_ML.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("DOOM DATA not found: %s" % DATA)

df = pd.read_csv(DATA)
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"
X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(-1)
y = df[target].values

# split train/test consistent with pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

n_boot = 20
preds_boot = np.zeros((n_boot, X_test.shape[0]))

logger.info("Training %d bootstrap models...", n_boot)
for i in tqdm(range(n_boot)):
    # sample with replacement from train
    idx = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True)
    X_b = X_train.iloc[idx]
    y_b = y_train[idx]

    m = RandomForestRegressor(n_estimators=200, random_state=42 + i, n_jobs=-1)
    m.fit(X_b, y_b)
    preds_boot[i, :] = m.predict(X_test)

# compute intervals
lower = np.percentile(preds_boot, 5, axis=0)
upper = np.percentile(preds_boot, 95, axis=0)
median = np.median(preds_boot, axis=0)

out_df = pd.DataFrame({
    "uid_index": np.arange(X_test.shape[0]),
    "pred_median": median,
    "pred_lower_5": lower,
    "pred_upper_95": upper,
    "y_true": y_test
})

out_df.to_csv(RESULTS_ML / "bootstrap_intervals.csv", index=False)
logger.info("Saved bootstrap intervals -> %s", RESULTS_ML / "bootstrap_intervals.csv")

# Summary plot for first 200 samples
n_plot = min(200, X_test.shape[0])
plt.figure(figsize=(10,6))
plt.errorbar(np.arange(n_plot), median[:n_plot], yerr=[median[:n_plot]-lower[:n_plot], upper[:n_plot]-median[:n_plot]], fmt='o', alpha=0.6)
plt.scatter(np.arange(n_plot), y_test[:n_plot], c='red', s=10, label='true')
plt.xlabel("Test sample index")
plt.ylabel("Predicted psychopathy")
plt.title("Bootstrap median + 5-95% intervals (first samples)")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_ML / "bootstrap_summary.png", dpi=150)
plt.close()
logger.info("Saved bootstrap summary plot -> %s", RESULTS_ML / "bootstrap_summary.png")
