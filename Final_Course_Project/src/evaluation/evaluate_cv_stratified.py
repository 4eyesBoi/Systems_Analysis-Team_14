#!/usr/bin/env python3
"""
Stratified cross-validation focused on tail performance (P90).

Saves:
 - results/ml/stratified_cv_metrics.csv
 - results/ml/stratified_cv_summary.txt
 - results/ml/stratified_cv_tail_plot.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("strat_cv")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "psychopathy_DOOM_DATA.csv"
MODEL_OUT = ROOT / "results" / "ml"
MODEL_OUT.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"DOOM DATA not found: {DATA}")

df = pd.read_csv(DATA)
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"
logger.info("Using target: %s", target)

# Prepare features
X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(-1)
y = df[target]

# Create bins for stratification (by quantiles)
n_bins = 10
bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
logger.info("Created %d bins for stratification", bins.nunique())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
tail_results = []

fold = 0
p90_train_vals = []
for train_idx, test_idx in skf.split(X, bins):
    fold += 1
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # compute P90 on train
    p90 = np.percentile(y_train, 90)
    p90_train_vals.append(p90)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)

    # tail in this fold
    mask_tail = y_test > p90
    if mask_tail.sum() > 0:
        mse_tail = mean_squared_error(y_test[mask_tail], preds[mask_tail])
        mae_tail = mean_absolute_error(y_test[mask_tail], preds[mask_tail])
        rmse_tail = np.sqrt(mse_tail)
    else:
        mse_tail = np.nan; mae_tail = np.nan; rmse_tail = np.nan

    results.append({
        "fold": fold,
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "p90": float(p90),
        "tail_count": int(mask_tail.sum())
    })

    tail_results.append({
        "fold": fold,
        "mse_tail": float(mse_tail) if not np.isnan(mse_tail) else None,
        "mae_tail": float(mae_tail) if not np.isnan(mae_tail) else None,
        "rmse_tail": float(rmse_tail) if not np.isnan(rmse_tail) else None
    })

# Save CSVs
df_results = pd.DataFrame(results)
df_tail = pd.DataFrame(tail_results)

df_results.to_csv(MODEL_OUT / "stratified_cv_metrics.csv", index=False)
df_tail.to_csv(MODEL_OUT / "stratified_cv_tail_metrics.csv", index=False)

# Summary
summary = {
    "mse_mean": df_results["mse"].mean(),
    "mse_std": df_results["mse"].std(),
    "mse_tail_mean": df_tail["mse_tail"].dropna().mean(),
    "mse_tail_std": df_tail["mse_tail"].dropna().std(),
    "n_folds": len(df_results)
}

with open(MODEL_OUT / "stratified_cv_summary.txt", "w", encoding="utf-8") as f:
    f.write("Stratified CV summary (tail-aware)\n")
    f.write("="*60 + "\n")
    for k,v in summary.items():
        f.write(f"{k}: {v}\n")
logger.info("Saved stratified CV results to %s", MODEL_OUT)

# Plot tail counts per fold
plt.figure(figsize=(6,4))
plt.bar(df_results["fold"], df_results["tail_count"])
plt.xlabel("Fold")
plt.ylabel("Tail samples (count)")
plt.title("Tail samples (y > P90_train) per fold")
plt.tight_layout()
plt.savefig(MODEL_OUT / "stratified_cv_tail_plot.png", dpi=150)
plt.close()
logger.info("Saved tail plot to %s", MODEL_OUT / "stratified_cv_tail_plot.png")
