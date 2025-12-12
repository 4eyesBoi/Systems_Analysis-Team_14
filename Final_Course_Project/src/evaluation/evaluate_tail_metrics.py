#!/usr/bin/env python3
"""
Evaluate tail metrics for the psychopathy prediction model.

This script:
 - Loads DOOM DATA from data/
 - Loads trained model from results/rf_final.pkl
 - Computes:
      * Global MSE, MAE, RMSE
      * Tail metrics at multiple percentiles (P25, P50, P75, P90, P95, P99)
 - Saves:
      * metrics_tail_multi.csv
      * tail_distribution.png
      * tail_bar_comparison.png
      * tail_percentile_analysis.png
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import logging
import warnings

# ======================================================
# LOGGING SETUP
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]       # project root
DATA_PATH = ROOT / "data" / "processed" / "psychopathy_DOOM_DATA.csv"
MODEL_PATH = ROOT / "results" / "rf_final.pkl"
OUT_DIR = ROOT / "results" / "tail_metrics"

OUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Output directory: {OUT_DIR}")

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
if not DATA_PATH.exists():
    logger.error(f"Data not found: {DATA_PATH}")
    raise FileNotFoundError(f"Data not found: {DATA_PATH}")

if not MODEL_PATH.exists():
    logger.error(f"Model not found: {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

logger.info(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Determine target exactly like your model
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"

logger.info(f"Using target column: {target}")

X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(-1)
y = df[target]

if X.shape[0] == 0:
    logger.error("Dataset is empty after processing")
    raise ValueError("Empty dataset")

logger.info(f"Dataset shape: {X.shape}")

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
logger.info(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
preds = model.predict(X)

logger.info(f"Predictions - min: {preds.min():.4f}, max: {preds.max():.4f}, mean: {preds.mean():.4f}")

# ----------------------------------------------------------
# Compute tail (P90)
# ----------------------------------------------------------
p90 = np.percentile(y, 90)
mask_tail = y > p90

y_tail = y[mask_tail]
pred_tail = preds[mask_tail]

# ----------------------------------------------------------
# Compute metrics
# ----------------------------------------------------------
mse_global = mean_squared_error(y, preds)
mae_global = mean_absolute_error(y, preds)
rmse_global = np.sqrt(mse_global)

mse_tail = mean_squared_error(y_tail, pred_tail)
mae_tail = mean_absolute_error(y_tail, pred_tail)
rmse_tail = np.sqrt(mse_tail)

metrics = pd.DataFrame({
    "metric": ["MSE", "MAE", "RMSE"],
    "global": [mse_global, mae_global, rmse_global],
    "tail_P90": [mse_tail, mae_tail, rmse_tail]
})

metrics.to_csv(OUT_DIR / "metrics_tail.csv", index=False)
logger.info("Saved metrics to: metrics_tail.csv")

# ----------------------------------------------------------
# Plot 1: Distribution with tail highlighted
# ----------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(y, bins=40, alpha=0.6, label="All Data")
plt.axvline(p90, color="red", linestyle="--", label="P90 threshold")
plt.title("Target Distribution with P90 Tail")
plt.xlabel("Psychopathy (synth)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "tail_distribution.png", dpi=150)
plt.close()

# ----------------------------------------------------------
# Plot 2: bar comparison
# ----------------------------------------------------------
plt.figure(figsize=(8,5))
labels = ["Global", "Tail P90"]

plt.bar(labels, [rmse_global, rmse_tail], color=["blue", "red"], alpha=0.7)
plt.title("RMSE: Global vs Tail")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(OUT_DIR / "tail_bar_comparison.png", dpi=150)
plt.close()

# ----------------------------------------------------------
# MULTI-PERCENTILE ANALYSIS (NEW)
# ----------------------------------------------------------
logger.info("Computing multi-percentile analysis...")

percentiles = [25, 50, 75, 90, 95, 99]
metrics_multi = {"percentile": [], "mse": [], "mae": [], "rmse": [], "n_samples": []}

for p in percentiles:
    threshold = np.percentile(y, p)
    mask = y >= threshold
    
    y_p = y[mask]
    pred_p = preds[mask]
    
    mse_p = mean_squared_error(y_p, pred_p)
    mae_p = mean_absolute_error(y_p, pred_p)
    rmse_p = np.sqrt(mse_p)
    
    metrics_multi["percentile"].append(f"P{p}")
    metrics_multi["mse"].append(mse_p)
    metrics_multi["mae"].append(mae_p)
    metrics_multi["rmse"].append(rmse_p)
    metrics_multi["n_samples"].append(len(y_p))
    
    logger.info(f"P{p} (n={len(y_p)}): MSE={mse_p:.6f}, MAE={mae_p:.6f}, RMSE={rmse_p:.6f}")

df_multi = pd.DataFrame(metrics_multi)
df_multi.to_csv(OUT_DIR / "metrics_tail_multi_percentile.csv", index=False)
logger.info("Saved multi-percentile metrics to: metrics_tail_multi_percentile.csv")

# Plot multi-percentile comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(metrics_multi["percentile"], metrics_multi["mse"], marker='o', color='blue')
axes[0].set_title("MSE across Percentiles")
axes[0].set_ylabel("MSE")
axes[0].grid()

axes[1].plot(metrics_multi["percentile"], metrics_multi["mae"], marker='s', color='green')
axes[1].set_title("MAE across Percentiles")
axes[1].set_ylabel("MAE")
axes[1].grid()

axes[2].plot(metrics_multi["percentile"], metrics_multi["rmse"], marker='^', color='red')
axes[2].set_title("RMSE across Percentiles")
axes[2].set_ylabel("RMSE")
axes[2].grid()

plt.tight_layout()
plt.savefig(OUT_DIR / "tail_percentile_analysis.png", dpi=150)
plt.close()

logger.info("Tail metrics evaluation complete.")
logger.info(f"Files saved to: {OUT_DIR}")
