# scenario_1_ml_simulation.py
# Ubicado en: project_root/src/scenario_1_ml_simulation.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
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

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

# ======================================================
# 1. PATH SETUP
# ======================================================

ROOT = Path(__file__).resolve().parent.parent       # project root
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "ml"
OUT.mkdir(parents=True, exist_ok=True)

# ======================================================
# 2. LOAD DOOM DATA
# ======================================================

CSV = DATA_DIR / "processed" / "psychopathy_DOOM_DATA.csv"
if not CSV.exists():
    logger.error(f"DOOM DATA no encontrada en: {CSV}")
    raise FileNotFoundError(f"❌ No se encontró el archivo DOOM DATA en: {CSV}")

logger.info(f"Cargando datos desde: {CSV}")
df = pd.read_csv(CSV)

# Detecting target automatically
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"
logger.info(f"Target column: {target}")

X_full = (
    df.drop(columns=[target])
    .select_dtypes(include=[np.number])
    .fillna(-1)
)
y_full = df[target]

if X_full.shape[0] == 0:
    logger.error("Dataset está vacío después del procesamiento")
    raise ValueError("Dataset vacío")

logger.info(f"Dataset shape: {X_full.shape}, Samples: {len(y_full)}")

# ======================================================
# 3. PART 1 — MODEL STABILITY
# ======================================================

seeds = [1, 5, 10, 21, 42, 99]
mse_list = []
mae_list = []
rmse_list = []

logger.info("Evaluando estabilidad del modelo con múltiples seeds...")

for s in seeds:
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=s)

    model = RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mse)
    
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)

    logger.info(f"Seed {s}: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

# ---------- plot MSE vs Seed ----------
plt.figure(figsize=(8, 4))
plt.plot(seeds, mse_list, marker='o')
plt.title("MSE across different seeds")
plt.xlabel("Seed")
plt.ylabel("MSE")
plt.savefig(OUT / "ml_mse_seeds.png", dpi=150)
plt.close()

# ---------- boxplot ----------
plt.figure(figsize=(6, 4))
sns.boxplot(data=mse_list)
plt.title("Distribution of MSEs (seeds)")
plt.savefig(OUT / "ml_mse_box.png", dpi=150)
plt.close()

# ======================================================
# 4. PART 2 — REAL VS PREDICTED DISTRIBUTION
# ======================================================

# Run canonical seed 42
logger.info("Running model with seed 42 for distribution analysis...")
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Validate predictions
if len(preds) == 0:
    logger.error("No se generaron predicciones")
    raise ValueError("Predicciones vacías")

logger.info(f"Predicciones: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")

plt.figure(figsize=(8, 4))
sns.histplot(y_test, label="True", kde=True, stat="density", alpha=0.5)
sns.histplot(preds, label="Pred", kde=True, stat="density", alpha=0.5)
plt.legend()
plt.title("Distribution: True vs Predicted (seed=42)")
plt.savefig(OUT / "hist_true_vs_pred.png", dpi=150)
plt.close()

# ======================================================
# 5. PART 3 — VARIABLE IMPORTANCE
# ======================================================

feat_imp = model.feature_importances_
feat_names = X_full.columns
idx = np.argsort(feat_imp)[-20:]   # top 20

plt.figure(figsize=(8, 6))
plt.barh(range(len(idx)), feat_imp[idx])
plt.yticks(range(len(idx)), feat_names[idx])
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig(OUT / "feature_importance_top20.png", dpi=150)
plt.close()

# ======================================================
# 6. PART 4 — NOISE SENSITIVITY
# ======================================================

logger.info("Evaluating noise sensitivity (chaos analysis)...")
noise_levels = [0.01, 0.03, 0.05, 0.1]
mse_pert = []
mae_pert = []

for nl in noise_levels:
    X_test_noisy = X_test.copy()

    for col in X_test_noisy.columns:
        sigma = X_test_noisy[col].std()
        noise = np.random.normal(0, nl * sigma, size=X_test_noisy.shape[0])
        X_test_noisy[col] += noise

    preds_noisy = model.predict(X_test_noisy)
    mse_n = mean_squared_error(y_test, preds_noisy)
    mae_n = mean_absolute_error(y_test, preds_noisy)
    mse_pert.append(mse_n)
    mae_pert.append(mae_n)

    logger.info(f"Noise {nl*100:.1f}%: MSE={mse_n:.6f}, MAE={mae_n:.6f}")

plt.figure(figsize=(7, 4))
plt.plot([l * 100 for l in noise_levels], mse_pert, marker='o')
plt.xlabel("Noise level (% of std)")
plt.ylabel("MSE on noisy test set")
plt.title("Sensitivity to Input Noise")
plt.savefig(OUT / "sensitivity_noise.png", dpi=150)
plt.close()

# ======================================================
# 7. SAVE RESULTS
# ======================================================

pd.DataFrame({"seed": seeds, "mse": mse_list, "mae": mae_list, "rmse": rmse_list}).to_csv(
    OUT / "ml_seed_results.csv", index=False
)

pd.DataFrame({
    "noise_level_frac": noise_levels,
    "mse": mse_pert,
    "mae": mae_pert
}).to_csv(OUT / "ml_noise_results.csv", index=False)

logger.info("ML simulation completed successfully.")
logger.info(f"Results saved in: {OUT}")
logger.info(f"Summary - Mean MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
logger.info(f"Summary - Mean MAE: {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}")
