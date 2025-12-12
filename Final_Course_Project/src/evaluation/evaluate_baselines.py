import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

ROOT = Path(__file__).resolve().parents[2]

# -------------------------
# Load DOOM DATA
# -------------------------
doom_data = ROOT / "data" / "processed" / "psychopathy_DOOM_DATA.csv"

if not doom_data.exists():
    logger.error(f"DOOM DATA not found: {doom_data}")
    raise FileNotFoundError(f"DOOM DATA not found: {doom_data}")

logger.info(f"Loading DOOM data from: {doom_data}")
df_doom = pd.read_csv(doom_data)
target = "psychopathy_synth" if "psychopathy_synth" in df_doom.columns else "psychopathy"
logger.info(f"Target column: {target}")

X_doom = df_doom.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(-1)
y_doom = df_doom[target]

if X_doom.shape[0] == 0:
    logger.error("DOOM dataset is empty after processing")
    raise ValueError("Empty dataset")

logger.info(f"DOOM data shape: {X_doom.shape}")

# -------------------------
# Load original RAW data
# -------------------------
raw_data = ROOT / "data" / "raw" / "PersonalityData_ExternalVersion001.csv"

if not raw_data.exists():
    logger.error(f"RAW data not found: {raw_data}")
    raise FileNotFoundError(f"RAW data not found: {raw_data}")

logger.info(f"Loading RAW data from: {raw_data}")
df_raw = pd.read_csv(raw_data)

# Determine target for RAW dataset separately (may differ from DOOM dataset)
raw_target = "psychopathy_synth" if "psychopathy_synth" in df_raw.columns else "psychopathy"
if raw_target != target:
    logger.warning(f"Target differs: DOOM uses '{target}', RAW uses '{raw_target}'")

X_raw = df_raw.drop(columns=[raw_target]).select_dtypes(include=[np.number]).fillna(-1)
y_raw = df_raw[raw_target]

if X_raw.shape[0] == 0:
    logger.error("RAW dataset is empty after processing")
    raise ValueError("Empty RAW dataset")

logger.info(f"RAW data shape: {X_raw.shape}")

# -------------------------
# Helper for MSE
# -------------------------
def cv_mse(estimator, X, y, name):
    scores = -cross_val_score(
        estimator,
        X, y,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    mean = float(scores.mean())
    std = float(scores.std())
    logger.info(f"{name} | MSE: {mean:.6f} ±{std:.6f}")
    return mean, std

logger.info("\n" + "="*60)
logger.info("EVALUATING BASELINES")
logger.info("="*60)

# collect results to save
_baseline_results = []

# -----------------------------------
# BASELINE 1 — Linear Regression
# -----------------------------------
m, s = cv_mse(LinearRegression(), X_raw, y_raw, "Linear (RAW)")
_baseline_results.append({"model": "Linear (RAW)", "mse_mean": m, "mse_std": s})

# -----------------------------------
# BASELINE 2 — RandomForest (RAW)
# -----------------------------------
rf_raw = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m, s = cv_mse(rf_raw, X_raw, y_raw, "RandomForest (RAW)")
_baseline_results.append({"model": "RandomForest (RAW)", "mse_mean": m, "mse_std": s})

# -----------------------------------
# MODEL FINAL — RandomForest + DOOM DATA
# -----------------------------------
rf_doom = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m, s = cv_mse(rf_doom, X_doom, y_doom, "RandomForest (DOOM DATA)")
_baseline_results.append({"model": "RandomForest (DOOM DATA)", "mse_mean": m, "mse_std": s})

# (OPCIONAL) BASELINE 3 — XGBoost
# -----------------------------------
try:
    import xgboost as xgb
    logger.info("XGBoost available, evaluating...")
    model_xgb = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, verbosity=0)
    m, s = cv_mse(model_xgb, X_doom, y_doom, "XGBoost (DOOM DATA)")
    _baseline_results.append({"model": "XGBoost (DOOM DATA)", "mse_mean": m, "mse_std": s})
except ImportError:
    logger.info("XGBoost not installed. Skipping XGBoost baseline.")
except Exception as e:
    logger.warning(f"XGBoost evaluation failed: {e}")

# save results to results/ml/
results_dir = ROOT / "results" / "ml"
results_dir.mkdir(parents=True, exist_ok=True)
try:
    import json
    df_out = pd.DataFrame(_baseline_results)
    out_csv = results_dir / "baselines_results.csv"
    df_out.to_csv(out_csv, index=False)
    # also save a JSON summary
    out_json = results_dir / "baselines_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_baseline_results, f, indent=2)
    logger.info("Saved baseline results -> %s", out_csv)
except Exception as e:
    logger.warning("No se pudieron guardar los resultados en CSV/JSON: %s", e)

logger.info("="*60)
logger.info("Baselines evaluation complete.")
logger.info("="*60)
