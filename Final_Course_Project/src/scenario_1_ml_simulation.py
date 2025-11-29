# scenario_1_ml_simulation.py
# Ubicado en: project_root/src/scenario_1_ml_simulation.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ======================================================
# 1. RUTAS DEL PROYECTO
# ======================================================

ROOT = Path(__file__).resolve().parent.parent       # project root
DATA_DIR = ROOT / "data"
OUT = ROOT / "results" / "ml"
OUT.mkdir(parents=True, exist_ok=True)

# ======================================================
# 2. CARGAR DOOM DATA
# ======================================================

CSV = DATA_DIR / "psychopathy_DOOM_DATA.csv"
if not CSV.exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo DOOM DATA en: {CSV}")

df = pd.read_csv(CSV)

# target detectado automáticamente
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"

X_full = (
    df.drop(columns=[target])
    .select_dtypes(include=[np.number])
    .fillna(-1)
)
y_full = df[target]

# ======================================================
# 3. PARTE 1 — ESTABILIDAD DEL MODELO
# ======================================================

seeds = [1, 5, 10, 21, 42, 99]
mse_list = []

for s in seeds:
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=s)

    model = RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mse_list.append(mse)

    print(f"Seed {s}: MSE = {mse:.6f}")

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
# 4. PARTE 2 — DISTRIBUCIÓN REAL VS PREDICHA
# ======================================================

# Run canonical seed 42
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

plt.figure(figsize=(8, 4))
sns.histplot(y_test, label="True", kde=True, stat="density", alpha=0.5)
sns.histplot(preds, label="Pred", kde=True, stat="density", alpha=0.5)
plt.legend()
plt.title("Distribution: True vs Predicted (seed=42)")
plt.savefig(OUT / "hist_true_vs_pred.png", dpi=150)
plt.close()

# ======================================================
# 5. PARTE 3 — IMPORTANCIA DE VARIABLES
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
# 6. PARTE 4 — SENSIBILIDAD AL RUIDO
# ======================================================

noise_levels = [0.01, 0.03, 0.05, 0.1]
mse_pert = []

for nl in noise_levels:
    X_test_noisy = X_test.copy()

    for col in X_test_noisy.columns:
        sigma = X_test_noisy[col].std()
        noise = np.random.normal(0, nl * sigma, size=X_test_noisy.shape[0])
        X_test_noisy[col] += noise

    preds_noisy = model.predict(X_test_noisy)
    mse_n = mean_squared_error(y_test, preds_noisy)
    mse_pert.append(mse_n)

    print(f"Noise {nl*100:.1f}% -> MSE {mse_n:.6f}")

plt.figure(figsize=(7, 4))
plt.plot([l * 100 for l in noise_levels], mse_pert, marker='o')
plt.xlabel("Noise level (% of std)")
plt.ylabel("MSE on noisy test set")
plt.title("Sensitivity to Input Noise")
plt.savefig(OUT / "sensitivity_noise.png", dpi=150)
plt.close()

# ======================================================
# 7. GUARDAR RESÚMENES
# ======================================================

pd.DataFrame({"seed": seeds, "mse": mse_list}).to_csv(
    OUT / "ml_seed_results.csv", index=False
)

pd.DataFrame({
    "noise_level_frac": noise_levels,
    "mse": mse_pert
}).to_csv(OUT / "ml_noise_results.csv", index=False)

print("ML simulation completed. Results saved in:", OUT)
