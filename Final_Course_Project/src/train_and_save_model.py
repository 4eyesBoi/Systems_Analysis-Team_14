# train_and_save_model.py
# Ubicado en: project_root/src/train_and_save_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path
import os

# ============================================
# 1. Rutas del proyecto
# ============================================

ROOT = Path(__file__).resolve().parent.parent   # project_root
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivo DOOM data (tú ya lo generaste con generate_doom_data.py)
CSV = DATA_DIR / "psychopathy_DOOM_DATA.csv"

# ============================================
# 2. Cargar dataset balanceado
# ============================================

if not CSV.exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo DOOM DATA en: {CSV}")

df = pd.read_csv(CSV)

# target generado por tu pipeline
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"

# ============================================
# 3. Features y target
# ============================================

X = df.drop(columns=[target])
y = df[target]

# Asegurar solo numéricas
X = X.select_dtypes(include=[np.number]).fillna(-1)

# ============================================
# 4. División en train/test
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ============================================
# 5. Entrenar modelo
# ============================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("Training final model...")
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Final model MSE (test): {mse:.6f}")

# ============================================
# 6. Guardar modelo y datasets de simulación
# ============================================

joblib.dump(model, RESULTS_DIR / "rf_final.pkl")

X_test.to_csv(RESULTS_DIR / "X_test_for_simulation.csv", index=False)
y_test.reset_index(drop=True).to_csv(RESULTS_DIR / "y_test_for_simulation.csv", index=False)

print("Saved model ->", RESULTS_DIR / "rf_final.pkl")
