# scenario_2_cellular_automata.py
# Ubicado en: project_root/src/scenario_2_cellular_automata.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label

# ======================================================
# 1. RUTAS DEL PROYECTO
# ======================================================

ROOT = Path(__file__).resolve().parent.parent        # project root
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "results"
OUT = ROOT / "results" / "ca"
OUT.mkdir(parents=True, exist_ok=True)

# ======================================================
# 2. PARÁMETROS DE LA SIMULACIÓN
# ======================================================

GRID_SIZE = 40              # tamaño del autómata
ITERS = 80                  # iteraciones totales
NOISE = 0.03                # magnitud del ruido ambiental
NEIGHBOR_WEIGHT = 0.6       # influencia de vecinos
THRESH_CLUSTER = 0.8        # umbral para cluster de “alto riesgo”

# ======================================================
# 3. CARGAR DOOM DATA Y MODELO
# ======================================================

CSV = DATA_DIR / "psychopathy_DOOM_DATA.csv"
if not CSV.exists():
    raise FileNotFoundError(f"❌ DOOM DATA no encontrada en: {CSV}")

df = pd.read_csv(CSV)
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"

X_full = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(-1)
y_full = df[target]

# ------------------------------------------------------
# Cargar modelo entrenado
# ------------------------------------------------------

MODEL_PATH = MODEL_DIR / "rf_final.pkl"

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
else:
    raise FileNotFoundError(
        f"❌ No se encontró el modelo final.\n"
        f"Primero ejecuta: train_and_save_model.py\n"
        f"Se esperaba el archivo en: {MODEL_PATH}"
    )

# ======================================================
# 4. INICIALIZACIÓN DEL AUTÓMATA CELULAR
# ======================================================

# Muestras aleatorias del dataset como "micro-individuos"
sampled = X_full.sample(GRID_SIZE * GRID_SIZE, replace=True).reset_index(drop=True)

# Predicción inicial del modelo
preds_init = model.predict(sampled)

# Crear grid inicial (mapa psicopático inicial)
grid = preds_init.reshape(GRID_SIZE, GRID_SIZE)

# Para guardar métricas
means = []
variances = []
num_clusters = []

# ======================================================
# 5. DEFINICIÓN DE UN PASO DEL AUTÓMATA
# ======================================================

def step(grid):
    new = grid.copy()

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            # recopilar vecinos
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                        neighbors.append(grid[ni, nj])

            local_mean = np.mean(neighbors) if neighbors else grid[i, j]

            # actualización: mezcla entre valor propio y vecinos
            new[i, j] = (
                (1 - NEIGHBOR_WEIGHT) * grid[i, j] +
                NEIGHBOR_WEIGHT * local_mean
            )

            # añadir ruido ambiental
            new[i, j] += np.random.uniform(-NOISE, NOISE)

            # asegurar rango válido (0 a 1)
            new[i, j] = max(0.0, min(1.0, new[i, j]))

    return new

# ======================================================
# 6. GUARDAR IMAGEN INICIAL
# ======================================================

plt.figure(figsize=(6, 5))
plt.imshow(grid, cmap="inferno", vmin=0, vmax=1)
plt.title("Iteration 0")
plt.colorbar()
plt.savefig(OUT / "ca_iter_000.png", dpi=150)
plt.close()

# ======================================================
# 7. SIMULACIÓN PRINCIPAL
# ======================================================

for t in range(1, ITERS + 1):

    grid = step(grid)

    means.append(grid.mean())
    variances.append(grid.var())

    mask = grid > THRESH_CLUSTER   # zonas de alto riesgo
    labeled, ncomp = label(mask)
    num_clusters.append(ncomp)

    if t % 10 == 0 or t == ITERS:
        plt.figure(figsize=(6, 5))
        plt.imshow(grid, cmap="inferno", vmin=0, vmax=1)
        plt.title(f"Iteration {t:03d}")
        plt.colorbar()
        plt.savefig(OUT / f"ca_iter_{t:03d}.png", dpi=150)
        plt.close()

# ======================================================
# 8. GRÁFICAS DE MÉTRICAS
# ======================================================

# media
plt.figure()
plt.plot(range(1, ITERS + 1), means)
plt.xlabel("Iteration")
plt.ylabel("Mean psychopathy")
plt.title("Mean Over Time")
plt.savefig(OUT / "ca_mean_over_time.png", dpi=150)
plt.close()

# varianza
plt.figure()
plt.plot(range(1, ITERS + 1), variances)
plt.xlabel("Iteration")
plt.ylabel("Variance")
plt.title("Variance Over Time")
plt.savefig(OUT / "ca_variance_over_time.png", dpi=150)
plt.close()

# clusters
plt.figure()
plt.plot(range(1, ITERS + 1), num_clusters)
plt.xlabel("Iteration")
plt.ylabel("Number of Clusters")
plt.title("High-Psychopathy Clusters Over Time")
plt.savefig(OUT / "ca_clusters_over_time.png", dpi=150)
plt.close()

print("Cellular automata simulation done. Images and metrics saved in:", OUT)
