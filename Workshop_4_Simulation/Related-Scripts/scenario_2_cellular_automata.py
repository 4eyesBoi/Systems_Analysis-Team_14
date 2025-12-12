# scenario_2_cellular_automata.py
# Ubicado en: project_root/src/scenario_2_cellular_automata.py

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import label, convolve
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

# ======================================================
# 1. PATH SETUP
# ======================================================

ROOT = Path(__file__).resolve().parent.parent        # project root
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "results"
OUT = ROOT / "results" / "ca"
OUT.mkdir(parents=True, exist_ok=True)

# ======================================================
# 2. SIMULATION PARAMETERS
# ======================================================

GRID_SIZE = 40              # cellular automaton size
ITERS = 80                  # total iterations
NOISE = 0.03                # environmental noise magnitude
NEIGHBOR_WEIGHT = 0.6       # neighbor influence
THRESH_CLUSTER = 0.8        # threshold for "high risk" cluster
USE_CONVOLUTION = True      # use 2D convolution for optimization
logger.info(f"CA Parameters: GRID_SIZE={GRID_SIZE}, ITERS={ITERS}, NOISE={NOISE}, NEIGHBOR_WEIGHT={NEIGHBOR_WEIGHT}")

# ======================================================
# 3. LOAD DOOM DATA AND MODEL
# ======================================================

CSV = DATA_DIR / "processed" / "psychopathy_DOOM_DATA.csv"
if not CSV.exists():
    logger.error(f"DOOM DATA not found in: {CSV}")
    raise FileNotFoundError(f"DOOM DATA not found in: {CSV}")

logger.info(f"Loading data from: {CSV}")
df = pd.read_csv(CSV)
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"
logger.info(f"Target column: {target}")

X_full = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(-1)
y_full = df[target]

# Validate dataset size
required_samples = GRID_SIZE * GRID_SIZE
if len(X_full) < required_samples:
    logger.warning(f"Dataset has {len(X_full)} samples, needs {required_samples}. Using replace=True.")

logger.info(f"Dataset shape: {X_full.shape}")

# ------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------

MODEL_PATH = MODEL_DIR / "rf_final.pkl"

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from: {MODEL_PATH}")
else:
    logger.error(f"Model not found in: {MODEL_PATH}")
    raise FileNotFoundError(
        f"Final model not found.\n"
        f"First run: train_and_save_model.py\n"
        f"Expected file: {MODEL_PATH}"
    )

# ======================================================
# 4. INITIALIZE CELULAR AUTOMATA GRID
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
# 5. DEFINICIÓN DE PASOS DEL AUTÓMATA (DOS VERSIONES)
# ======================================================

def step_convolution(grid):
    """
    Versión optimizada usando convolución 2D (O(n log n))
    Mucho más rápido para grillas grandes.
    """
    # Kernel para promedio de vecinos (9 celdas: 8 vecinos + centro)
    kernel = np.ones((3, 3)) / 9.0
    
    # Aplicar convolución con padding (boundary='constant' asume 0 afuera)
    local_means = convolve(grid, kernel, mode='constant', cval=0)
    
    # Actualización: mezcla entre valor propio y vecinos
    new = (1 - NEIGHBOR_WEIGHT) * grid + NEIGHBOR_WEIGHT * local_means
    
    # Añadir ruido ambiental
    new += np.random.uniform(-NOISE, NOISE, size=grid.shape)
    
    # Asegurar rango válido [0, 1]
    new = np.clip(new, 0.0, 1.0)
    
    return new

def step_loop(grid):
    """
    Versión original usando bucles anidados (O(n²))
    Más lenta pero más explícita.
    """
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

# Seleccionar versión según configuración
step_func = step_convolution if USE_CONVOLUTION else step_loop
logger.info(f"Usando versión: {'convolución 2D (optimizada)' if USE_CONVOLUTION else 'bucles (original)'}")

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

    grid = step_func(grid)

    means.append(grid.mean())
    variances.append(grid.var())

    mask = grid > THRESH_CLUSTER   # zonas de alto riesgo
    labeled, ncomp = label(mask)
    num_clusters.append(ncomp)

    if t % 10 == 0 or t == ITERS:
        logger.info(f"Iteration {t}: mean={grid.mean():.4f}, var={grid.var():.4f}, clusters={ncomp}")
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

logger.info("Cellular automata simulation completed.")
logger.info(f"Final statistics - Mean: {means[-1]:.4f}, Variance: {variances[-1]:.4f}, Clusters: {num_clusters[-1]}")
logger.info(f"Results saved in: {OUT}")
