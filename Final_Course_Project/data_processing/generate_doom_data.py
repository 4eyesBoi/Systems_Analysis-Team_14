#!/usr/bin/env python3
"""
Generador de DOOM Data sintética usando SMOGN (versión robusta para proyecto)

Salida por defecto:
 - Preprocesado: project_root/data/<target>_preprocessed.csv
 - SMOGN:        project_root/data/<target>_SMOGN.csv
 - DOOM DATA:    project_root/data/<target>_DOOM_DATA.csv

Coloca este archivo en: project_root/data_processing/generate_doom_data.py
Ejecútalo desde la raíz del proyecto o desde data_processing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import smogn
except Exception:  # pragma: no cover - runtime dependency
    smogn = None

from sklearn.preprocessing import MinMaxScaler


# --------------------------
# Helpers
# --------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    logging.info("Dataset cargado. Shape: %s", df.shape)
    return df


def convert_and_clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Convirtiendo columnas a numéricas ...")
    df_num = df.copy()
    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

    # quitar columnas totalmente vacías
    df_num = df_num.dropna(axis=1, how="all")

    # rellenar NA con la media (requerido por SMOGN)
    # .mean() sobre columnas no-numéricas ya fue forzada a NaN y eliminada arriba
    df_num = df_num.fillna(df_num.mean())

    # si alguna columna sigue con NaN (por ejemplo si toda la columna era NaN), eliminarla
    df_num = df_num.dropna(axis=1, how="any")

    logging.info("Shape después de limpieza: %s", df_num.shape)
    return df_num


def normalize_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no existe en dataset.")
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[target + "_norm"] = scaler.fit_transform(df[[target]])
    logging.info("Target '%s' normalizado en rango 0..1.", target)
    return df_norm, scaler


def apply_smogn(df: pd.DataFrame, target_norm: str, out_dir: Path, scaler_target):
    """
    Aplica dos variantes de SMOGN y guarda resultados en out_dir:
      - <target>_SMOGN.csv
      - <target>_DOOM_DATA.csv
    """
    if smogn is None:
        raise RuntimeError("El paquete 'smogn' no está disponible. Instale con `pip install smogn`.")

    if target_norm not in df.columns:
        raise ValueError(f"ERROR: No existe la columna '{target_norm}' en el dataset.")

    target_base = target_norm[:-5] if target_norm.endswith("_norm") else target_norm

    t = df[target_norm]
    p50 = t.quantile(0.50)
    p90 = t.quantile(0.90)
    p97 = t.quantile(0.97)
    logging.info("Percentiles %s: P50=%s P90=%s P97=%s", target_norm, p50, p90, p97)

    logging.info("Aplicando SMOGN (auto)...")
    df_smogn = smogn.smoter(
        data=df,
        y=target_norm,
        rel_method="auto",
        rel_thres=0.4,
        samp_method="extreme",
        k=15,
    )
    logging.info("SMOGN (auto) completado. Shape: %s", df_smogn.shape)

    logging.info("Aplicando SMOGN DOOM (manual control points)...")
    rel_points = [[0.00, 0.0, 0], [p50, 0.0, 0], [p90, 0.5, 0], [p97, 1.0, 0]]
    logging.info("Control points: %s", rel_points)

    try:
        df_doom = smogn.smoter(
            data=df,
            y=target_norm,
            rel_method="manual",
            rel_ctrl_pts_rg=rel_points,
            rel_thres=0.2,
            k=30,
            samp_method="balance",
        )
    except TypeError:
        # fallback para versiones con distinto nombre de parámetro
        df_doom = smogn.smoter(
            data=df,
            y=target_norm,
            rel_method="manual",
            rel_ctrl_pts=rel_points,
            rel_thres=0.2,
            k=30,
            samp_method="balance",
        )

    logging.info("SMOGN DOOM completado. Shape: %s", df_doom.shape)

    # Invertir solo la columna target (siempre intentar, pero en fallo dejar normalizado)
    def inverse_safe(df_synth):
        df_ret = df_synth.copy()
        try:
            inv = scaler_target.inverse_transform(df_synth[[target_norm]])
            df_ret[f"{target_base}_synth"] = inv
        except Exception:
            logging.warning("No se pudo invertir la normalización del target; dejando normalizado.")
        return df_ret

    df_smogn = inverse_safe(df_smogn)
    df_doom = inverse_safe(df_doom)

    out1 = out_dir / f"{target_base}_SMOGN.csv"
    out2 = out_dir / f"{target_base}_DOOM_DATA.csv"

    df_smogn.to_csv(out1, index=False)
    df_doom.to_csv(out2, index=False)

    logging.info("Guardado SMOGN -> %s", out1)
    logging.info("Guardado DOOM  -> %s", out2)


# --------------------------
# MAIN
# --------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Generar DOOM data sintética usando SMOGN")
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="CSV input file (default: first CSV found in project_root/data)")
    parser.add_argument("--target", "-t", default="psychopathy", help="Target column name (default: psychopathy)")
    parser.add_argument("--out-dir", "-o", type=Path, default=None,
                        help="Output directory for generated CSVs (default: project_root/data)")
    parser.add_argument("--skip-normalize", action="store_true", help="Skip normalizing the target")
    parser.add_argument("--no-smogn", action="store_true", help="Do preprocessing only; do not run SMOGN")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")

    args = parser.parse_args()

    # determine project root relative to this script: assume script is in project_root/data_processing
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    # default data folder (project_root/data)
    default_data_folder = project_root / "data/raw"
    default_data_folder.mkdir(parents=True, exist_ok=True)

    # decide input CSV
    if args.input is None:
        csvs = sorted(default_data_folder.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {default_data_folder}. Provide --input to specify the file.")
        input_path = csvs[0]
        logging.info("Autodetectado input CSV: %s", input_path)
    else:
        input_path = args.input

    # decide out dir
    if args.out_dir is None:
        out_dir = default_data_folder
    else:
        out_dir = args.out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    logging.info("Usando archivo: %s", input_path)
    logging.info("Salida a carpeta: %s", out_dir)

    # Load and preprocess
    df = load_dataset(input_path)
    df_num = convert_and_clean_numeric(df)

    # Optionally normalize target
    if args.skip_normalize:
        scaler = None
        df_norm = df_num
    else:
        df_norm, scaler = normalize_target(df_num, args.target)

    # Save preprocessed (useful artifact)
    preproc_path = out_dir / f"{args.target}_preprocessed.csv"
    df_norm.to_csv(preproc_path, index=False)
    logging.info("Preprocesado guardado en: %s", preproc_path)

    if args.no_smogn:
        logging.info("--no-smogn especificado. Solo preprocesado, saliendo.")
        return

    # Run SMOGN variants (requires smogn)
    apply_smogn(df_norm, target_norm=args.target + "_norm" if not args.skip_normalize else args.target, out_dir=out_dir, scaler_target=scaler)


if __name__ == "__main__":
    main()
