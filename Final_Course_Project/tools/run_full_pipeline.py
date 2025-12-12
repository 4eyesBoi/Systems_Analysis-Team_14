#!/usr/bin/env python3
"""
Helper to run the full pipeline in order.
Executes:
 1) generate DOOM data (optional: skip if already created)
 2) train model
 3) evaluate tail metrics
 4) run stratified CV
 5) run baseline comparisons
 6) hyperparameter search (optional, set flag)
 7) bootstrap uncertainty (optional, set flag)
 8) scenario simulations
 9) compare scenarios  
10) build submission
"""
import subprocess
from pathlib import Path
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_full")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_PROCESS = ROOT / "data_processing"


def run(cmd):
    logger.info("Running: %s", " ".join(cmd))
    res = subprocess.run(cmd, shell=False)
    if res.returncode != 0:
        logger.error("Command failed: %s", cmd)
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generate", action="store_true", help="Skip data generation")
    parser.add_argument("--run-hparam", action="store_true", help="Run hyperparameter search (long)")
    parser.add_argument("--run-bootstrap", action="store_true", help="Run bootstrap uncertainty (long)")
    args = parser.parse_args()

    # 1) generate doom data
    if not args.skip_generate:
        run([
            sys.executable,
            str(DATA_PROCESS / "generate_doom_data.py"),
            "--input", "data/raw/PersonalityData_ExternalVersion001.csv",
            "--out-dir", "data/processed"
        ])
    else:
        logger.info("Skipping generate_doom_data.py")

    # 2) train final model
    run([sys.executable, str(SRC / "train_and_save_model.py")])

    # 3) tail metrics
    run([sys.executable, str(SRC / "evaluation" / "evaluate_tail_metrics.py")])

    # 4) stratified CV
    run([sys.executable, str(SRC / "evaluation" / "evaluate_cv_stratified.py")])

    # 5) baselines
    run([sys.executable, str(SRC / "evaluation" / "evaluate_baselines.py")])

    # 6) hyperparameter search (optional)
    if args.run_hparam:
        run([sys.executable, str(SRC / "evaluation" / "hparam_search.py")])

    # 7) bootstrap uncertainty (optional)
    if args.run_bootstrap:
        run([sys.executable, str(SRC / "evaluation" / "bootstrap_uncertainty.py")])

    # 8) simulations
    run([sys.executable, str(SRC / "scenario_1_ml_simulation.py")])
    run([sys.executable, str(SRC / "scenario_2_cellular_automata.py")])

    # 9) compare scenarios  ⭐ NEW ⭐
    run([sys.executable, str(SRC / "analysis" / "compare_scenarios.py")])

    # 10) submission
    run([sys.executable, str(SRC / "submission" / "build_submission.py")])

    logger.info("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
