# Final Course Project — Psychopathy Prediction based on Twitter Usage

## Summary
This repository contains a complete pipeline to preprocess data, generate synthetic DOOM data (SMOGN), train RandomForest models, evaluate tail performance and run two simulation scenarios (ML stability and Cellular Automata).

## Project structure
- `data/raw/` — original CSV
- `data/processed/` — preprocessed, SMOGN and DOOM data
- `data_processing/` — generate_doom_data.py
- `src/` — main scripts (train, simulation, evaluation)
- `src/evaluation/` — evaluation scripts (tail, baselines, stratified CV, hparam, bootstrap)
- `results/` — models, images, CSVs
- `report/` — LaTeX report (main.tex)
- `tools/` — helper run scripts
- `test/` — aditional smoke test

## Quickstart (from project root)
```bash
# create virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt

# generate doom data (only if missing)
python data_processing/generate_doom_data.py --input data/raw/PersonalityData_ExternalVersion001.csv --out-dir data/processed

# train model
python src/train_and_save_model.py

# evaluate tail and baselines
python src/evaluation/evaluate_tail_metrics.py
python src/evaluation/evaluate_cv_stratified.py
python src/evaluation/evaluate_baselines.py

# run simulations
python src/scenario_1_ml_simulation.py
python src/scenario_2_cellular_automata.py

# optional
python tools/run_full_pipeline.py --run-hparam --run-bootstrap
