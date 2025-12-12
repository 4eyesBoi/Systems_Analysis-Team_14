import pandas as pd
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "results" / "rf_final.pkl"
DATA = ROOT / "data" / "processed" / "psychopathy_DOOM_DATA.csv"

df = pd.read_csv(DATA)
target = "psychopathy_synth" if "psychopathy_synth" in df.columns else "psychopathy"

# Generate an ID column if not provided
if "uid" not in df.columns:
    df["uid"] = df.index + 1

X = df.drop(columns=[target]).select_dtypes(include=["float", "int"]).fillna(-1)

model = joblib.load(MODEL)
preds = model.predict(X)

submission = pd.DataFrame({
    "myID": df["uid"],
    "psychopathy": preds
})

out_file = ROOT / "results" / "submission_kaggle.csv"
submission.to_csv(out_file, index=False)
print("Submission generated ->", out_file)
