import json
import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
BASE = Path(__file__).resolve().parents[1]
TRAIN_SAMPLES_PATH = BASE / "results" / "external_train_samples.csv"
SUBJECT_FOLDS_PATH = BASE / "results" / "cv_folds_subjects.json"
OUT_PATH = BASE / "results" / "cv_folds_samples.json"

# =========================
# Load data
# =========================
df = pd.read_csv(TRAIN_SAMPLES_PATH)

with open(SUBJECT_FOLDS_PATH, "r", encoding="utf-8") as f:
    subject_folds = json.load(f)

# =========================
# Build sample-level folds
# =========================
sample_folds = {
    "n_splits": subject_folds["n_splits"],
    "random_state": subject_folds["random_state"],
    "folds": []
}

for fold_info in subject_folds["folds"]:
    fold_id = fold_info["fold"]

    train_subjects = set(fold_info["train_subjects"])
    val_subjects = set(fold_info["val_subjects"])

    train_samples = df[df["subject_id"].isin(train_subjects)]["name"].tolist()
    val_samples = df[df["subject_id"].isin(val_subjects)]["name"].tolist()

    sample_folds["folds"].append({
        "fold": fold_id,
        "train_samples": train_samples,
        "val_samples": val_samples
    })

# =========================
# Save
# =========================
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sample_folds, f, ensure_ascii=False, indent=2)

print("Saved sample-level CV folds to:", OUT_PATH)

# =========================
# Sanity check (optional)
# =========================
for fold in sample_folds["folds"]:
    overlap = set(fold["train_samples"]).intersection(set(fold["val_samples"]))
    if overlap:
        raise RuntimeError(f"Sample leakage in fold {fold['fold']}: {overlap}")

print("Leakage check passed: no sample appears in both train and validation.")
