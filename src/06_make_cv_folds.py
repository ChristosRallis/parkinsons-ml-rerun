import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# =========================
# Config
# =========================
N_SPLITS = 3
RANDOM_STATE = 42

# =========================
# Paths
# =========================
BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
OUT_PATH = BASE / "results" / "cv_folds_subjects.json"

# =========================
# Load external train
# =========================
df = pd.read_csv(TRAIN_PATH)

# We only need subject-level table for folds
# One row per subject with its label (status)
subjects = df[["subject_id", "status"]].drop_duplicates().copy()

X_subj = subjects["subject_id"].to_numpy()
y_subj = subjects["status"].astype(int).to_numpy()
groups_subj = subjects["subject_id"].to_numpy()  # group = itself (subject)

print("Total subjects:", len(subjects))
print("Subject class counts:\n", subjects["status"].value_counts(), "\n")

# =========================
# Make folds (subject-wise)
# =========================
cv = StratifiedGroupKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

folds = []
fold_id = 0

for train_idx, val_idx in cv.split(X_subj, y_subj, groups=groups_subj):
    fold_id += 1

    train_subjects = X_subj[train_idx].tolist()
    val_subjects = X_subj[val_idx].tolist()

    folds.append({
        "fold": fold_id,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects
    })

# =========================
# Save folds
# =========================
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {"n_splits": N_SPLITS, "random_state": RANDOM_STATE, "folds": folds},
        f,
        ensure_ascii=False,
        indent=2
    )

print("Saved folds to:", OUT_PATH)

# =========================
# Report each fold composition
# =========================
for fold in folds:
    val_subj = fold["val_subjects"]
    val_df = subjects[subjects["subject_id"].isin(val_subj)]
    print(f"\nFOLD {fold['fold']} (validation subjects: {len(val_subj)})")
    print("  PD subjects:", int((val_df["status"] == 1).sum()))
    print("  Healthy subjects:", int((val_df["status"] == 0).sum()))
