import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

# ============================================================
# CONFIG
# ============================================================
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ============================================================
# PATHS
# ============================================================
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data" / "parkinsons.data"
RESULTS_PATH = BASE_PATH / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

# ============================================================
# CREATE GROUP ID (subject_id)
# phon_R01_S01_1 -> phon_R01_S01
# ============================================================
df["subject_id"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))
print(df)
# ============================================================
# SUBJECT-LEVEL TABLE (for stratification)
# One row per subject with its label (status)
# ============================================================
subjects = df[["subject_id", "status"]].drop_duplicates()
print(subjects)
# Safety check: each subject must have exactly one label
n_labels_per_subject = subjects.groupby("subject_id")["status"].nunique()
if (n_labels_per_subject > 1).any():
    bad = n_labels_per_subject[n_labels_per_subject > 1]
    raise ValueError(
        f"Some subjects have multiple labels (unexpected):\n{bad}")

print("Total samples:", len(df))
print("Total subjects:", len(subjects))
print("\nSubject class counts (0=Healthy, 1=PD):")
print(subjects["status"].value_counts())

# ============================================================
# STRATIFIED SHUFFLE SPLIT AT SUBJECT LEVEL
# ============================================================
X_subj = subjects["subject_id"].to_numpy()
y_subj = subjects["status"].to_numpy()

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(sss.split(X_subj, y_subj))

train_subjects = set(X_subj[train_idx])
test_subjects = set(X_subj[test_idx])

# ============================================================
# MAP BACK TO SAMPLE LEVEL
# ============================================================
train_df = df[df["subject_id"].isin(train_subjects)].copy()
test_df = df[df["subject_id"].isin(test_subjects)].copy()

# ============================================================
# LEAKAGE CHECK (must be empty)
# ============================================================
leakage = train_subjects.intersection(test_subjects)
if leakage:
    raise RuntimeError(
        f"Subject leakage detected (should be empty): {leakage}")

print("\nSubject leakage check: OK (no overlap)")

# ============================================================
# REPORT (subjects + samples)
# ============================================================
train_subjects_df = subjects[subjects["subject_id"].isin(train_subjects)]
test_subjects_df = subjects[subjects["subject_id"].isin(test_subjects)]

print("\nTRAIN SET (subjects)")
print("  subjects:", train_subjects_df.shape[0])
print("  subject class counts:\n", train_subjects_df["status"].value_counts())

print("\nTRAIN SET (samples)")
print("  samples:", train_df.shape[0])
print("  sample class counts:\n", train_df["status"].value_counts())

print("\nTEST SET (subjects)")
print("  subjects:", test_subjects_df.shape[0])
print("  subject class counts:\n", test_subjects_df["status"].value_counts())

print("\nTEST SET (samples)")
print("  samples:", test_df.shape[0])
print("  sample class counts:\n", test_df["status"].value_counts())

# ============================================================
# SAVE SUBJECT SPLIT (reproducibility for thesis)
# ============================================================
train_subjects_df.to_csv(
    RESULTS_PATH / "external_train_subjects.csv", index=False)
test_subjects_df.to_csv(
    RESULTS_PATH / "external_test_subjects.csv", index=False)

print("\nSaved:")
print(" - results/external_train_subjects.csv")
print(" - results/external_test_subjects.csv")
