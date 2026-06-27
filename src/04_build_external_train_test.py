import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "parkinsons.data"
RESULTS = BASE / "results"

TRAIN_SUBJ_PATH = RESULTS / "external_train_subjects.csv"
TEST_SUBJ_PATH = RESULTS / "external_test_subjects.csv"

# =========================
# Load full dataset
# =========================
df = pd.read_csv(DATA_PATH)

# Create subject_id (same rule as before)
# phon_R01_S01_1 -> phon_R01_S01
df["subject_id"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))

# =========================
# Load subject splits
# =========================
train_subjects = set(pd.read_csv(TRAIN_SUBJ_PATH)["subject_id"])
test_subjects = set(pd.read_csv(TEST_SUBJ_PATH)["subject_id"])

# =========================
# Build sample-level splits
# =========================
train_df = df[df["subject_id"].isin(train_subjects)].copy()
test_df = df[df["subject_id"].isin(test_subjects)].copy()

# =========================
# Sanity checks
# =========================
leakage = set(train_df["subject_id"]).intersection(set(test_df["subject_id"]))
print("Leakage subjects (must be empty):", leakage)

print("\nExternal TRAIN:", train_df.shape,
      "| subjects:", train_df["subject_id"].nunique())
print("Train class counts (samples):")
print(train_df["status"].value_counts())

print("\nExternal TEST:", test_df.shape,
      "| subjects:", test_df["subject_id"].nunique())
print("Test class counts (samples):")
print(test_df["status"].value_counts())

# =========================
# Save (very useful for later steps)
# =========================
train_df.to_csv(RESULTS / "external_train_samples.csv", index=False)
test_df.to_csv(RESULTS / "external_test_samples.csv", index=False)

print("\nSaved:")
print(" - results/external_train_samples.csv")
print(" - results/external_test_samples.csv")
