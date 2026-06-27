import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "parkinsons.data"

# =========================
# Column names (από parkinsons.names)
# =========================
columns = [
    "name",
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "status",
    "RPDE", "DFA", "spread1", "spread2", "PPE"
]

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH)

print("RAW shape:", df.shape)

# =========================
# Basic checks
# =========================
print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution (status):")
print(df["status"].value_counts())

# =========================
# Subject grouping (CORRECT)
# phon_R01_S01_1 -> phon_R01_S01
# =========================
df["subject_id"] = df["name"].apply(lambda x: "_".join(x.split("_")[:3]))

print("\nNumber of subjects:", df["subject_id"].nunique())
print("\nSamples per subject (first 10):")
print(df["subject_id"].value_counts().head(10))
