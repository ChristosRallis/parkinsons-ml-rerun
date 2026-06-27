import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings

# Σιγάζουμε τα warnings για να βλέπουμε καθαρά τα logs
warnings.filterwarnings("ignore")

# ==========================================
# 1. Setup Paths & Config tuning _nofs_pipeline
# ==========================================
BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
OUT_DIR = BASE / "results" / "tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = -1

# ==========================================
# 2. Load Data & Folds
# ==========================================
print("Loading data...")
df = pd.read_csv(TRAIN_PATH)

# !!! SOS ΑΣΦΑΛΕΙΑΣ !!!
# Κάνουμε reset index για να είμαστε 100% σίγουροι ότι τα indices είναι 0, 1, 2...
# Έτσι η αντιστοίχιση με το JSON θα είναι απόλυτα ακριβής.
df = df.reset_index(drop=True)

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()

print(f"Data Loaded. Shape: {X.shape}")

# Load JSON Folds
with open(FOLDS_PATH, "r", encoding="utf-8") as f:
    folds_data = json.load(f)

# ==========================================
# 3. Create Custom CV Iterator from JSON
# ==========================================
# Στόχος: Να μετατρέψουμε τα ονόματα αρχείων (rec1.wav) σε αριθμούς γραμμών (row 0)
custom_cv = []

print("\nVerifying Folds Integrity...")

for i, fold_info in enumerate(folds_data["folds"]):
    train_names = set(fold_info["train_samples"])
    val_names = set(fold_info["val_samples"])

    # Βρίσκουμε σε ποιες γραμμές του DataFrame βρίσκονται αυτά τα ονόματα
    # Χρησιμοποιούμε το index του df που μόλις κάναμε reset.
    train_idx = df.index[df["name"].isin(train_names)].tolist()
    val_idx = df.index[df["name"].isin(val_names)].tolist()

    # --- CHECK 1: Βρήκαμε όλα τα δείγματα; ---
    if len(train_idx) != len(train_names):
        missing = len(train_names) - len(train_idx)
        raise ValueError(
            f"CRITICAL ERROR in Fold {i+1}: Could not find {missing} training samples in DataFrame!")

    if len(val_idx) != len(val_names):
        missing = len(val_names) - len(val_idx)
        raise ValueError(
            f"CRITICAL ERROR in Fold {i+1}: Could not find {missing} validation samples in DataFrame!")

    # --- CHECK 2: Υπάρχει διαρροή (Overlap); ---
    if set(train_idx).intersection(set(val_idx)):
        raise ValueError(
            f"CRITICAL ERROR in Fold {i+1}: Data Leakage detected! Overlap between Train/Val indices.")

    custom_cv.append((train_idx, val_idx))
    print(f"  Fold {i+1}: OK (Train={len(train_idx)}, Val={len(val_idx)})")

print("All folds verified correctly. Proceeding to grid search.")

# ==========================================
# 4. Define Hyperparameter Grids
# ==========================================
param_grids = {
    "LogisticRegression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000,
             solver="liblinear", random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l1", "l2"]
        }
    },

    "LinearSVM": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10, 100]
        }
    },

    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"]
        }
    },

    "XGBoost": {
        "model": XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        ),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0]
        }
    }
}

# ==========================================
# 5. Run Grid Search
# ==========================================
best_params_results = {}

print("\nStarting Hyperparameter Tuning (No-FS)...")
print("=" * 60)

for name, config in param_grids.items():
    print(f"🔹 Tuning {name}...")

    # Χρησιμοποιούμε το F1 score γιατί είναι πιο ασφαλές σε imbalanced data από το accuracy
    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=custom_cv,        # <--- ΤΟ ΚΛΕΙΔΙ: Χρήση των δικών μας Folds
        scoring="f1",
        n_jobs=N_JOBS,
        verbose=1
    )

    grid.fit(X, y)

    print(f"    Best F1: {grid.best_score_:.4f}")
    print(f"    Best Params: {grid.best_params_}")

    best_params_results[name] = {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_
    }

# ==========================================
# 6. Save Best Params to JSON
# ==========================================
out_file = OUT_DIR / "05a_best_params_no_fs.json"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(best_params_results, f, indent=4)

print("=" * 60)
print(f" Tuning Complete! Best parameters saved to:\n{out_file}")
