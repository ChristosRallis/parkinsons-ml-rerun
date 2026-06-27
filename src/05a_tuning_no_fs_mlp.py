# 05a_tuning_no_fs_mlp.py
# Hyperparameter tuning for MLP (NO Feature Selection)
# Uses RandomizedSearchCV with custom CV folds from JSON

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================

RANDOM_STATE = 42
N_JOBS = 1
SCORING = "f1"

# Number of random configurations to test
N_ITER = 120

# ==========================================
# PATHS
# ==========================================

BASE = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"

OUT_DIR = BASE / "results" / "tuning_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "05a_best_params_no_fs_mlp.json"
OUT_CSV = OUT_DIR / "05a_best_params_no_fs_mlp.csv"
OUT_FULL = OUT_DIR / "05a_all_results_no_fs_mlp.csv"

# ==========================================
# HELPERS
# ==========================================


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_custom_cv_from_folds(df: pd.DataFrame, folds_data: dict):
    custom_cv = []

    for fold_info in folds_data["folds"]:

        train_names = set(fold_info["train_samples"])
        val_names = set(fold_info["val_samples"])

        train_idx = df.index[df["name"].isin(train_names)].tolist()
        val_idx = df.index[df["name"].isin(val_names)].tolist()

        if set(train_idx).intersection(set(val_idx)):
            raise ValueError("Data leakage detected in custom CV!")

        custom_cv.append((train_idx, val_idx))

    return custom_cv


# ==========================================
# LOAD DATA
# ==========================================

print("📥 Loading data and folds for MLP tuning (No-FS)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()

folds_data = load_json(FOLDS_PATH)

custom_cv = build_custom_cv_from_folds(df, folds_data)

print(f"✅ Samples: {len(df)}")
print(f"✅ Features: {X.shape[1]}")
print(f"✅ Custom folds loaded: {len(custom_cv)}")
print(f"🎯 RandomizedSearch iterations: {N_ITER}")
print("-" * 60)

# ==========================================
# PIPELINE
# ==========================================

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE
    ))
])

# ==========================================
# SEARCH SPACE
# ==========================================

param_dist = {

    "mlp__hidden_layer_sizes": [
        (16,),
        (32,),
        (64,),
        (32, 16),
        (64, 32),
        (64, 32, 16),
        (128,),
        (128, 64)
    ],

    "mlp__activation": [
        "relu",
        "tanh"
    ],

    "mlp__alpha": np.logspace(-6, -1, 20),

    "mlp__learning_rate_init": np.logspace(-4, -2, 20),

    "mlp__solver": [
        "adam"
    ],

    "mlp__batch_size": [
        8,
        16,
        32,
        64
    ]
}

# ==========================================
# RANDOMIZED SEARCH
# ==========================================

search = RandomizedSearchCV(

    estimator=pipe,

    param_distributions=param_dist,

    n_iter=N_ITER,

    scoring=SCORING,

    cv=custom_cv,

    verbose=2,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    refit=True
)

print("🚀 Starting MLP tuning...")

search.fit(X, y)

print("✅ Tuning complete.")
print("-" * 60)

# ==========================================
# RESULTS
# ==========================================

best_params = search.best_params_
best_score = search.best_score_

print("🏆 BEST PARAMS:")
print(best_params)

print(f"\n🏆 BEST CV F1: {best_score:.6f}")

# Save JSON

out_json_data = {

    "Model": "MLP",

    "best_params": best_params,

    "best_cv_f1": float(best_score),

    "n_iter": N_ITER
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_json_data, f, indent=4)

# Save summary CSV

summary_df = pd.DataFrame([{
    "Model": "MLP",
    "Best_CV_F1": best_score,
    **best_params
}])

summary_df.to_csv(OUT_CSV, index=False)

# Save full results

full_df = pd.DataFrame(search.cv_results_)
full_df.to_csv(OUT_FULL, index=False)

print(f"\n✅ Saved best params to: {OUT_JSON}")
print(f"✅ Saved summary CSV to: {OUT_CSV}")
print(f"✅ Saved full results to: {OUT_FULL}")

print("\n🏁 MLP tuning finished successfully.")
