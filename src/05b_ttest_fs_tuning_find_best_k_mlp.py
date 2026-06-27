# 05b_ttest_fs_tuning_find_best_k_mlp.py
# ------------------------------------------------------------
# Tuning only fs__k for T-test Feature Selection + MLP
# - MLP hyperparams are FIXED from: results/tuning_mlp/05a_best_params_no_fs_mlp.json
# - custom folds from: results/cv_folds_samples.json
# - scoring: F1
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================
RANDOM_STATE = 42
N_JOBS = 1
SCORING = "f1"

K_LIST = [5, 8, 10, 12, 15, 18, 22]

# ==========================================
# PATHS
# ==========================================
BASE = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
BEST_MLP_PARAMS_PATH = BASE / "results" / \
    "tuning_mlp" / "05a_best_params_no_fs_mlp.json"

OUT_DIR = BASE / "results" / "evaluation_t-test_fs_mlp" / "t-test_tuning_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "best_k_mlp.json"
OUT_CSV = OUT_DIR / "best_k_mlp.csv"
OUT_CV_RESULTS = OUT_DIR / "cv_results_mlp.csv"

# ==========================================
# HELPERS
# ==========================================


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_custom_cv_from_folds(df: pd.DataFrame, folds_data: dict):
    custom_cv = []
    for i, fold_info in enumerate(folds_data["folds"], start=1):
        train_names = set(fold_info["train_samples"])
        val_names = set(fold_info["val_samples"])

        train_idx = df.index[df["name"].isin(train_names)].tolist()
        val_idx = df.index[df["name"].isin(val_names)].tolist()

        if len(train_idx) != len(train_names):
            raise ValueError(f"Fold {i}: Missing train samples in df.")
        if len(val_idx) != len(val_names):
            raise ValueError(f"Fold {i}: Missing val samples in df.")
        if set(train_idx).intersection(set(val_idx)):
            raise ValueError(f"Fold {i}: Leakage detected (overlap).")

        custom_cv.append((train_idx, val_idx))
    return custom_cv


def build_pipeline(best_mlp_params: dict):
    # Pipeline: scaler -> SelectKBest -> MLP
    fs = SelectKBest(score_func=f_classif)

    clf = MLPClassifier(
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("fs", fs),
        ("mlp", clf)
    ])

    # apply tuned MLP params (keys: mlp__...)
    pipe.set_params(**best_mlp_params)

    return pipe


# ==========================================
# LOAD DATA
# ==========================================
print("📥 Loading data/folds/best-params (T-test FS tuning MLP)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()

folds_data = load_json(FOLDS_PATH)
best_mlp_data = load_json(BEST_MLP_PARAMS_PATH)

best_mlp_params = best_mlp_data.get("best_params", None)
if best_mlp_params is None:
    raise ValueError(f"Could not find 'best_params' in {BEST_MLP_PARAMS_PATH}")

custom_cv = build_custom_cv_from_folds(df, folds_data)

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features")
print(f"✅ Using {len(custom_cv)} custom folds from: {FOLDS_PATH.name}")
print(f"✅ Fixed MLP hyperparams loaded from: {BEST_MLP_PARAMS_PATH.name}")
print(f"🎯 Tuning only fs__k over: {K_LIST} | scoring={SCORING}")
print("-" * 70)

# ==========================================
# GRID SEARCH (ONLY k)
# ==========================================
pipe = build_pipeline(best_mlp_params)

param_grid = {"fs__k": K_LIST}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=SCORING,
    cv=custom_cv,
    n_jobs=N_JOBS,
    refit=True,
    return_train_score=False,
    verbose=0
)

grid.fit(X, y)

best_k = int(grid.best_params_["fs__k"])
best_score = float(grid.best_score_)

print(f"✅ best_k={best_k} | best_mean_cv_f1={best_score:.6f}")
print("-" * 70)

# ==========================================
# SAVE OUTPUTS
# ==========================================
summary_df = pd.DataFrame([{
    "Model": "MLP",
    "Best_k": best_k,
    "Best_mean_CV_F1": best_score,
    "K_candidates": str(K_LIST)
}])

res = pd.DataFrame(grid.cv_results_)
keep_cols = ["param_fs__k", "mean_test_score",
             "std_test_score", "rank_test_score"]
res = res[keep_cols].copy()
res.rename(columns={
    "param_fs__k": "k",
    "mean_test_score": "mean_f1",
    "std_test_score": "std_f1",
    "rank_test_score": "rank"
}, inplace=True)

out_json_data = {
    "scoring": SCORING,
    "k_list": K_LIST,
    "best_k_mlp": {
        "best_k": best_k,
        "best_mean_cv_f1": best_score
    }
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_json_data, f, indent=4)

summary_df.to_csv(OUT_CSV, index=False)
res.to_csv(OUT_CV_RESULTS, index=False)

print(f"✅ Saved: {OUT_JSON}")
print(f"✅ Saved: {OUT_CSV}")
print(f"✅ Saved: {OUT_CV_RESULTS}")

print("\n🏁 Done.")
print(summary_df)
