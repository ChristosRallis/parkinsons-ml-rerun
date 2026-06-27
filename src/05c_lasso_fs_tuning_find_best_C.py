# 05c_lasso_fs_tuning_find_best_C.py
# ------------------------------------------------------------
# LASSO feature selection tuning (analogous to your t-test k tuning):
# - Uses external_train_samples.csv
# - Uses custom folds from cv_folds_samples.json
# - Loads best classifier hyperparams from: results/tuning/05a_best_params_no_fs.json
# - Tunes only: lasso__estimator__C  (L1 LogisticRegression inside SelectFromModel)
#
# Saves:
# - best C per model: results/evaluation_lasso_fs/lasso_tuning/best_C_per_model.json
# - csv summary:      results/evaluation_lasso_fs/lasso_tuning/best_C_per_model.csv
# - cv results:       results/evaluation_lasso_fs/lasso_tuning/cv_results_per_model.csv
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIG
# ==========================================
RANDOM_STATE = 42
N_JOBS = -1
SCORING = "f1"

# Candidate C values for LASSO selector (log-scale is typical)
C_LIST = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]

# ==========================================
# 2. PATHS
# ==========================================
BASE = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"

OUT_DIR = BASE / "results" / "evaluation_lasso_fs" / "lasso_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "best_C_per_model.json"
OUT_CSV = OUT_DIR / "best_C_per_model.csv"
OUT_CV_RESULTS = OUT_DIR / "cv_results_per_model.csv"

# ==========================================
# 3. HELPERS
# ==========================================


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_none_strings(d: dict) -> dict:
    """Convert 'None' strings to None."""
    out = {}
    for k, v in d.items():
        out[k] = None if v == "None" else v
    return out


def to_pipeline_clf_params(best_params: dict) -> dict:
    """
    Ensure hyperparams target the classifier step inside our Pipeline.
    If params already have 'clf__', keep them.
    Otherwise, prefix with 'clf__' (useful for RF/XGB params stored without prefix).
    """
    best_params = fix_none_strings(best_params)
    out = {}
    for k, v in best_params.items():
        if k.startswith("clf__"):
            out[k] = v
        else:
            out[f"clf__{k}"] = v
    return out


def build_model_pipeline(model_name: str):
    """
    Pipeline: scaler -> LASSO selector -> classifier
    LASSO selector is SelectFromModel(LogisticRegression L1).
    """

    # LASSO selector estimator (IMPORTANT: L1 + solver that supports it)
    lasso_est = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
        # optional: class_weight="balanced"  (enable if you want bias towards recall)
        # class_weight="balanced",
    )

    fs = SelectFromModel(
        estimator=lasso_est,
        # keep fixed for stability (analogous to not tuning extra knobs)
        threshold="mean"
    )

    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)
    elif model_name == "LinearSVM":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_name == "XGBoost":
        clf = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", fs),
        ("clf", clf),
    ])
    return pipe


def build_custom_cv_from_folds(df: pd.DataFrame, folds_data: dict):
    """
    Convert fold sample names -> row indices, with sanity checks.
    Returns list of (train_idx, val_idx).
    """
    custom_cv = []
    for i, fold_info in enumerate(folds_data["folds"], start=1):
        train_names = set(fold_info["train_samples"])
        val_names = set(fold_info["val_samples"])

        train_idx = df.index[df["name"].isin(train_names)].tolist()
        val_idx = df.index[df["name"].isin(val_names)].tolist()

        if len(train_idx) != len(train_names):
            raise ValueError(
                f"Fold {i}: Missing {len(train_names)-len(train_idx)} train samples in df.")
        if len(val_idx) != len(val_names):
            raise ValueError(
                f"Fold {i}: Missing {len(val_names)-len(val_idx)} val samples in df.")

        if set(train_idx).intersection(set(val_idx)):
            raise ValueError(
                f"Fold {i}: Overlap between train and val indices (leakage).")

        custom_cv.append((train_idx, val_idx))

    return custom_cv


# ==========================================
# 4. LOAD DATA
# ==========================================
print("📥 Loading data/folds/best-params (LASSO FS tuning)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()

folds_data = load_json(FOLDS_PATH)
best_params_data = load_json(BEST_PARAMS_PATH)

custom_cv = build_custom_cv_from_folds(df, folds_data)

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features")
print(f"✅ Using {len(custom_cv)} custom folds from: {FOLDS_PATH.name}")
print(f"✅ Best hyperparams loaded from: {BEST_PARAMS_PATH.name}")
print(f"🎯 Tuning only lasso__estimator__C over: {C_LIST} | scoring={SCORING}")
print("-" * 70)

# ==========================================
# 5. TUNE C PER MODEL
# ==========================================
summary_rows = []
cv_results_rows = []

for model_name in MODELS:
    print(f"🔧 Tuning LASSO C for: {model_name}")

    pipe = build_model_pipeline(model_name)

    # Apply fixed best classifier hyperparams from No-FS tuning
    raw_best_params = best_params_data[model_name]["best_params"]
    clf_params = to_pipeline_clf_params(raw_best_params)
    pipe.set_params(**clf_params)

    # Grid only on LASSO selector C
    param_grid = {"lasso__estimator__C": C_LIST}

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

    best_C = float(grid.best_params_["lasso__estimator__C"])
    best_score = float(grid.best_score_)

    print(f"   ✅ best_C={best_C} | best_mean_cv_f1={best_score:.4f}")

    summary_rows.append({
        "Model": model_name,
        "Best_C": best_C,
        "Best_mean_CV_F1": round(best_score, 6),
        "C_candidates": str(C_LIST)
    })

    # Save per-C results (mean/std across folds)
    res = pd.DataFrame(grid.cv_results_)
    keep_cols = ["param_lasso__estimator__C",
                 "mean_test_score", "std_test_score", "rank_test_score"]
    res = res[keep_cols].copy()
    res["Model"] = model_name
    res.rename(columns={
        "param_lasso__estimator__C": "C",
        "mean_test_score": "mean_f1",
        "std_test_score": "std_f1"
    }, inplace=True)

    for _, row in res.iterrows():
        cv_results_rows.append({
            "Model": row["Model"],
            "C": float(row["C"]),
            "mean_f1": float(row["mean_f1"]),
            "std_f1": float(row["std_f1"]),
            "rank": int(row["rank_test_score"])
        })

print("-" * 70)

# ==========================================
# 6. SAVE OUTPUTS
# ==========================================
summary_df = pd.DataFrame(summary_rows).sort_values(
    by="Best_mean_CV_F1", ascending=False)
cvres_df = pd.DataFrame(cv_results_rows).sort_values(by=["Model", "rank"])

out_json_data = {
    "scoring": SCORING,
    "C_list": C_LIST,
    "best_C_per_model": {
        r["Model"]: {"best_C": float(
            r["Best_C"]), "best_mean_cv_f1": float(r["Best_mean_CV_F1"])}
        for _, r in summary_df.iterrows()
    }
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_json_data, f, indent=4)

summary_df.to_csv(OUT_CSV, index=False)
cvres_df.to_csv(OUT_CV_RESULTS, index=False)

print(f"✅ Saved: {OUT_JSON}")
print(f"✅ Saved: {OUT_CSV}")
print(f"✅ Saved: {OUT_CV_RESULTS}")

print("\n🏁 Done. Best C per model:")
print(summary_df[["Model", "Best_C", "Best_mean_CV_F1"]])
