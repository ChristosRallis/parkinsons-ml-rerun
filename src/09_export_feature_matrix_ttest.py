# export_feature_matrix_ttest.py
# ------------------------------------------------------------
# Creates the Feature Selection Matrix for the T-TEST (SelectKBest) pipeline.
#
# Output:
#   results/feature_matrices/feature_selection_matrix_ttest.csv
#
# Matrix format:
#   - Rows: Features (22)
#   - Columns: Algorithms (LogisticRegression, LinearSVM, RandomForest, XGBoost)
#   - Values: 1 (selected) / 0 (not selected)
#   - Extra column: Selected_By_Count (how many models selected the feature)
#
# Uses FINAL settings (full external_train fit):
#   - best classifier hyperparams from: results/tuning/05a_best_params_no_fs.json
#   - best k per model from: results/evaluation_t-test_fs/t-test_tuning/best_k_per_model.json
# ------------------------------------------------------------

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
N_JOBS = -1
MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]

# =========================
# PATHS
# =========================
BASE = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE / "results" / "external_train_samples.csv"

BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"
BEST_K_PATH = (
    BASE
    / "results"
    / "evaluation_t-test_fs"
    / "t-test_tuning"
    / "best_k_per_model.json"
)

OUT_DIR = BASE / "results" / "feature_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "feature_selection_matrix_ttest.csv"


# =========================
# HELPERS
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_none_strings(params: dict) -> dict:
    """Convert 'None' strings to None."""
    return {k: (None if v == "None" else v) for k, v in params.items()}


def to_pipeline_clf_params(params: dict) -> dict:
    """
    Ensure classifier params are under 'clf__' prefix (because we always use a Pipeline).
    """
    params = clean_none_strings(params)
    out = {}
    for k, v in params.items():
        out[k if k.startswith("clf__") else f"clf__{k}"] = v
    return out


def build_ttest_pipeline(model_name: str, k: int, clf_params: dict):
    fs = SelectKBest(score_func=f_classif, k=int(k))

    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)
    elif model_name == "LinearSVM":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_name == "XGBoost":
        clf = XGBClassifier(eval_metric="logloss",
                            random_state=RANDOM_STATE, n_jobs=N_JOBS)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("fs", fs),
        ("clf", clf),
    ])

    pipe.set_params(**clf_params)
    return pipe


# =========================
# MAIN
# =========================
print("📥 Loading external_train and configs (T-TEST Feature Matrix)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

# Features & target
X = df.drop(columns=["name", "status", "subject_id"], errors="ignore")
y = df["status"].astype(int).to_numpy()

feature_names = list(X.columns)
print(f"✅ Train loaded: {X.shape[0]} samples, {X.shape[1]} features")

best_params_data = load_json(BEST_PARAMS_PATH)
best_k_data = load_json(BEST_K_PATH)

best_k_per_model = best_k_data.get("best_k_per_model", {})
if not best_k_per_model:
    raise ValueError(
        f"Could not find 'best_k_per_model' inside: {BEST_K_PATH}")

# Prepare empty matrix (rows=features, cols=models)
matrix = pd.DataFrame(index=feature_names, columns=MODELS, data=0, dtype=int)

# Fit FINAL model per algorithm on FULL external_train and extract support mask
for model_name in MODELS:
    if model_name not in best_k_per_model:
        raise KeyError(
            f"Missing best_k for model '{model_name}' in {BEST_K_PATH}")

    k = int(best_k_per_model[model_name]["best_k"])
    raw_params = best_params_data[model_name]["best_params"]
    clf_params = to_pipeline_clf_params(raw_params)

    model = build_ttest_pipeline(model_name, k, clf_params)
    model.fit(X, y)

    # boolean mask length = n_features
    support = model.named_steps["fs"].get_support()
    if len(support) != len(feature_names):
        raise RuntimeError("Support mask length mismatch with feature names.")

    selected_features = [f for f, keep in zip(feature_names, support) if keep]
    print(f"🔎 {model_name}: k={k} | selected={len(selected_features)}")

    # Fill matrix column
    matrix.loc[selected_features, model_name] = 1

# Add helpful column for Excel sorting
matrix["Selected_By_Count"] = matrix[MODELS].sum(axis=1)

# Sort features: most commonly selected first (optional but useful)
matrix = matrix.sort_values(by="Selected_By_Count", ascending=False)

# Save
matrix.to_csv(OUT_CSV, index=True)
print(f"\n✅ Saved Feature Selection Matrix (T-TEST) to:\n{OUT_CSV}")
print("\nTop rows preview:")
print(matrix.head(10))
