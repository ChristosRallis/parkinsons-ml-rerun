# src/09_export_feature_matrix_lasso.py
# ------------------------------------------------------------
# Creates the Feature Selection Matrix for the LASSO FS pipeline.
#
# Output:
#   results/feature_matrices/feature_selection_matrix_lasso.csv
#
# Matrix format:
#   - Rows: Features (22)
#   - Columns: Algorithms (LogisticRegression, LinearSVM, RandomForest, XGBoost)
#   - Values: 1 (selected) / 0 (not selected)
#   - Extra column: Selected_By_Count (how many models selected the feature)
#
# Uses FINAL settings (fit on FULL external_train):
#   - best classifier hyperparams from: results/tuning/05a_best_params_no_fs.json
#   - best C per model from: results/evaluation_lasso_fs/lasso_tuning/best_C_per_model.json
#
# IMPORTANT:
#   This extracts selected features from the LASSO selector step ONLY.
#   It does not try to interpret downstream model feature importance.
# ------------------------------------------------------------

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

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
BEST_C_PATH = BASE / "results" / "evaluation_lasso_fs" / \
    "lasso_tuning" / "best_C_per_model.json"

OUT_DIR = BASE / "results" / "feature_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "feature_selection_matrix_lasso.csv"


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
    """Ensure classifier params are under 'clf__' prefix (Pipeline usage)."""
    params = clean_none_strings(params)
    out = {}
    for k, v in params.items():
        out[k if k.startswith("clf__") else f"clf__{k}"] = v
    return out


def build_lasso_pipeline(model_name: str, best_C: float, clf_params: dict):
    # LASSO selector (embedded)
    lasso_est = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=float(best_C),
        max_iter=5000,
        random_state=RANDOM_STATE,
        # if you used class_weight="balanced" in your LASSO scripts, add it here too
        # class_weight="balanced",
    )

    fs = SelectFromModel(
        estimator=lasso_est,
        threshold="mean",  # MUST match your LASSO pipeline
    )

    # Downstream classifier (same 4 algorithms)
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
        ("lasso", fs),   # <-- step name used later for get_support()
        ("clf", clf),
    ])

    pipe.set_params(**clf_params)
    return pipe


# =========================
# MAIN
# =========================
print("📥 Loading external_train and configs (LASSO Feature Matrix)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

X = df.drop(columns=["name", "status", "subject_id"], errors="ignore")
y = df["status"].astype(int).to_numpy()

feature_names = list(X.columns)
print(f"✅ Train loaded: {X.shape[0]} samples, {X.shape[1]} features")

best_params_data = load_json(BEST_PARAMS_PATH)
best_c_data = load_json(BEST_C_PATH)

best_c_per_model = best_c_data.get("best_C_per_model", {})
if not best_c_per_model:
    raise ValueError(
        f"Could not find 'best_C_per_model' inside: {BEST_C_PATH}")

# Prepare empty matrix
matrix = pd.DataFrame(index=feature_names, columns=MODELS, data=0, dtype=int)

# Fit FINAL model per algorithm on FULL external_train and extract selector support mask
for model_name in MODELS:
    if model_name not in best_c_per_model:
        raise KeyError(
            f"Missing best_C for model '{model_name}' in {BEST_C_PATH}")

    best_C = float(best_c_per_model[model_name]["best_C"])
    raw_params = best_params_data[model_name]["best_params"]
    clf_params = to_pipeline_clf_params(raw_params)

    model = build_lasso_pipeline(model_name, best_C, clf_params)
    model.fit(X, y)

    support = model.named_steps["lasso"].get_support()  # boolean mask
    if len(support) != len(feature_names):
        raise RuntimeError("Support mask length mismatch with feature names.")

    selected_features = [f for f, keep in zip(feature_names, support) if keep]
    n_sel = len(selected_features)

    # Safety: ensure at least 1 feature selected
    if n_sel == 0:
        raise RuntimeError(
            f"LASSO selected 0 features for {model_name}. "
            "Consider changing threshold or C grid."
        )

    print(f"🔎 {model_name}: best_C={best_C} | selected={n_sel}")

    matrix.loc[selected_features, model_name] = 1

# Add helper column for Excel sorting
matrix["Selected_By_Count"] = matrix[MODELS].sum(axis=1)

# Sort: most commonly selected first
matrix = matrix.sort_values(by="Selected_By_Count", ascending=False)

# Save
matrix.to_csv(OUT_CSV, index=True)
print(f"\n✅ Saved Feature Selection Matrix (LASSO) to:\n{OUT_CSV}")
print("\nTop rows preview:")
print(matrix.head(10))
