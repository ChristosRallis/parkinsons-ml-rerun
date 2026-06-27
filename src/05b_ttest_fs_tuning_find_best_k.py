import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

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

K_LIST = [5, 8, 10, 12, 15, 18, 22]

MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]

# ==========================================
# 2. PATHS (SYNCED WITH YOUR PROJECT)
# ==========================================
BASE = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"

OUT_DIR = BASE / "results" / "evaluation_t-test_fs" / "t-test_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "best_k_per_model.json"
OUT_CSV = OUT_DIR / "best_k_per_model.csv"
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
        if v == "None":
            out[k] = None
        else:
            out[k] = v
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
    Pipeline: scaler -> SelectKBest(ANOVA F-test) -> classifier
    NOTE: We keep scaler for ALL models for consistency. Trees don't need it but it doesn't leak.
    """
    fs = SelectKBest(score_func=f_classif)

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
        ("fs", fs),
        ("clf", clf)
    ])
    return pipe


def build_custom_cv_from_folds(df: pd.DataFrame, folds_data: dict):
    """
    Convert fold sample names -> row indices, with sanity checks.
    Returns a list of (train_idx, val_idx) tuples usable as cv=custom_cv in sklearn.
    """
    custom_cv = []
    for i, fold_info in enumerate(folds_data["folds"], start=1):
        train_names = set(fold_info["train_samples"])
        val_names = set(fold_info["val_samples"])

        train_idx = df.index[df["name"].isin(train_names)].tolist()
        val_idx = df.index[df["name"].isin(val_names)].tolist()

        # checks: missing samples
        if len(train_idx) != len(train_names):
            raise ValueError(
                f"Fold {i}: Missing {len(train_names)-len(train_idx)} train samples in df.")
        if len(val_idx) != len(val_names):
            raise ValueError(
                f"Fold {i}: Missing {len(val_names)-len(val_idx)} val samples in df.")

        # checks: overlap
        if set(train_idx).intersection(set(val_idx)):
            raise ValueError(
                f"Fold {i}: Overlap between train and val indices (leakage).")

        custom_cv.append((train_idx, val_idx))

    return custom_cv


# ==========================================
# 4. LOAD DATA
# ==========================================
print("📥 Loading data/folds/best-params...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()

folds_data = load_json(FOLDS_PATH)
best_params_data = load_json(BEST_PARAMS_PATH)

custom_cv = build_custom_cv_from_folds(df, folds_data)

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features")
print(f"✅ Using {len(custom_cv)} custom folds from: {FOLDS_PATH.name}")
print(f"✅ Best hyperparams loaded from: {BEST_PARAMS_PATH.name}")
print(f"🎯 Tuning only fs__k over: {K_LIST} | scoring={SCORING}")
print("-" * 70)

# ==========================================
# 5. TUNE k PER MODEL
# ==========================================
summary_rows = []
cv_results_rows = []

for model_name in MODELS:
    print(f"🔧 Tuning t-test k for: {model_name}")

    # Build pipeline
    pipe = build_model_pipeline(model_name)

    # Load & apply fixed best classifier hyperparams from previous phase
    raw_best_params = best_params_data[model_name]["best_params"]
    clf_params = to_pipeline_clf_params(raw_best_params)
    pipe.set_params(**clf_params)

    # Grid only on k
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

    print(f"   ✅ best_k={best_k} | best_mean_cv_f1={best_score:.4f}")

    summary_rows.append({
        "Model": model_name,
        "Best_k": best_k,
        "Best_mean_CV_F1": round(best_score, 6),
        "K_candidates": str(K_LIST)
    })

    # Save per-k results (mean/std across folds)
    res = pd.DataFrame(grid.cv_results_)
    # Keep only useful columns
    keep_cols = ["param_fs__k", "mean_test_score",
                 "std_test_score", "rank_test_score"]
    res = res[keep_cols].copy()
    res["Model"] = model_name
    res.rename(columns={"param_fs__k": "k", "mean_test_score": "mean_f1",
               "std_test_score": "std_f1"}, inplace=True)

    for _, row in res.iterrows():
        cv_results_rows.append({
            "Model": row["Model"],
            "k": int(row["k"]),
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

# JSON output (easy to load later)
out_json_data = {
    "scoring": SCORING,
    "k_list": K_LIST,
    "best_k_per_model": {
        r["Model"]: {"best_k": int(
            r["Best_k"]), "best_mean_cv_f1": float(r["Best_mean_CV_F1"])}
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

print("\n🏁 Done. Best k per model:")
print(summary_df[["Model", "Best_k", "Best_mean_CV_F1"]])
