# 05d_extratrees_fs_tuning_find_best_k.py
# ------------------------------------------------------------
# ExtraTrees FS tuning (analogous to t-test k tuning):
# - Uses external_train_samples.csv
# - Uses custom folds from cv_folds_samples.json
# - Loads best classifier hyperparams from: results/tuning/05a_best_params_no_fs.json
# - Tunes only: fs__max_features (top-k selected by SelectFromModel)
#
# Pipeline:
#   StandardScaler -> SelectFromModel(ExtraTrees, threshold=-inf, max_features=k) -> Classifier
#
# Saves:
# - best k per model: results/evaluation_extratrees_fs/extratrees_tuning/best_k_per_model.json
# - csv summary:      results/evaluation_extratrees_fs/extratrees_tuning/best_k_per_model.csv
# - cv results:       results/evaluation_extratrees_fs/extratrees_tuning/cv_results_per_model.csv
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_JOBS = -1
SCORING = "f1"

# Διάλεξε λίστα όπως στο t-test (ή ό,τι θες)
K_LIST = [5, 8, 10, 12, 15, 18, 22]

MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"

OUT_DIR = BASE / "results" / "evaluation_extratrees_fs" / "extratrees_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "best_k_per_model.json"
OUT_CSV = OUT_DIR / "best_k_per_model.csv"
OUT_CV_RESULTS = OUT_DIR / "cv_results_per_model.csv"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_none_strings(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        out[k] = None if v == "None" else v
    return out


def to_pipeline_clf_params(best_params: dict) -> dict:
    best_params = fix_none_strings(best_params)
    out = {}
    for k, v in best_params.items():
        out[k if k.startswith("clf__") else f"clf__{k}"] = v
    return out


def build_custom_cv_from_folds(df: pd.DataFrame, folds_data: dict):
    custom_cv = []
    for i, fold_info in enumerate(folds_data["folds"], start=1):
        tr = set(fold_info["train_samples"])
        va = set(fold_info["val_samples"])
        tr_idx = df.index[df["name"].isin(tr)].tolist()
        va_idx = df.index[df["name"].isin(va)].tolist()

        if len(tr_idx) != len(tr):
            raise ValueError(f"Fold {i}: missing train samples.")
        if len(va_idx) != len(va):
            raise ValueError(f"Fold {i}: missing val samples.")
        if set(tr_idx).intersection(set(va_idx)):
            raise ValueError(f"Fold {i}: leakage overlap indices.")

        custom_cv.append((tr_idx, va_idx))
    return custom_cv


def build_model_pipeline(model_name: str):
    # ExtraTrees selector: top-k via max_features
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    fs = SelectFromModel(
        estimator=et,
        threshold=-float("inf"),  # keep all, but we cap via max_features
    )

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
        raise ValueError(model_name)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("fs", fs),
        ("clf", clf),
    ])


print("📥 Loading data/folds/best-params (ExtraTrees FS tuning)...")
df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()

folds_data = load_json(FOLDS_PATH)
best_params_data = load_json(BEST_PARAMS_PATH)
custom_cv = build_custom_cv_from_folds(df, folds_data)

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features")
print(f"✅ Using {len(custom_cv)} custom folds from: {FOLDS_PATH.name}")
print(f"🎯 Tuning only fs__max_features over: {K_LIST} | scoring={SCORING}")
print("-" * 70)

summary_rows = []
cv_results_rows = []

for model_name in MODELS:
    print(f"🔧 Tuning ExtraTrees k for: {model_name}")

    pipe = build_model_pipeline(model_name)

    raw_best = best_params_data[model_name]["best_params"]
    pipe.set_params(**to_pipeline_clf_params(raw_best))

    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"fs__max_features": K_LIST},
        scoring=SCORING,
        cv=custom_cv,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
        return_train_score=False,
    )
    grid.fit(X, y)

    best_k = int(grid.best_params_["fs__max_features"])
    best_score = float(grid.best_score_)
    print(f"   ✅ best_k={best_k} | best_mean_cv_f1={best_score:.4f}")

    summary_rows.append({
        "Model": model_name,
        "Best_k": best_k,
        "Best_mean_CV_F1": round(best_score, 6),
        "K_candidates": str(K_LIST),
    })

    res = pd.DataFrame(grid.cv_results_)
    keep_cols = ["param_fs__max_features", "mean_test_score",
                 "std_test_score", "rank_test_score"]
    res = res[keep_cols].copy()
    res.rename(columns={
        "param_fs__max_features": "k",
        "mean_test_score": "mean_f1",
        "std_test_score": "std_f1",
    }, inplace=True)
    res["Model"] = model_name

    for _, r in res.iterrows():
        cv_results_rows.append({
            "Model": r["Model"],
            "k": int(r["k"]),
            "mean_f1": float(r["mean_f1"]),
            "std_f1": float(r["std_f1"]),
            "rank": int(r["rank_test_score"]),
        })

summary_df = pd.DataFrame(summary_rows).sort_values(
    by="Best_mean_CV_F1", ascending=False)
cvres_df = pd.DataFrame(cv_results_rows).sort_values(by=["Model", "rank"])

out_json = {
    "scoring": SCORING,
    "k_list": K_LIST,
    "best_k_per_model": {
        r["Model"]: {"best_k": int(
            r["Best_k"]), "best_mean_cv_f1": float(r["Best_mean_CV_F1"])}
        for _, r in summary_df.iterrows()
    }
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=4)

summary_df.to_csv(OUT_CSV, index=False)
cvres_df.to_csv(OUT_CV_RESULTS, index=False)

print(f"✅ Saved: {OUT_JSON}")
print(f"✅ Saved: {OUT_CSV}")
print(f"✅ Saved: {OUT_CV_RESULTS}")
print("\n🏁 Done. Best k per model:")
print(summary_df[["Model", "Best_k", "Best_mean_CV_F1"]])
