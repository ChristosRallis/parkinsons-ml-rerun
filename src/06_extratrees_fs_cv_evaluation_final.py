# 06_extratrees_fs_cv_evaluation_final.py
# ------------------------------------------------------------
# 3-fold CV evaluation for ExtraTrees FS pipeline using:
# - best classifier hyperparams: results/tuning/05a_best_params_no_fs.json
# - best k per model: results/evaluation_extratrees_fs/extratrees_tuning/best_k_per_model.json
# - custom folds: results/cv_folds_samples.json
#
# Saves:
# - metrics per fold: results/evaluation_extratrees_fs/metrics/(1)_extratrees_fs_metrics_per_fold.csv
# - summary:          results/evaluation_extratrees_fs/metrics/(1)_extratrees_fs_metrics_summary.csv
# - predictions:      results/evaluation_extratrees_fs/predictions/(1)_predictions_<Model>.csv
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_JOBS = -1
MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"
BEST_K_PATH = BASE / "results" / "evaluation_extratrees_fs" / \
    "extratrees_tuning" / "best_k_per_model.json"

EVAL_DIR = BASE / "results" / "evaluation_extratrees_fs"
METRICS_DIR = EVAL_DIR / "metrics"
PREDS_DIR = EVAL_DIR / "predictions"
for d in [EVAL_DIR, METRICS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_none_strings(params: dict) -> dict:
    return {k: (None if v == "None" else v) for k, v in params.items()}


def to_pipeline_clf_params(params: dict) -> dict:
    params = clean_none_strings(params)
    out = {}
    for k, v in params.items():
        out[k if k.startswith("clf__") else f"clf__{k}"] = v
    return out


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def build_pipeline(model_name: str, best_k: int, clf_params: dict):
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    fs = SelectFromModel(estimator=et, threshold=-
                         float("inf"), max_features=int(best_k))

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

    pipe = Pipeline([("scaler", StandardScaler()), ("fs", fs), ("clf", clf)])
    pipe.set_params(**clf_params)
    return pipe


print("Loading Data and Configuration (ExtraTrees FS CV evaluation)...")
df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()
names_arr = df["name"].values

folds_data = load_json(FOLDS_PATH)
best_params_data = load_json(BEST_PARAMS_PATH)
best_k_data = load_json(BEST_K_PATH)["best_k_per_model"]

fold_metrics = []
all_preds = []

print("Starting 3-fold CV evaluation (ExtraTrees FS)...")
print("=" * 70)

for fold_idx, fold in enumerate(folds_data["folds"], start=1):
    tr = set(fold["train_samples"])
    va = set(fold["val_samples"])
    tr_idx = df.index[df["name"].isin(tr)].tolist()
    va_idx = df.index[df["name"].isin(va)].tolist()

    if set(tr_idx).intersection(set(va_idx)):
        raise ValueError(f"Leakage in fold {fold_idx}")

    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]
    va_names = names_arr[va_idx]

    print(f"🔄 Fold {fold_idx}: Train={len(tr_idx)}, Val={len(va_idx)}")

    for model_name in MODELS:
        k = int(best_k_data[model_name]["best_k"])
        clf_params = to_pipeline_clf_params(
            best_params_data[model_name]["best_params"])

        model = build_pipeline(model_name, k, clf_params)
        model.fit(X_tr, y_tr)

        # selected features count (should equal k, but keep for sanity)
        n_sel = int(np.sum(model.named_steps["fs"].get_support()))

        y_pred = model.predict(X_va)
        y_score = model.predict_proba(X_va)[:, 1] if hasattr(
            model, "predict_proba") else model.decision_function(X_va)

        fold_metrics.append({
            "Fold": fold_idx,
            "Model": model_name,
            "k_used": k,
            "n_selected_features": n_sel,
            "Accuracy": accuracy_score(y_va, y_pred),
            "Precision": precision_score(y_va, y_pred, zero_division=0),
            "Recall": recall_score(y_va, y_pred, zero_division=0),
            "F1": f1_score(y_va, y_pred, zero_division=0),
            "ROC_AUC": safe_auc(y_va, y_score),
        })

        all_preds.append(pd.DataFrame({
            "sample_name": va_names,
            "y_true": y_va,
            "y_prob": y_score,
            "fold": fold_idx,
            "model": model_name,
            "k_used": k,
            "n_selected_features": n_sel,
        }))

metrics_df = pd.DataFrame(fold_metrics)
metrics_file = METRICS_DIR / "(1)_extratrees_fs_metrics_per_fold.csv"
metrics_df.to_csv(metrics_file, index=False)

summary = (
    metrics_df.groupby("Model")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "n_selected_features"]]
    .agg(["mean", "std"])
    .sort_index()
)
summary_file = METRICS_DIR / "(1)_extratrees_fs_metrics_summary.csv"
summary.to_csv(summary_file)

total_preds = pd.concat(all_preds, ignore_index=True)
for model_name in total_preds["model"].unique():
    out = PREDS_DIR / f"(1)_predictions_{model_name}.csv"
    total_preds[total_preds["model"] == model_name].drop(
        columns=["model"]).to_csv(out, index=False)

print(f"✅ Saved metrics per fold: {metrics_file}")
print(f"✅ Saved summary metrics: {summary_file}")
print("✅ CV Evaluation Complete (ExtraTrees FS).")
print(summary)
