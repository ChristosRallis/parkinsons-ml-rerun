# 06_ttest_fs_cv_evaluation_final.py
# ------------------------------------------------------------
# 3-fold CV evaluation using:
# - fixed hyperparameters from: results/tuning/05a_best_params_no_fs.json
# - fixed best_k per model from: results/evaluation_t-test_fs/t-test_tuning/best_k_per_model.json
# - custom folds from: results/cv_folds_samples.json
#
# Saves:
# - metrics per fold:   results/evaluation_t-test_fs/metrics/(1)_ttest_fs_metrics_per_fold.csv
# - summary mean±std:   results/evaluation_t-test_fs/metrics/(1)_ttest_fs_metrics_summary.csv
# - raw predictions:    results/evaluation_t-test_fs/predictions/(1)_predictions_<Model>.csv
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. PATHS & CONFIG (SYNCED WITH YOUR PROJECT)
# ==========================================
RANDOM_STATE = 42

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"

# best_k from the previous step
BEST_K_PATH = (
    BASE
    / "results"
    / "evaluation_t-test_fs"
    / "t-test_tuning"
    / "best_k_per_model.json"
)

# Outputs (new t-test FS evaluation folder)
EVAL_DIR = BASE / "results" / "evaluation_t-test_fs"
METRICS_DIR = EVAL_DIR / "metrics"
PREDS_DIR = EVAL_DIR / "predictions"
for d in [EVAL_DIR, METRICS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Models order
MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]


# ==========================================
# 2. HELPERS
# ==========================================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_none_strings(params: dict) -> dict:
    """Convert 'None' strings to actual None."""
    out = {}
    for k, v in params.items():
        out[k] = None if v == "None" else v
    return out


def clean_params_for_model(model_name: str, params: dict) -> dict:
    """
    Your best_params json may store params either with 'clf__' prefix (for pipelines)
    or without prefix (for non-pipeline models). Here we ALWAYS use a Pipeline
    (scaler -> fs -> clf) for all models, so classifier params must be 'clf__...'.
    """
    params = clean_none_strings(params)

    cleaned = {}
    for k, v in params.items():
        if k.startswith("clf__"):
            cleaned[k] = v
        else:
            cleaned[f"clf__{k}"] = v
    return cleaned


def safe_roc_auc(y_true, y_score):
    """Return NaN if only one class in y_true."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan


def build_pipeline(model_name: str, best_k: int, clf_params: dict):
    """
    Pipeline: StandardScaler -> SelectKBest(ANOVA F-test) -> Classifier
    We keep scaling for all models for consistency; it does not cause leakage.
    """
    fs = SelectKBest(score_func=f_classif, k=best_k)

    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=RANDOM_STATE
        )
    elif model_name == "LinearSVM":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "XGBoost":
        clf = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("fs", fs),
            ("clf", clf),
        ]
    )

    # Apply classifier hyperparams under clf__ prefix
    pipe.set_params(**clf_params)
    return pipe


# ==========================================
# 3. LOAD DATA
# ==========================================
print("Loading Data and Configuration (t-test FS CV evaluation)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()
names_arr = df["name"].values

folds_data = load_json(FOLDS_PATH)
best_params_data = load_json(BEST_PARAMS_PATH)
best_k_data = load_json(BEST_K_PATH)

# Extract best_k per model
best_k_per_model = best_k_data.get("best_k_per_model", {})
if not best_k_per_model:
    raise ValueError(
        f"Could not find 'best_k_per_model' inside {BEST_K_PATH}. "
        "Open the json and confirm its structure."
    )

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features")
print(f"✅ Folds: {FOLDS_PATH}")
print(f"✅ Best params: {BEST_PARAMS_PATH}")
print(f"✅ Best k: {BEST_K_PATH}")
print("-" * 70)

# ==========================================
# 4. EVALUATION LOOP (PER FOLD)
# ==========================================
fold_metrics = []
all_predictions = []

print("Starting 3-fold CV evaluation (t-test FS)...")
print("=" * 70)

for fold_idx, fold_info in enumerate(folds_data["folds"]):
    fold_num = fold_idx + 1

    train_names = set(fold_info["train_samples"])
    val_names = set(fold_info["val_samples"])

    train_idx = df.index[df["name"].isin(train_names)].tolist()
    val_idx = df.index[df["name"].isin(val_names)].tolist()

    # Quick integrity check
    if set(train_idx).intersection(set(val_idx)):
        raise ValueError(
            f"Leakage detected in fold {fold_num}: overlap indices.")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    val_names_current = names_arr[val_idx]

    print(f"🔄 Fold {fold_num}: Train={len(train_idx)}, Val={len(val_idx)}")

    for model_name in MODELS:
        # best k for this model
        if model_name not in best_k_per_model:
            raise KeyError(
                f"Missing best_k for model '{model_name}' in {BEST_K_PATH}")

        k = int(best_k_per_model[model_name]["best_k"])

        # best hyperparams for this model
        raw_params = best_params_data[model_name]["best_params"]
        clf_params = clean_params_for_model(model_name, raw_params)

        try:
            model = build_pipeline(model_name, k, clf_params)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_val)

            # Scores (probabilities preferred)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_val)[:, 1]
            else:
                # SVC has probability=True here; this is a fallback
                y_score = model.decision_function(X_val)

            auc_val = safe_roc_auc(y_val, y_score)

            scores = {
                "Fold": fold_num,
                "Model": model_name,
                "k_used": k,
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred, zero_division=0),
                "Recall": recall_score(y_val, y_pred, zero_division=0),
                "F1": f1_score(y_val, y_pred, zero_division=0),
                "ROC_AUC": auc_val,
            }
            fold_metrics.append(scores)

            preds_df = pd.DataFrame(
                {
                    "sample_name": val_names_current,
                    "y_true": y_val,
                    "y_prob": y_score,  # keep as score/prob for threshold tuning later
                    "fold": fold_num,
                    "model": model_name,
                    "k_used": k,
                }
            )
            all_predictions.append(preds_df)

        except Exception as e:
            print(f"❌ Error in Fold {fold_num} - Model {model_name}: {e}")

print("=" * 70)

# ==========================================
# 5. SAVE RESULTS
# ==========================================
metrics_df = pd.DataFrame(fold_metrics)

metrics_file = METRICS_DIR / "(1)_ttest_fs_metrics_per_fold.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"✅ Saved metrics per fold: {metrics_file}")

summary = (
    metrics_df.groupby("Model")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
    .agg(["mean", "std"])
    .sort_index()
)

summary_file = METRICS_DIR / "(1)_ttest_fs_metrics_summary.csv"
summary.to_csv(summary_file)
print(f"✅ Saved summary metrics: {summary_file}")

# Raw predictions split per model (like your no_fs workflow)
if all_predictions:
    total_preds_df = pd.concat(all_predictions, ignore_index=True)

    for model_name in total_preds_df["model"].unique():
        model_preds = total_preds_df[total_preds_df["model"]
                                     == model_name].copy()
        model_preds = model_preds.drop(columns=["model"])

        pred_file = PREDS_DIR / f"(1)_predictions_{model_name}.csv"
        model_preds.to_csv(pred_file, index=False)
        print(f"✅ Saved raw predictions for {model_name}: {pred_file}")

print("\n✅ CV Evaluation Complete (t-test FS).")
print("Summary Preview:")
print(summary)
