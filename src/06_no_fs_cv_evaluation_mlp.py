# 06_no_fs_cv_evaluation_mlp.py
# ------------------------------------------------------------
# 3-fold CV evaluation for MLP (NO Feature Selection) using:
# - custom folds from: results/cv_folds_samples.json
# - best MLP params from: results/tuning_mlp/05a_best_params_no_fs_mlp.json
#
# Saves:
# - metrics per fold:   results/evaluation_no_fs_mlp/metrics/(1)_no_fs_mlp_metrics_per_fold.csv
# - summary mean±std:   results/evaluation_no_fs_mlp/metrics/(1)_no_fs_mlp_metrics_summary.csv
# - raw predictions:    results/evaluation_no_fs_mlp/predictions/(1)_predictions_MLP.csv
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ==========================================
# PATHS & CONFIG
# ==========================================
RANDOM_STATE = 42

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"

BEST_MLP_PARAMS_PATH = BASE / "results" / \
    "tuning_mlp" / "05a_best_params_no_fs_mlp.json"

EVAL_DIR = BASE / "results" / "evaluation_no_fs_mlp"
METRICS_DIR = EVAL_DIR / "metrics"
PREDS_DIR = EVAL_DIR / "predictions"
for d in [EVAL_DIR, METRICS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==========================================
# HELPERS
# ==========================================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def build_mlp_pipeline(best_params: dict):
    """
    Pipeline: StandardScaler -> MLPClassifier
    best_params keys are like: 'mlp__hidden_layer_sizes', ...
    """
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

    # Apply tuned params
    pipe.set_params(**best_params)
    return pipe


# ==========================================
# LOAD DATA
# ==========================================
print("Loading Data and Configuration (No-FS MLP CV evaluation)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()
names_arr = df["name"].values

folds_data = load_json(FOLDS_PATH)
best_mlp_data = load_json(BEST_MLP_PARAMS_PATH)

best_params = best_mlp_data.get("best_params", None)
if best_params is None:
    raise ValueError(
        f"Could not find 'best_params' inside {BEST_MLP_PARAMS_PATH}")

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features")
print(f"✅ Folds: {FOLDS_PATH}")
print(f"✅ Best MLP params: {BEST_MLP_PARAMS_PATH}")
print("-" * 70)

# ==========================================
# EVALUATION LOOP (PER FOLD)
# ==========================================
fold_metrics = []
all_predictions = []

print("Starting 3-fold CV evaluation (No-FS MLP)...")
print("=" * 70)

for fold_idx, fold_info in enumerate(folds_data["folds"]):
    fold_num = fold_idx + 1

    train_names = set(fold_info["train_samples"])
    val_names = set(fold_info["val_samples"])

    train_idx = df.index[df["name"].isin(train_names)].tolist()
    val_idx = df.index[df["name"].isin(val_names)].tolist()

    if set(train_idx).intersection(set(val_idx)):
        raise ValueError(
            f"Leakage detected in fold {fold_num}: overlap indices.")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    val_names_current = names_arr[val_idx]

    print(f"🔄 Fold {fold_num}: Train={len(train_idx)}, Val={len(val_idx)}")

    model = build_mlp_pipeline(best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_score = model.predict_proba(X_val)[:, 1]

    scores = {
        "Fold": fold_num,
        "Model": "MLP",
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "ROC_AUC": safe_roc_auc(y_val, y_score),
    }
    fold_metrics.append(scores)

    preds_df = pd.DataFrame({
        "sample_name": val_names_current,
        "y_true": y_val,
        "y_prob": y_score,
        "fold": fold_num,
    })
    all_predictions.append(preds_df)

print("=" * 70)

# ==========================================
# SAVE RESULTS
# ==========================================
metrics_df = pd.DataFrame(fold_metrics)

metrics_file = METRICS_DIR / "(1)_no_fs_mlp_metrics_per_fold.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"✅ Saved metrics per fold: {metrics_file}")

summary = (
    metrics_df[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
    .agg(["mean", "std"])
)

summary_file = METRICS_DIR / "(1)_no_fs_mlp_metrics_summary.csv"
summary.to_csv(summary_file)
print(f"✅ Saved summary metrics: {summary_file}")

if all_predictions:
    total_preds_df = pd.concat(all_predictions, ignore_index=True)
    pred_file = PREDS_DIR / "(1)_predictions_MLP.csv"
    total_preds_df.to_csv(pred_file, index=False)
    print(f"✅ Saved raw predictions: {pred_file}")

print("\n✅ CV Evaluation Complete (No-FS MLP).")
print("Summary Preview:")
print(summary)
