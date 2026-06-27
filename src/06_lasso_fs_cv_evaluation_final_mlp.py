# 06_lasso_fs_cv_evaluation_final_mlp.py
# ------------------------------------------------------------
# 3-fold CV evaluation for LASSO Feature Selection + MLP using:
# - custom folds from: results/cv_folds_samples.json
# - fixed best MLP params from: results/tuning_mlp/05a_best_params_no_fs_mlp.json
# - fixed best_C from: results/evaluation_lasso_fs_mlp/lasso_tuning_mlp/best_C_mlp.json
#
# Saves:
# - metrics per fold:   results/evaluation_lasso_fs_mlp/metrics/(1)_lasso_fs_mlp_metrics_per_fold.csv
# - summary mean±std:   results/evaluation_lasso_fs_mlp/metrics/(1)_lasso_fs_mlp_metrics_summary.csv
# - raw predictions:    results/evaluation_lasso_fs_mlp/predictions/(1)_predictions_MLP.csv
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
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
BEST_C_PATH = BASE / "results" / "evaluation_lasso_fs_mlp" / \
    "lasso_tuning_mlp" / "best_C_mlp.json"

EVAL_DIR = BASE / "results" / "evaluation_lasso_fs_mlp"
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


def build_pipeline(best_C: float, best_mlp_params: dict):
    # LASSO selector (embedded)
    lasso_est = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        C=best_C
    )

    fs = SelectFromModel(
        estimator=lasso_est,
        threshold="mean"
    )

    mlp = MLPClassifier(
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", fs),
        ("mlp", mlp)
    ])

    # apply tuned MLP params (mlp__...)
    pipe.set_params(**best_mlp_params)
    return pipe


# ==========================================
# LOAD DATA
# ==========================================
print("Loading Data and Configuration (LASSO FS MLP CV evaluation)...")

df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()
names_arr = df["name"].values

folds_data = load_json(FOLDS_PATH)
best_mlp_data = load_json(BEST_MLP_PARAMS_PATH)
best_c_data = load_json(BEST_C_PATH)

best_mlp_params = best_mlp_data.get("best_params", None)
if best_mlp_params is None:
    raise ValueError(
        f"Could not find 'best_params' inside {BEST_MLP_PARAMS_PATH}")

best_C = float(best_c_data.get("best_C_mlp", {}).get("best_C", None))
if best_C is None:
    raise ValueError(f"Could not find best_C in {BEST_C_PATH}")

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features")
print(f"✅ Best C (MLP): {best_C}")
print(f"✅ Folds: {FOLDS_PATH}")
print(f"✅ Best MLP params: {BEST_MLP_PARAMS_PATH}")
print("-" * 70)

# ==========================================
# EVALUATION LOOP (PER FOLD)
# ==========================================
fold_metrics = []
all_predictions = []

print("Starting 3-fold CV evaluation (LASSO FS MLP)...")
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

    model = build_pipeline(best_C, best_mlp_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_score = model.predict_proba(X_val)[:, 1]

    selector = model.named_steps["lasso"]
    n_selected = int(np.sum(selector.get_support()))

    scores = {
        "Fold": fold_num,
        "Model": "MLP",
        "best_C_used": best_C,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "ROC_AUC": safe_roc_auc(y_val, y_score),
        "n_selected_features": n_selected
    }
    fold_metrics.append(scores)

    preds_df = pd.DataFrame({
        "sample_name": val_names_current,
        "y_true": y_val,
        "y_prob": y_score,
        "fold": fold_num,
        "best_C_used": best_C
    })
    all_predictions.append(preds_df)

print("=" * 70)

# ==========================================
# SAVE RESULTS
# ==========================================
metrics_df = pd.DataFrame(fold_metrics)

metrics_file = METRICS_DIR / "(1)_lasso_fs_mlp_metrics_per_fold.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"✅ Saved metrics per fold: {metrics_file}")

summary = (
    metrics_df.groupby("Model")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "n_selected_features"]]
    .agg(["mean", "std"])
)

summary_file = METRICS_DIR / "(1)_lasso_fs_mlp_metrics_summary.csv"
summary.to_csv(summary_file)
print(f"✅ Saved summary metrics: {summary_file}")

if all_predictions:
    total_preds_df = pd.concat(all_predictions, ignore_index=True)
    pred_file = PREDS_DIR / "(1)_predictions_MLP.csv"
    total_preds_df.to_csv(pred_file, index=False)
    print(f"✅ Saved raw predictions: {pred_file}")

print("\n✅ CV Evaluation Complete (LASSO FS MLP).")
print("Summary Preview:")
print(summary)
