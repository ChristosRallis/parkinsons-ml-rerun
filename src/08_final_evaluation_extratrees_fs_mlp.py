# 08_final_evaluation_extratrees_fs_mlp.py
# ------------------------------------------------------------
# FINAL EVALUATION: EXTRATREES FEATURE SELECTION + MLP PIPELINE
#
# Uses:
# - best MLP hyperparams: results/tuning_mlp/05a_best_params_no_fs_mlp.json
# - best k:              results/evaluation_extratrees_fs_mlp/extratrees_tuning_mlp/best_k_mlp.json
# - tuned threshold:     results/evaluation_extratrees_fs_mlp/threshold_tuning/(2)_best_thresholds_constrained_mlp.csv
#
# Trains on full external_train_samples.csv and evaluates on external_test_samples.csv
#
# Saves:
# - metrics:      results/evaluation_extratrees_fs_mlp/final_test_results/final_test_metrics.csv
# - predictions:  results/evaluation_extratrees_fs_mlp/final_test_results/final_test_predictions.csv
# - plots:        cm_MLP.png, roc_curve_MLP.png
# ------------------------------------------------------------

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ==========================================
# PATHS & CONFIG
# ==========================================
RANDOM_STATE = 42

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
TEST_PATH = BASE / "results" / "external_test_samples.csv"

BEST_MLP_PARAMS_PATH = BASE / "results" / \
    "tuning_mlp" / "05a_best_params_no_fs_mlp.json"
BEST_K_PATH = BASE / "results" / "evaluation_extratrees_fs_mlp" / \
    "extratrees_tuning_mlp" / "best_k_mlp.json"
THRESHOLDS_PATH = BASE / "results" / "evaluation_extratrees_fs_mlp" / \
    "threshold_tuning" / "(2)_best_thresholds_constrained_mlp.csv"

OUT_DIR = BASE / "results" / "evaluation_extratrees_fs_mlp" / "final_test_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# HELPERS
# ==========================================


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_clean_Xy(csv_path: Path, label: str):
    df = pd.read_csv(csv_path)
    if "status" not in df.columns:
        raise ValueError(f"Missing 'status' column in {label}: {csv_path}")

    y = df["status"].astype(int).to_numpy()
    X = df.drop(columns=["status"], errors="ignore")
    drop_cols = [c for c in ["name", "subject_id",
                             "Unnamed: 0"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    names = df["name"].values if "name" in df.columns else np.arange(len(df))
    return X, y, names


def build_pipeline(best_k: int, best_mlp_params: dict):
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    fs = SelectFromModel(
        estimator=et,
        threshold=-np.inf,
        max_features=best_k
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
        ("fs", fs),
        ("mlp", mlp),
    ])

    pipe.set_params(**best_mlp_params)
    return pipe


# ==========================================
# EXECUTION
# ==========================================
print("🚀 Starting FINAL evaluation on EXTERNAL TEST (Pipeline: ExtraTrees FS MLP)...")

# Load configs
best_mlp_data = load_json(BEST_MLP_PARAMS_PATH)
best_mlp_params = best_mlp_data.get("best_params", None)
if best_mlp_params is None:
    raise ValueError(f"Could not find 'best_params' in {BEST_MLP_PARAMS_PATH}")

best_k_data = load_json(BEST_K_PATH)
best_k = int(best_k_data.get("best_k_mlp", {}).get("best_k", None))
if best_k is None:
    raise ValueError(f"Could not find best_k in {BEST_K_PATH}")

th_df = pd.read_csv(THRESHOLDS_PATH)
best_threshold = float(
    th_df.loc[0, "Best_Threshold"]) if "Best_Threshold" in th_df.columns else 0.5

# Load data
X_train, y_train, train_names = load_clean_Xy(TRAIN_PATH, "TRAIN")
X_test, y_test, test_names = load_clean_Xy(TEST_PATH, "TEST")

# Align columns (safety)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"✅ Data Loaded & Aligned. Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Loaded threshold for MLP: {best_threshold}")
print(f"✅ Using best_k for ExtraTrees selector: {best_k}")

# Train
model = build_pipeline(best_k, best_mlp_params)
model.fit(X_train, y_train)

# Predict
y_prob = model.predict_proba(X_test)[:, 1]

# Apply threshold
th = float(best_threshold)
y_pred = (y_prob >= th).astype(int)

# Metrics
auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else np.nan
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, zero_division=0)
prec = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# How many features selected on full train
selector = model.named_steps["fs"]
n_selected_full_train = int(np.sum(selector.get_support()))

results_df = pd.DataFrame([{
    "Model": "MLP",
    "k_used": best_k,
    "Threshold_Used": th,
    "n_selected_features_full_train": n_selected_full_train,
    "AUC": auc,
    "Accuracy": acc,
    "Recall": rec,
    "Precision": prec,
    "F1": f1
}])

# Save predictions
preds_rows = []
for sn, yt, yp, ypr in zip(test_names, y_test, y_prob, y_pred):
    preds_rows.append({
        "sample_name": sn,
        "Model": "MLP",
        "y_true": int(yt),
        "y_prob": float(yp),
        "y_pred": int(ypr),
        "k_used": best_k,
        "Threshold_Used": th
    })
preds_df = pd.DataFrame(preds_rows)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt_cm = plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(
    f"MLP (ExtraTrees k={best_k}, Th={th:.2f})\nExternal Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT_DIR / "cm_MLP.png", dpi=300)
plt.close(plt_cm)

# ROC curve
if len(np.unique(y_test)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"MLP (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve on External Test Set (ExtraTrees FS MLP)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve_MLP.png", dpi=300)
    plt.close()

# Save CSVs
metrics_path = OUT_DIR / "final_test_metrics.csv"
preds_path = OUT_DIR / "final_test_predictions.csv"
results_df.to_csv(metrics_path, index=False)
preds_df.to_csv(preds_path, index=False)

print("\n" + "=" * 70)
print("🏁 FINAL EXTERNAL TEST RESULTS (ExtraTrees FS MLP)")
print("=" * 70)
print(results_df)
print("=" * 70)
print(f"✅ Saved metrics: {metrics_path}")
print(f"✅ Saved predictions: {preds_path}")
print(f"✅ Saved plots in: {OUT_DIR}")
