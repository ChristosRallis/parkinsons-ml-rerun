# 07_threshold_tuning_extratrees_fs_mlp_constrained.py
# ------------------------------------------------------------
# Threshold tuning on out-of-fold predictions produced by:
# results/evaluation_extratrees_fs_mlp/predictions/(1)_predictions_MLP.csv
#
# Strategy:
# 1) Constraint: Recall >= MIN_RECALL_CONSTRAINT
# 2) Among valid thresholds: maximize F1
# 3) Tie-breaker: maximize Balanced Accuracy
# ------------------------------------------------------------

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

warnings.filterwarnings("ignore")

# ==========================================
# PATHS
# ==========================================
BASE = Path(__file__).resolve().parents[1]

PRED_FILE = (
    BASE / "results" / "evaluation_extratrees_fs_mlp" /
    "predictions" / "(1)_predictions_MLP.csv"
)

OUT_DIR = (
    BASE / "results" / "evaluation_extratrees_fs_mlp" /
    "threshold_tuning"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# CONFIG
# ==========================================
THRESHOLDS = np.arange(0.01, 1.00, 0.01)
MIN_RECALL_CONSTRAINT = 0.90

# ==========================================
# LOAD
# ==========================================
print("Starting Constrained Threshold Tuning (Pipeline: ExtraTrees FS MLP)...")
print(f"📂 Reading: {PRED_FILE}")
print(
    f"🎯 Strategy: Recall >= {MIN_RECALL_CONSTRAINT*100:.0f}% -> Maximize F1 -> Maximize Balanced Acc")
print("=" * 80)

if not PRED_FILE.exists():
    raise FileNotFoundError(f"Prediction file not found: {PRED_FILE}")

df = pd.read_csv(PRED_FILE)

if "y_true" not in df.columns or "y_prob" not in df.columns:
    raise ValueError(
        "Missing required columns y_true/y_prob in predictions file.")

y_true = df["y_true"].values
y_prob = df["y_prob"].values

if len(np.unique(y_true)) < 2:
    raise ValueError(
        "Only one class present in y_true; cannot tune threshold.")

# ==========================================
# EVALUATE THRESHOLDS
# ==========================================
rows = []

for th in THRESHOLDS:
    y_pred = (y_prob >= th).astype(int)

    rows.append({
        "Threshold": float(th),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    })

cand_df = pd.DataFrame(rows)

# Constraint filtering
valid = cand_df[cand_df["Recall"] >= MIN_RECALL_CONSTRAINT]

if not valid.empty:
    best_row = valid.sort_values(
        by=["F1", "Balanced_Acc"],
        ascending=[False, False]
    ).iloc[0]
    note = "Constraint Met ✅"
else:
    best_row = cand_df.sort_values(
        by=["F1", "Balanced_Acc"],
        ascending=[False, False]
    ).iloc[0]
    note = f"Constraint Failed (Max Recall: {cand_df['Recall'].max():.2f}) ⚠️"

# Default comparison at 0.50
y_pred_def = (y_prob >= 0.50).astype(int)
def_f1 = f1_score(y_true, y_pred_def, zero_division=0)

best_th = float(best_row["Threshold"])

print(
    f"✅ Best Th: {best_th:.2f} | F1: {best_row['F1']:.4f} | Recall: {best_row['Recall']:.4f} | {note}")
print("=" * 80)

# ==========================================
# SAVE OUTPUTS
# ==========================================
best_df = pd.DataFrame([{
    "Model": "MLP",
    "Best_Threshold": round(best_th, 2),
    "Strategy_Note": note,
    "Best_F1": round(float(best_row["F1"]), 4),
    "Best_Recall": round(float(best_row["Recall"]), 4),
    "Best_Bal_Acc": round(float(best_row["Balanced_Acc"]), 4),
    "Best_Precision": round(float(best_row["Precision"]), 4),
    "Default_F1_0.50": round(float(def_f1), 4),
    "Constraint_Target": f"Recall>={MIN_RECALL_CONSTRAINT}"
}])

best_out = OUT_DIR / "(2)_best_thresholds_constrained_mlp.csv"
detailed_out = OUT_DIR / "(2)_all_thresholds_detailed_mlp.csv"

best_df.to_csv(best_out, index=False)
cand_df.to_csv(detailed_out, index=False)

print(f"✅ Saved best thresholds: {best_out}")
print(f"✅ Saved detailed thresholds: {detailed_out}")

print("\nBest threshold row:")
print(best_df)
