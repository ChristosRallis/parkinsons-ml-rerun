# 07_threshold_tuning_lasso_fs_constrained.py
# ------------------------------------------------------------
# Threshold tuning on out-of-fold predictions produced by:
# results/evaluation_lasso_fs/predictions/(1)_predictions_*.csv
#
# Strategy:
# 1) Constraint: Recall >= MIN_RECALL_CONSTRAINT
# 2) Among valid thresholds: maximize F1
# 3) Tie-breaker: maximize Balanced Accuracy
#
# Saves:
# - best thresholds per model:
#   results/evaluation_lasso_fs/threshold_tuning/(2)_best_thresholds_constrained.csv
# - detailed metrics for all thresholds:
#   results/evaluation_lasso_fs/threshold_tuning/(2)_all_thresholds_detailed.csv
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
# 1. PATHS
# ==========================================
BASE = Path(__file__).resolve().parents[1]

PREDS_DIR = BASE / "results" / "evaluation_lasso_fs" / "predictions"
OUT_DIR = BASE / "results" / "evaluation_lasso_fs" / "threshold_tuning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CONFIG
# ==========================================
THRESHOLDS = np.arange(0.01, 1.00, 0.01)
MIN_RECALL_CONSTRAINT = 0.90

# ==========================================
# 3. LOAD FILES
# ==========================================
print("Starting Constrained Threshold Tuning (Pipeline: LASSO FS)...")
print(f"📂 Reading from: {PREDS_DIR}")
print(
    f"🎯 Strategy: Recall >= {MIN_RECALL_CONSTRAINT*100:.0f}% -> Maximize F1 -> Maximize Balanced Acc"
)
print("=" * 80)

pred_files = sorted(list(PREDS_DIR.glob("(1)_predictions_*.csv")))
if not pred_files:
    pred_files = sorted(list(PREDS_DIR.glob("predictions_*.csv")))

if not pred_files:
    raise FileNotFoundError(f"No prediction files found in: {PREDS_DIR}")

summary_rows = []
detailed_rows = []

for p_file in pred_files:
    model_name = p_file.stem.replace(
        "(1)_predictions_", "").replace("predictions_", "")
    print(f"🔍 Tuning {model_name}...")

    df = pd.read_csv(p_file)

    if "y_true" not in df.columns or "y_prob" not in df.columns:
        print(f"   ⚠️ Skipping {model_name}: missing y_true/y_prob columns.")
        continue

    y_true = df["y_true"].values
    y_prob = df["y_prob"].values

    if len(np.unique(y_true)) < 2:
        print(f"   ⚠️ Skipping {model_name}: only one class present.")
        continue

    candidates = []

    for th in THRESHOLDS:
        y_pred = (y_prob >= th).astype(int)

        metrics = {
            "Threshold": float(th),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        }
        candidates.append(metrics)

        detailed_rows.append({
            "Model": model_name,
            **metrics
        })

    cand_df = pd.DataFrame(candidates)

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
        f"   ✅ Best Th: {best_th:.2f} | F1: {best_row['F1']:.4f} | Recall: {best_row['Recall']:.4f} | {note}"
    )

    summary_rows.append({
        "Model": model_name,
        "Best_Threshold": round(best_th, 2),
        "Strategy_Note": note,
        "Best_F1": round(float(best_row["F1"]), 4),
        "Best_Recall": round(float(best_row["Recall"]), 4),
        "Best_Bal_Acc": round(float(best_row["Balanced_Acc"]), 4),
        "Best_Precision": round(float(best_row["Precision"]), 4),
        "Default_F1_0.50": round(float(def_f1), 4),
        "Constraint_Target": f"Recall>={MIN_RECALL_CONSTRAINT}"
    })

print("=" * 80)

# ==========================================
# 4. SAVE OUTPUTS
# ==========================================
summary_df = pd.DataFrame(summary_rows).sort_values(
    by="Best_F1", ascending=False)
detailed_df = pd.DataFrame(detailed_rows)

summary_out = OUT_DIR / "(2)_best_thresholds_constrained.csv"
detailed_out = OUT_DIR / "(2)_all_thresholds_detailed.csv"

summary_df.to_csv(summary_out, index=False)
detailed_df.to_csv(detailed_out, index=False)

print(f"✅ Saved best thresholds: {summary_out}")
print(f"✅ Saved detailed thresholds report: {detailed_out}")

print("\nTop Models (Constrained Thresholds):")
print(summary_df[["Model", "Best_Threshold",
      "Best_F1", "Best_Recall", "Strategy_Note"]])
