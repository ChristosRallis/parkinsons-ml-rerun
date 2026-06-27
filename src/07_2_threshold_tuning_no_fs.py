import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. Setup Paths
# ==========================================
BASE = Path(__file__).resolve().parents[1]
# Προσοχή: Στοχεύουμε στον φάκελο no_fs
PREDS_DIR = BASE / "results" / "evaluation_no_fs" / "predictions"
TUNING_OUT_DIR = BASE / "results" / "evaluation_no_fs" / "threshold_tuning"

TUNING_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Configuration (CONSTRAINED OPTIMIZATION)
# ==========================================
THRESHOLDS = np.arange(0.01, 1.00, 0.01)

# Στόχοι:
# 1. Ασφάλεια: Recall >= MIN_RECALL_CONSTRAINT
# 2. Βελτιστοποίηση: Maximize F1 Score
MIN_RECALL_CONSTRAINT = 0.90  # Θέλουμε τουλάχιστον 90% ευαισθησία

# ==========================================
# 3. Tuning Loop
# ==========================================
print("Starting Constrained Threshold Tuning (Pipeline: no_fs)...")
print(f"📂 Reading from: {PREDS_DIR}")
print(f"🎯 Strategy: Filter Recall >= {MIN_RECALL_CONSTRAINT*100}% -> Maximize F1 -> Maximize Balanced Acc")
print("=" * 70)

summary_rows = []
detailed_rows = []

# Εύρεση αρχείων
pred_files = sorted(list(PREDS_DIR.glob("(1)_predictions_*.csv")))
if not pred_files:
    pred_files = sorted(list(PREDS_DIR.glob("predictions_*.csv")))

if not pred_files:
    print("❌ No prediction files found!")
    exit()

for p_file in pred_files:
    model_name = p_file.stem.replace("(1)_predictions_", "").replace("predictions_", "")
    print(f"🔍 Tuning {model_name}...")
    
    df = pd.read_csv(p_file)
    
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        continue
    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    
    if len(np.unique(y_true)) < 2:
        print(f"⚠️  Skipping {model_name}: Only one class present.")
        continue
    
    # Λίστα για να μαζέψουμε όλα τα metrics για κάθε threshold
    candidates = []

    # --- Loop over thresholds ---
    for th in THRESHOLDS:
        y_pred = (y_prob >= th).astype(int)
        
        metrics = {
            "Threshold": th,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)
        }
        candidates.append(metrics)
        
        # Αποθήκευση για το detailed report
        detailed_rows.append({
            "Model": model_name,
            **metrics
        })

    # Μετατροπή σε DataFrame για εύκολο φιλτράρισμα
    cand_df = pd.DataFrame(candidates)

    # --- Βήμα 1: Φιλτράρισμα (Constraint: Recall >= 90%) ---
    valid_candidates = cand_df[cand_df["Recall"] >= MIN_RECALL_CONSTRAINT]

    # --- Βήμα 2: Επιλογή ---
    if not valid_candidates.empty:
        # Αν βρήκαμε thresholds που περνάνε το τεστ, διαλέγουμε το καλύτερο F1
        # Σε ισοπαλία F1, κοιτάμε Balanced_Acc
        best_row = valid_candidates.sort_values(by=["F1", "Balanced_Acc"], ascending=[False, False]).iloc[0]
        note = "Constraint Met ✅"
    else:
        # FALLBACK: Αν κανένα δεν πιάνει 90% Recall, παίρνουμε απλά το καλύτερο F1 από όλα
        best_row = cand_df.sort_values(by=["F1", "Balanced_Acc"], ascending=[False, False]).iloc[0]
        note = f"Constraint Failed (Max Recall: {cand_df['Recall'].max():.2f}) ⚠️"

    # --- Υπολογισμός Default (0.50) για σύγκριση ---
    y_pred_def = (y_prob >= 0.50).astype(int)
    def_f1 = f1_score(y_true, y_pred_def, zero_division=0)
    
    best_th = best_row["Threshold"]
    best_f1 = best_row["F1"]
    
    print(f"   ✅ Best Th: {best_th:.2f} | F1: {best_f1:.4f} | Recall: {best_row['Recall']:.4f} | {note}")

    summary_rows.append({
        "Model": model_name,
        "Best_Threshold": round(best_th, 2),
        "Strategy_Note": note,
        "Best_F1": round(best_f1, 4),
        "Best_Recall": round(best_row['Recall'], 4),
        "Best_Bal_Acc": round(best_row['Balanced_Acc'], 4),
        "Default_F1_0.50": round(def_f1, 4),
        "Constraint_Target": f"Recall>={MIN_RECALL_CONSTRAINT}"
    })

print("=" * 70)

# Save Results
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(by="Best_F1", ascending=False)

summary_out = TUNING_OUT_DIR / "(2)_best_thresholds_constrained.csv"
summary_df.to_csv(summary_out, index=False)
print(f"✅ Saved Constrained Thresholds: {summary_out}")

detailed_df = pd.DataFrame(detailed_rows)
detailed_out = TUNING_OUT_DIR / "(2)_all_thresholds_detailed.csv"
detailed_df.to_csv(detailed_out, index=False)
print(f"✅ Saved Detailed Data: {detailed_out}")

print("\nTop Models (Constrained Optimization):")
print(summary_df[["Model", "Best_Threshold", "Best_F1", "Best_Recall", "Strategy_Note"]])