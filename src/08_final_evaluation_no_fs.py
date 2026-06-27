import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION (ΑΠΟΛΥΤΕΣ ΔΙΑΔΡΟΜΕΣ)
# ==========================================
# Οι διαδρομές που μου έδωσες ακριβώς:
TRAIN_FILE_PATH = r"C:\Users\chris\OneDrive\Υπολογιστής\parkinsons-ml\results\external_train_samples.csv"
TEST_FILE_PATH = r"C:\Users\chris\OneDrive\Υπολογιστής\parkinsons-ml\results\external_test_samples.csv"

TARGET_COL = "status"
# Αφαιρούμε name, subject_id (Leakage!) και τυχόν Unnamed indexes
DROP_COLS = ["name", "subject_id", "Unnamed: 0"]

# ==========================================
# 2. Setup Output Paths
# ==========================================
BASE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE / "results" / "evaluation_no_fs" / "final_test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds (Αν δεν βρεθεί το αρχείο, θα χρησιμοποιηθεί default 0.5)
THRESHOLDS_FILE = BASE / "results" / "evaluation_no_fs" / \
    "threshold_tuning" / "(2)_best_thresholds_constrained.csv"

# ==========================================
# 3. Load & Clean Data
# ==========================================
print("🚀 Starting Final Evaluation on TEST SET (Pipeline: no_fs)...")


def load_clean_data(path_str, label):
    path = Path(path_str)

    if not path.exists():
        print(f"❌ CRITICAL ERROR: Το αρχείο δεν βρέθηκε: {path}")
        exit()

    print(f"📥 Loading {label}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading csv: {e}")
        exit()

    # Έλεγχος Target
    if TARGET_COL not in df.columns:
        print(f"❌ Error: Η στήλη '{TARGET_COL}' λείπει από το {label}.")
        exit()

    # Διαχωρισμός
    y = df[TARGET_COL].values.ravel()
    X = df.drop(columns=[TARGET_COL], errors='ignore')

    # Αφαίρεση μη-features
    cols_to_drop = [c for c in DROP_COLS if c in X.columns]
    if cols_to_drop:
        print(f"   ✂️  Removing columns from {label}: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    return X, y


# Φόρτωση δεδομένων
X_train, y_train = load_clean_data(TRAIN_FILE_PATH, "Train Set")
X_test, y_test = load_clean_data(TEST_FILE_PATH, "Test Set")

# ⚠️ CRITICAL: Ευθυγράμμιση στηλών (Feature Alignment)
# Εξασφαλίζουμε ότι το Test έχει ΑΚΡΙΒΩΣ τις ίδιες στήλες με το Train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"✅ Data Loaded & Aligned:")
print(f"   Train Shape: {X_train.shape}")
print(f"   Test Shape:  {X_test.shape}")

# Φόρτωση Thresholds
if THRESHOLDS_FILE.exists():
    thresh_df = pd.read_csv(THRESHOLDS_FILE)
    best_thresholds = dict(
        zip(thresh_df['Model'], thresh_df['Best_Threshold']))
    print(f"✅ Loaded Thresholds: {best_thresholds}")
else:
    print("⚠️ Warning: Thresholds file not found. Using default 0.5")
    best_thresholds = {}

# ==========================================
# 4. Scaling (Global)
# ==========================================
print("⚖️  Applying Scaling (StandardScaler) to ALL data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. Define Models (FINAL TUNED PARAMS)
# ==========================================
print("⚙️  Initializing Models...")

models = {
    "LogisticRegression": LogisticRegression(
        C=1,
        penalty='l1',
        solver='liblinear',
        random_state=42
    ),
    "LinearSVM": SVC(
        kernel='linear',
        C=0.01,
        probability=True,
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        max_features='sqrt',
        max_depth=None,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=100,
        subsample=1.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
}

# ==========================================
# 6. Training & Evaluation Loop
# ==========================================
plt.figure(figsize=(10, 8))  # Για το ROC Curve
results = []

for name, model in models.items():
    print(f"🔄 Processing {name}...")

    # 1. Train (Fit)
    model.fit(X_train_scaled, y_train)

    # 2. Predict Probabilities (Safe)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        d = model.decision_function(X_test_scaled)
        y_prob = (d - d.min()) / (d.max() - d.min() + 1e-10)

    # 3. Apply Threshold
    th = best_thresholds.get(name, 0.5)
    y_pred = (y_prob >= th).astype(int)

    # 4. Metrics
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name, "Threshold_Used": th, "AUC": auc,
        "Accuracy": acc, "Recall": rec, "Precision": prec, "F1": f1
    })

    # 5. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} (Th={th:.2f})\nTest Set CM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"cm_{name}.png", dpi=300)
    plt.close()

    # 6. Add to ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

# Finalize ROC Plot
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Final ROC Curves on External Test Set")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / "roc_curves_final.png", dpi=300)
plt.close()

# Save Metrics to CSV
final_df = pd.DataFrame(results).sort_values(
    by=["Recall", "F1"], ascending=False)
final_df.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)

print("\n" + "="*60)
print("🏁 FINAL TEST SET RESULTS (External Test)")
print("="*60)
print(final_df[['Model', 'Threshold_Used', 'F1', 'Recall', 'AUC']])
print("\n" + "="*60)
print(f"✅ Results saved in: {OUTPUT_DIR}")
