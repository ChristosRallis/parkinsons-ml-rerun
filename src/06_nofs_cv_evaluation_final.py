import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

# Φιλτράρουμε τα warnings για πιο καθαρό output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Setup Paths & Directories (SYNCED)
# ==========================================
BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
FOLDS_PATH = BASE / "results" / "cv_folds_samples.json"
PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"

#  Output Directories (σωστό project path)
EVAL_DIR = BASE / "results" / "evaluation_no_fs"
METRICS_DIR = EVAL_DIR / "metrics"
PREDS_DIR = EVAL_DIR / "predictions"

for d in [EVAL_DIR, METRICS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ==========================================
# 2. Helper Functions (Safety Checks)
# ==========================================


def clean_params(model_type, params):
    """
    Καθαρίζει τις παραμέτρους από prefixes (clf__) αν το μοντέλο δεν είναι Pipeline,
    και διορθώνει τυχόν strings "None" σε πραγματικό None object.
    """
    cleaned = {}
    for k, v in params.items():
        # 1. Διόρθωση Τύπων (Fix "None" string -> None object)
        if v == "None":
            v = None

        # 2. Αφαίρεση prefix 'clf__' για RF/XGB που δεν είναι σε Pipeline
        if model_type in ["RandomForest", "XGBoost"]:
            if k.startswith("clf__"):
                new_key = k.replace("clf__", "")
                cleaned[new_key] = v
            else:
                cleaned[k] = v
        else:
            # Για SVM/Logistic που είναι Pipelines, θέλουμε το prefix
            cleaned[k] = v

    return cleaned


def safe_roc_auc(y_true, y_prob):
    """
    Υπολογίζει το ROC AUC με ασφάλεια.
    Αν το y_true έχει μόνο μία κλάση (π.χ. μόνο 0 ή μόνο 1), επιστρέφει NaN.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return np.nan


def get_model(name, params):
    """
    Αρχικοποιεί το μοντέλο με τις παραμέτρους.
    """
    # Καθαρίζουμε τις παραμέτρους πριν τις δώσουμε
    safe_params = clean_params(name, params)

    if name == "LogisticRegression":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000,
             solver="liblinear", random_state=RANDOM_STATE))
        ])
    elif name == "LinearSVM":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))
        ])
    elif name == "RandomForest":
        model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif name == "XGBoost":
        # Αφαιρέσαμε το use_label_encoder=False που είναι deprecated
        model = XGBClassifier(eval_metric='logloss',
                              random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {name}")

    # Ανάθεση παραμέτρων
    try:
        model.set_params(**safe_params)
    except Exception as e:
        print(f"⚠️ Warning: Could not set params for {name}. Error: {e}")

    return model


# ==========================================
# 3. Load Data
# ==========================================
print("Loading Data and Configuration...")

df = pd.read_csv(TRAIN_PATH)
df = df.reset_index(drop=True)  # Σημαντικό για να ταιριάζουν τα indices

X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()
names_arr = df["name"].values

# Load JSONs
with open(FOLDS_PATH, "r", encoding="utf-8") as f:
    folds_data = json.load(f)

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    best_params_data = json.load(f)

print(
    f"Loaded {len(df)} samples. Best params loaded for: {list(best_params_data.keys())}")

# ==========================================
# 4. Evaluation Loop
# ==========================================
fold_metrics = []
all_predictions = []

print("\nStarting Evaluation per Fold...")
print("=" * 60)

for fold_idx, fold_info in enumerate(folds_data["folds"]):
    fold_num = fold_idx + 1

    # 1. Αντιστοίχιση ονομάτων σε indices (Απόλυτη Ακρίβεια)
    train_names_set = set(fold_info["train_samples"])
    val_names_set = set(fold_info["val_samples"])

    train_idx = df.index[df["name"].isin(train_names_set)].tolist()
    val_idx = df.index[df["name"].isin(val_names_set)].tolist()

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    val_names_current = names_arr[val_idx]

    print(f"🔄 Fold {fold_num}: Train={len(train_idx)}, Val={len(val_idx)}")

    # 2. Iterate Models
    for model_name in ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]:
        # Load raw params
        raw_params = best_params_data[model_name]["best_params"]

        # Build & Train (με safety checks)
        try:
            model = get_model(model_name, raw_params)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_val)

            # Probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                # Fallback (αν και έχουμε probability=True στο SVM)
                y_prob = model.decision_function(X_val)

            # 3. Calculate Metrics (με safe ROC)
            auc_val = safe_roc_auc(y_val, y_prob)

            scores = {
                "Fold": fold_num,
                "Model": model_name,
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred, zero_division=0),
                "Recall": recall_score(y_val, y_pred, zero_division=0),
                "F1": f1_score(y_val, y_pred, zero_division=0),
                "ROC_AUC": auc_val
            }
            fold_metrics.append(scores)

            # 4. Store Predictions
            preds_df = pd.DataFrame({
                "sample_name": val_names_current,
                "y_true": y_val,
                "y_prob": y_prob,
                "fold": fold_num,
                "model": model_name
            })
            all_predictions.append(preds_df)

        except Exception as e:
            print(f"❌ Error in Fold {fold_num} - Model {model_name}: {e}")

print("=" * 60)

# ==========================================
# 5. Save Results (With Prefix (1)_)
# ==========================================

# A. Metrics per Fold
metrics_df = pd.DataFrame(fold_metrics)
# ΠΡΟΣΘΗΚΗ ΠΡΟΘΕΜΑΤΟΣ (1)_
metrics_file = METRICS_DIR / "(1)_no_fs_metrics_per_fold.csv"
metrics_df.to_csv(metrics_file, index=False)
print(f"✅ Saved metrics per fold: {metrics_file}")

# B. Summary (Mean ± Std)
summary = metrics_df.groupby("Model")[
    ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].agg(["mean", "std"])
# ΠΡΟΣΘΗΚΗ ΠΡΟΘΕΜΑΤΟΣ (1)_
summary_file = METRICS_DIR / "(1)_no_fs_metrics_summary.csv"
summary.to_csv(summary_file)
print(f"✅ Saved summary metrics: {summary_file}")

# C. Raw Predictions (Split by Model)
if all_predictions:
    total_preds_df = pd.concat(all_predictions, ignore_index=True)

    for model_name in total_preds_df["model"].unique():
        model_preds = total_preds_df[total_preds_df["model"]
                                     == model_name].copy()
        model_preds = model_preds.drop(columns=["model"])

        # ΠΡΟΣΘΗΚΗ ΠΡΟΘΕΜΑΤΟΣ (1)_
        pred_file = PREDS_DIR / f"(1)_predictions_{model_name}.csv"
        model_preds.to_csv(pred_file, index=False)
        print(f"✅ Saved raw predictions for {model_name}")

print("\nEvaluation Cycle Complete.")
print("Summary Preview:")
print(summary)
