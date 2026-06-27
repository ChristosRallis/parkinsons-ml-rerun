# 08_final_evaluation_lasso_fs.py
# ------------------------------------------------------------
# FINAL EVALUATION: LASSO FEATURE SELECTION PIPELINE
# Visual Style: Matches your No-FS / t-test style for consistency.
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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
# 1. PATHS & CONFIG
# ==========================================
RANDOM_STATE = 42
MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
TEST_PATH = BASE / "results" / "external_test_samples.csv"

# Config files
BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"
BEST_C_PATH = BASE / "results" / "evaluation_lasso_fs" / \
    "lasso_tuning" / "best_C_per_model.json"
THRESHOLDS_PATH = BASE / "results" / "evaluation_lasso_fs" / \
    "threshold_tuning" / "(2)_best_thresholds_constrained.csv"

# Output directory
OUT_DIR = BASE / "results" / "evaluation_lasso_fs" / "final_test_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. HELPERS
# ==========================================


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_none_strings(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        out[k] = None if v == "None" else v
    return out


def to_pipeline_clf_params(params: dict) -> dict:
    params = clean_none_strings(params)
    out = {}
    for k, v in params.items():
        if k.startswith("clf__"):
            out[k] = v
        else:
            out[f"clf__{k}"] = v
    return out


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


def build_pipeline(model_name: str, best_C: float, clf_params: dict):
    # LASSO selector
    lasso_est = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=float(best_C),
        max_iter=5000,
        random_state=RANDOM_STATE,
        # optional: class_weight="balanced"
    )

    fs = SelectFromModel(
        estimator=lasso_est,
        threshold="mean"
    )

    # Classifier
    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)
    elif model_name == "LinearSVM":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "XGBoost":
        clf = XGBClassifier(eval_metric="logloss",
                            random_state=RANDOM_STATE, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", fs),
        ("clf", clf),
    ])

    pipe.set_params(**clf_params)
    return pipe


# ==========================================
# 3. EXECUTION
# ==========================================
print("🚀 Starting FINAL evaluation on EXTERNAL TEST (Pipeline: LASSO FS)...")

# Load configs
best_params_data = load_json(BEST_PARAMS_PATH)
best_c_data = load_json(BEST_C_PATH)
best_c_per_model = best_c_data.get("best_C_per_model", {})

th_df = pd.read_csv(THRESHOLDS_PATH)
best_thresholds = dict(zip(th_df["Model"], th_df["Best_Threshold"]))

# Load data
X_train, y_train, train_names = load_clean_Xy(TRAIN_PATH, "TRAIN")
X_test, y_test, test_names = load_clean_Xy(TEST_PATH, "TEST")

# Align columns (safety)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"✅ Data Loaded & Aligned. Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Loaded Thresholds: {best_thresholds}")

# Initialize ROC figure globally
plt.figure(figsize=(10, 8))

results = []
all_preds_rows = []

for model_name in MODELS:
    print(f"🔄 Processing {model_name}...")

    if model_name not in best_c_per_model:
        print(f"   ⚠️ Missing best_C for {model_name}. Skipping.")
        continue

    best_C = float(best_c_per_model[model_name]["best_C"])

    raw_params = best_params_data[model_name]["best_params"]
    clf_params = to_pipeline_clf_params(raw_params)

    model = build_pipeline(model_name, best_C, clf_params)
    model.fit(X_train, y_train)

    # Predict proba/score
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        d = model.decision_function(X_test)
        y_prob = (d - d.min()) / (d.max() - d.min() + 1e-12)

    # Apply tuned threshold
    th = float(best_thresholds.get(model_name, 0.5))
    y_pred = (y_prob >= th).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_prob) if len(
        np.unique(y_test)) == 2 else np.nan
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": model_name,
        "best_C_used": best_C,
        "Threshold_Used": th,
        "AUC": auc,
        "Accuracy": acc,
        "Recall": rec,
        "Precision": prec,
        "F1": f1
    })

    # Save detailed predictions
    for sn, yt, yp, ypr in zip(test_names, y_test, y_prob, y_pred):
        all_preds_rows.append({
            "sample_name": sn,
            "Model": model_name,
            "y_true": int(yt),
            "y_prob": float(yp),
            "y_pred": int(ypr)
        })

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt_cm = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} (Th={th:.2f})\nTest Set CM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cm_{model_name}.png", dpi=300)
    plt.close(plt_cm)

    # Add to ROC curve
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

# Finalize ROC plot
plt.figure(1)
plt.plot([0, 1], [0, 1], "k--", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Final ROC Curves on External Test Set (LASSO FS)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(OUT_DIR / "roc_curves_final.png", dpi=300)
plt.close()

# Save CSVs
final_df = pd.DataFrame(results).sort_values(
    by=["Recall", "F1"], ascending=False)
final_df.to_csv(OUT_DIR / "final_test_metrics.csv", index=False)

preds_df = pd.DataFrame(all_preds_rows)
preds_df.to_csv(OUT_DIR / "final_test_predictions.csv", index=False)

print("\n" + "=" * 70)
print("🏁 FINAL EXTERNAL TEST RESULTS (LASSO FS)")
print("=" * 70)
print(final_df[["Model", "best_C_used",
      "Threshold_Used", "F1", "Recall", "AUC"]])
print("=" * 70)
print(f"✅ Saved metrics: {OUT_DIR / 'final_test_metrics.csv'}")
print(f"✅ Saved plots in: {OUT_DIR}")
