from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier  # <--- Τώρα θα δουλέψει!
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Φιλτράρισμα warnings
warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
N_SPLITS = 3
RANDOM_STATE = 42

# =========================
# Paths
# =========================
BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
OUT_DIR = BASE / "results"
OUT_DIR.mkdir(exist_ok=True)

# =========================
# Load external TRAIN set
# =========================
df = pd.read_csv(TRAIN_PATH)

# Separate features, labels, groups
X = df.drop(columns=["name", "status", "subject_id"])
y = df["status"].astype(int).to_numpy()
groups = df["subject_id"].to_numpy()

print(f"Running Sanity Check on {df.shape[0]} samples...")
print("-" * 50)

# =========================
# Models (Default Params for Sanity Check)
# =========================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            random_state=RANDOM_STATE
        ))
    ]),
    "LinearSVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="linear",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# =========================
# 3-Fold Stratified Group CV
# =========================
cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []
fold_id = 0

for train_idx, val_idx in cv.split(X, y, groups=groups):
    fold_id += 1

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    for model_name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)

        # Proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_val)
        else:
            y_proba = y_pred

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if len(np.unique(y_val)) == 2:
            auc = roc_auc_score(y_val, y_proba)
        else:
            auc = np.nan

        results.append({
            "fold": fold_id,
            "model": model_name,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc
        })

# =========================
# Save & Print
# =========================
results_df = pd.DataFrame(results)
summary = results_df.groupby("model")[["accuracy", "f1", "roc_auc"]].mean()

print("\nSanity Check Complete! Average Scores:")
print(summary)
print("-" * 50)
