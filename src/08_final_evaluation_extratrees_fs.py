# 08_final_evaluation_extratrees_fs.py
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_JOBS = -1
MODELS = ["LogisticRegression", "LinearSVM", "RandomForest", "XGBoost"]

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"
TEST_PATH = BASE / "results" / "external_test_samples.csv"

BEST_PARAMS_PATH = BASE / "results" / "tuning" / "05a_best_params_no_fs.json"
BEST_K_PATH = BASE / "results" / "evaluation_extratrees_fs" / \
    "extratrees_tuning" / "best_k_per_model.json"
THRESHOLDS_PATH = BASE / "results" / "evaluation_extratrees_fs" / \
    "threshold_tuning" / "(2)_best_thresholds_constrained.csv"

OUT_DIR = BASE / "results" / "evaluation_extratrees_fs" / "final_test_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_none_strings(params: dict) -> dict:
    return {k: (None if v == "None" else v) for k, v in params.items()}


def to_pipeline_clf_params(params: dict) -> dict:
    params = clean_none_strings(params)
    out = {}
    for k, v in params.items():
        out[k if k.startswith("clf__") else f"clf__{k}"] = v
    return out


def load_clean_Xy(csv_path: Path, label: str):
    df = pd.read_csv(csv_path)
    y = df["status"].astype(int).to_numpy()
    X = df.drop(columns=["status"], errors="ignore")
    drop_cols = [c for c in ["name", "subject_id",
                             "Unnamed: 0"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    names = df["name"].values if "name" in df.columns else np.arange(len(df))
    return X, y, names


def build_pipeline(model_name: str, best_k: int, clf_params: dict):
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    fs = SelectFromModel(estimator=et, threshold=-
                         float("inf"), max_features=int(best_k))

    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)
    elif model_name == "LinearSVM":
        clf = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_name == "XGBoost":
        clf = XGBClassifier(eval_metric="logloss",
                            random_state=RANDOM_STATE, n_jobs=N_JOBS)
    else:
        raise ValueError(model_name)

    pipe = Pipeline([("scaler", StandardScaler()), ("fs", fs), ("clf", clf)])
    pipe.set_params(**clf_params)
    return pipe


print("🚀 Starting FINAL evaluation on EXTERNAL TEST (Pipeline: ExtraTrees FS)...")

best_params_data = load_json(BEST_PARAMS_PATH)
best_k_data = load_json(BEST_K_PATH)["best_k_per_model"]
th_df = pd.read_csv(THRESHOLDS_PATH)
best_thresholds = dict(zip(th_df["Model"], th_df["Best_Threshold"]))

X_train, y_train, train_names = load_clean_Xy(TRAIN_PATH, "TRAIN")
X_test, y_test, test_names = load_clean_Xy(TEST_PATH, "TEST")
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"✅ Data Loaded & Aligned. Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Loaded Thresholds: {best_thresholds}")

plt.figure(figsize=(10, 8))

results = []
all_preds_rows = []

for model_name in MODELS:
    print(f"🔄 Processing {model_name}...")

    k = int(best_k_data[model_name]["best_k"])
    clf_params = to_pipeline_clf_params(
        best_params_data[model_name]["best_params"])

    model = build_pipeline(model_name, k, clf_params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(
        model, "predict_proba") else model.decision_function(X_test)

    th = float(best_thresholds.get(model_name, 0.5))
    y_pred = (y_prob >= th).astype(int)

    auc = roc_auc_score(y_test, y_prob) if len(
        np.unique(y_test)) == 2 else np.nan
    results.append({
        "Model": model_name,
        "k_used": k,
        "Threshold_Used": th,
        "AUC": auc,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    })

    for sn, yt, yp, ypr in zip(test_names, y_test, y_prob, y_pred):
        all_preds_rows.append({"sample_name": sn, "Model": model_name, "y_true": int(
            yt), "y_prob": float(yp), "y_pred": int(ypr)})

    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} (Th={th:.2f})\nTest Set CM")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cm_{model_name}.png", dpi=300)
    plt.close(fig)

    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Final ROC Curves on External Test Set (ExtraTrees FS)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(OUT_DIR / "roc_curves_final.png", dpi=300)
plt.close()

final_df = pd.DataFrame(results).sort_values(
    by=["Recall", "F1"], ascending=False)
final_df.to_csv(OUT_DIR / "final_test_metrics.csv", index=False)
pd.DataFrame(all_preds_rows).to_csv(
    OUT_DIR / "final_test_predictions.csv", index=False)

print("\n" + "=" * 70)
print("🏁 FINAL EXTERNAL TEST RESULTS (ExtraTrees FS)")
print("=" * 70)
print(final_df[["Model", "k_used", "Threshold_Used", "F1", "Recall", "AUC"]])
print("=" * 70)
print(f"✅ Saved metrics: {OUT_DIR / 'final_test_metrics.csv'}")
print(f"✅ Saved plots in: {OUT_DIR}")
