# 09_export_feature_matrix_lasso_mlp.py
# ------------------------------------------------------------
# Export Feature Selection Matrix for: LASSO FS + MLP
# Trains on FULL external_train_samples.csv using:
# - best MLP hyperparams: results/tuning_mlp/05a_best_params_no_fs_mlp.json
# - best LASSO C:         results/evaluation_lasso_fs_mlp/lasso_tuning_mlp/best_C_mlp.json
#
# Outputs:
# - results/feature_matrices/feature_matrix_lasso_fs_mlp.csv
# - results/feature_matrices/selected_features_lasso_fs_mlp.txt
# ------------------------------------------------------------

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "results" / "external_train_samples.csv"

BEST_MLP_PARAMS_PATH = BASE / "results" / \
    "tuning_mlp" / "05a_best_params_no_fs_mlp.json"
BEST_C_PATH = BASE / "results" / "evaluation_lasso_fs_mlp" / \
    "lasso_tuning_mlp" / "best_C_mlp.json"

OUT_DIR = BASE / "results" / "feature_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "feature_matrix_lasso_fs_mlp.csv"
OUT_TXT = OUT_DIR / "selected_features_lasso_fs_mlp.txt"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_clean_Xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["status"].astype(int).to_numpy()

    # keep ONLY feature columns
    X = df.drop(columns=["status"], errors="ignore")
    drop_cols = [c for c in ["name", "subject_id",
                             "Unnamed: 0"] if c in X.columns]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    return X, y


def build_pipeline(best_C: float, best_mlp_params: dict):
    lasso_est = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        C=best_C
    )

    fs = SelectFromModel(estimator=lasso_est, threshold="mean")

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
        ("mlp", mlp),
    ])

    pipe.set_params(**best_mlp_params)
    return pipe


def main():
    print("📥 Loading data and configs (LASSO FS MLP feature export)...")

    X_train, y_train = load_clean_Xy(TRAIN_PATH)

    best_mlp_data = load_json(BEST_MLP_PARAMS_PATH)
    best_mlp_params = best_mlp_data.get("best_params", None)
    if best_mlp_params is None:
        raise ValueError(
            f"Could not find 'best_params' in {BEST_MLP_PARAMS_PATH}")

    best_c_data = load_json(BEST_C_PATH)
    best_C = float(best_c_data.get("best_C_mlp", {}).get("best_C", None))
    if best_C is None:
        raise ValueError(f"Could not find best_C in {BEST_C_PATH}")

    print(f"✅ Train shape: {X_train.shape}")
    print(f"✅ best_C (LASSO selector): {best_C}")

    # Train full pipeline
    model = build_pipeline(best_C, best_mlp_params)
    model.fit(X_train, y_train)

    selector = model.named_steps["lasso"]
    support = selector.get_support()
    feature_names = X_train.columns.tolist()

    selected = [f for f, keep in zip(feature_names, support) if keep]
    n_selected = int(np.sum(support))

    print(f"✅ Selected features: {n_selected} / {len(feature_names)}")
    print("✅ Saving feature matrix...")

    # Feature Selection Matrix: rows=features, col=MLP
    mat_df = pd.DataFrame({
        "Feature": feature_names,
        "MLP": support.astype(int)
    })

    mat_df.to_csv(OUT_CSV, index=False)

    # Also save readable list
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"best_C = {best_C}\n")
        f.write(f"n_selected = {n_selected}\n\n")
        for feat in selected:
            f.write(f"{feat}\n")

    print(f"✅ Saved: {OUT_CSV}")
    print(f"✅ Saved: {OUT_TXT}")
    print("\nSelected features:")
    for feat in selected:
        print(" -", feat)


if __name__ == "__main__":
    main()
