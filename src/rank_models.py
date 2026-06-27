from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import re

# -----------------------------
# Helpers
# -----------------------------


def normalize_model_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("XGBoostoost", "XGBoost")  # fix typo if exists

    mapping = {
        "LogisticRegression": "Logistic Regression",
        "LinearSVM": "Linear SVM",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "MLP": "MLP",
    }
    return mapping.get(name, name)


def load_final_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["auc", "roc_auc", "roc-auc"]:
            colmap[c] = "roc_auc"
        elif lc == "accuracy":
            colmap[c] = "accuracy"
        elif lc == "precision":
            colmap[c] = "precision"
        elif lc == "recall":
            colmap[c] = "recall"
        elif lc in ["f1", "f1-score", "f1_score", "f1score"]:
            colmap[c] = "f1"
        elif lc in ["threshold_used", "threshold"]:
            colmap[c] = "threshold"
        elif lc == "model":
            colmap[c] = "model"

    df = df.rename(columns=colmap)
    df["model"] = df["model"].apply(normalize_model_name)

    keep = ["model", "accuracy", "precision",
            "recall", "f1", "roc_auc", "threshold"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan

    return df[keep]


def find_summary_csv(metrics_dir: Path) -> Optional[Path]:
    cands = sorted(metrics_dir.glob("*metrics_summary.csv"))
    if not cands:
        cands = sorted(metrics_dir.rglob("*metrics_summary.csv"))
    return cands[0] if cands else None


def parse_cv_summary(path: Path) -> pd.DataFrame:
    """
    Handles two formats:
    (A) 'mean/std as first row' + table with 'Model' row
    (B) MLP-only: 2 rows (mean, std) with columns Accuracy/Precision/Recall/F1/ROC_AUC
    """
    df = pd.read_csv(path)

    # (B) MLP-only format
    if df.shape[0] == 2 and str(df.iloc[0, 0]).strip().lower() == "mean" and str(df.iloc[1, 0]).strip().lower() == "std":
        # columns are like Accuracy, Precision, Recall, F1, ROC_AUC
        f1_std = float(df.loc[df.iloc[:, 0].str.lower()
                       == "std", "F1"].values[0])
        auc_std = float(df.loc[df.iloc[:, 0].str.lower()
                        == "std", "ROC_AUC"].values[0])
        return pd.DataFrame({
            "model": ["MLP"],
            "cv_f1_std": [f1_std],
            "cv_roc_auc_std": [auc_std],
        })

    # (A) main format
    first_col = df.columns[0]
    model_row_idx = None
    for i, val in enumerate(df[first_col].astype(str).tolist()):
        if val.strip() == "Model":
            model_row_idx = i
            break
    if model_row_idx is None:
        return pd.DataFrame(columns=["model", "cv_f1_std", "cv_roc_auc_std"])

    header_stats = df.iloc[0]  # mean/std labels
    data = df.iloc[model_row_idx + 1:].copy()
    data = data.rename(columns={first_col: "model"})
    data["model"] = data["model"].apply(normalize_model_name)

    metric_stat = {}
    for c in df.columns[1:]:
        stat = str(header_stats[c]).strip().lower()
        metric = re.sub(r"\.\d+$", "", c).strip()
        metric = metric.replace("ROC_AUC", "roc_auc").replace(
            "ROC-AUC", "roc_auc")
        metric = metric.replace("F1", "f1").replace("Accuracy", "accuracy").replace(
            "Precision", "precision").replace("Recall", "recall")
        metric = metric.lower()
        metric_stat[c] = (metric, stat)

    out = data[["model"]].copy()
    out["cv_f1_std"] = np.nan
    out["cv_roc_auc_std"] = np.nan

    for c, (metric, stat) in metric_stat.items():
        if metric == "f1" and stat == "std":
            out["cv_f1_std"] = pd.to_numeric(data[c], errors="coerce")
        if metric == "roc_auc" and stat == "std":
            out["cv_roc_auc_std"] = pd.to_numeric(data[c], errors="coerce")

    return out


def load_feature_counts(matrix_path: Path) -> dict:
    """
    Reads feature_selection_matrix_*.csv and returns counts per model.
    Handles weird leading empty column (like ',' at start of header).
    """
    mat = pd.read_csv(matrix_path)

    # Drop first unnamed column if it exists (e.g. feature names)
    if mat.columns[0].lower().startswith("unnamed") or mat.columns[0] == "":
        mat = mat.drop(columns=[mat.columns[0]])

    # Candidate model columns (exclude helper columns)
    exclude = {"Selected_By_Count", "selected_by_count", "Feature", "feature"}
    model_cols = [c for c in mat.columns if c not in exclude]

    counts = {}
    for c in model_cols:
        counts[normalize_model_name(c)] = int(
            pd.to_numeric(mat[c], errors="coerce").fillna(0).sum())
    return counts

# -----------------------------
# Main
# -----------------------------


def rank_all_models(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_root = project_root / "results"

    pipelines = {
        "no_fs": {
            "eval_dir": results_root / "evaluation_no_fs",
            "eval_dir_mlp": results_root / "evaluation_no_fs_mlp",
            "feature_matrix": None,
            "feature_matrix_mlp": None,
        },
        "t_test": {
            "eval_dir": results_root / "evaluation_t-test_fs",
            "eval_dir_mlp": results_root / "evaluation_t-test_fs_mlp",
            "feature_matrix": results_root / "feature_matrices" / "feature_selection_matrix_ttest.csv",
            "feature_matrix_mlp": None,
        },
        "lasso": {
            "eval_dir": results_root / "evaluation_lasso_fs",
            "eval_dir_mlp": results_root / "evaluation_lasso_fs_mlp",
            "feature_matrix": results_root / "feature_matrices" / "feature_selection_matrix_lasso.csv",
            "feature_matrix_mlp": results_root / "feature_matrices" / "feature_matrix_lasso_fs_mlp.csv",
        },
        "extratrees": {
            "eval_dir": results_root / "evaluation_extratrees_fs",
            "eval_dir_mlp": results_root / "evaluation_extratrees_fs_mlp",
            "feature_matrix": results_root / "feature_matrices" / "feature_selection_matrix_extratrees.csv",
            "feature_matrix_mlp": None,
        },
    }

    # Feature counts
    feature_counts = {}
    for pipe, info in pipelines.items():
        counts = {}
        if info["feature_matrix"] and info["feature_matrix"].exists():
            counts.update(load_feature_counts(info["feature_matrix"]))
        if info["feature_matrix_mlp"] and info["feature_matrix_mlp"].exists():
            m = pd.read_csv(info["feature_matrix_mlp"])
            if "MLP" in m.columns:
                counts["MLP"] = int(pd.to_numeric(
                    m["MLP"], errors="coerce").fillna(0).sum())
        feature_counts[pipe] = counts

    # Collect test metrics + CV std
    test_rows = []
    cv_rows = []

    for pipe, info in pipelines.items():
        # final test metrics (main + mlp)
        test_main = load_final_metrics(
            info["eval_dir"] / "final_test_results" / "final_test_metrics.csv")
        test_main["pipeline"] = pipe
        test_rows.append(test_main)

        test_mlp = load_final_metrics(
            info["eval_dir_mlp"] / "final_test_results" / "final_test_metrics.csv")
        test_mlp["pipeline"] = pipe
        test_rows.append(test_mlp)

        # cv summaries (main + mlp)
        s1 = find_summary_csv(info["eval_dir"] / "metrics")
        if s1:
            cv1 = parse_cv_summary(s1)
            cv1["pipeline"] = pipe
            cv_rows.append(cv1)

        s2 = find_summary_csv(info["eval_dir_mlp"] / "metrics")
        if s2:
            cv2 = parse_cv_summary(s2)
            cv2["pipeline"] = pipe
            cv_rows.append(cv2)

    test_df = pd.concat(test_rows, ignore_index=True)
    cv_df = pd.concat(cv_rows, ignore_index=True)

    df = test_df.merge(cv_df, on=["pipeline", "model"], how="left")

    # n_features rule: from feature matrix, else 22
    def get_n_features(row):
        pipe = row["pipeline"]
        model = row["model"]
        if model in feature_counts.get(pipe, {}):
            return feature_counts[pipe][model]
        return 22

    df["n_features"] = df.apply(get_n_features, axis=1).astype(int)

    # ranking rule (hierarchical):
    # 1) maximize F1
    # 2) maximize ROC-AUC
    # 3) minimize n_features
    # 4) minimize CV std (worst of f1_std, auc_std)
    df["cv_std_for_rank"] = df[["cv_f1_std", "cv_roc_auc_std"]].max(axis=1)

    ranked = df.sort_values(
        by=["f1", "roc_auc", "n_features", "cv_std_for_rank"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)

    best = ranked.iloc[[0]].copy()
    top5 = ranked.head(5).copy()
    return best, ranked, top5


if __name__ == "__main__":
    project_root = Path(".").resolve()
    best, ranked, top5 = rank_all_models(project_root)

    out_dir = project_root / "results" / "model_ranking"
    out_dir.mkdir(parents=True, exist_ok=True)

    best.to_csv(out_dir / "best_overall_model.csv", index=False)
    ranked.to_csv(out_dir / "ranked_models_all_20.csv", index=False)
    top5.to_csv(out_dir / "top5_models.csv", index=False)

    print("Saved:")
    print(" -", out_dir / "best_overall_model.csv")
    print(" -", out_dir / "ranked_models_all_20.csv")
    print(" -", out_dir / "top5_models.csv")
