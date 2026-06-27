import pandas as pd
from pathlib import Path


def create_filtered_csv():
    # Project root = ο φάκελος parkinsons-ml-rerun
    project_root = Path(__file__).resolve().parents[1]

    input_path = project_root / "results" / \
        "model_ranking" / "ranked_models_all_20.csv"
    output_path = project_root / "results" / \
        "model_ranking" / "classified_table_no_fs.csv"

    if not input_path.exists():
        print(f"Σφάλμα: Δεν βρέθηκε το αρχείο:\n{input_path}")
        print("Τρέξε πρώτα: python src/rank_models.py")
        return

    print("Διαβάζεται το ranked_models_all_20.csv...")
    df = pd.read_csv(input_path)

    if "pipeline" not in df.columns:
        print("Σφάλμα: Δεν υπάρχει στήλη 'pipeline' στο ranked_models_all_20.csv")
        return

    filtered_df = df[df["pipeline"] == "no_fs"].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)

    print("--------------------------------------------------")
    print("Επιτυχία! Δημιουργήθηκε το No-FS ranking table.")
    print(f"Κρατήθηκαν {len(filtered_df)} γραμμές από τις συνολικά {len(df)}.")
    print(f"Αποθηκεύτηκε στο:\n{output_path}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    create_filtered_csv()
