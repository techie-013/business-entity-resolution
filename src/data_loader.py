from pathlib import Path

import pandas as pd


def load_train(data_dir):
    data_dir = Path(data_dir)

    source1 = pd.read_csv(data_dir / "train_source1.tsv", sep="\t")
    source2 = pd.read_csv(data_dir / "train_source2.tsv", sep="\t")
    source3 = pd.read_csv(data_dir / "train_source3.tsv", sep="\t")
    ground_truth = pd.read_csv(
        data_dir / "train_ground_truth.tsv",
        sep="\t"
    )

    return source1, source2, source3, ground_truth


def load_test(data_dir):
    data_dir = Path(data_dir)

    source1 = pd.read_csv(data_dir / "test_source1.tsv", sep="\t")
    source2 = pd.read_csv(data_dir / "test_source2.tsv", sep="\t")
    source3 = pd.read_csv(data_dir / "test_source3.tsv", sep="\t")

    return source1, source2, source3


def parse_ground_truth(ground_truth):
    truth = {}

    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids) or not str(matched_ids).strip():
            truth[s1_id] = set()
            continue

        truth[s1_id] = {
            entity_id.strip()
            for entity_id in str(matched_ids).split(",")
            if entity_id.strip()
        }

    return truth


if __name__ == "__main__":
    data_dir = Path("dataset/train")

    source1, source2, source3, ground_truth = load_train(data_dir)

    print("Training data loaded successfully.")
    print("Source 1:", source1.shape)
    print("Source 2:", source2.shape)
    print("Source 3:", source3.shape)
    print("Ground truth:", ground_truth.shape)

    truth = parse_ground_truth(ground_truth)

    print("Parsed ground-truth entries:", len(truth))