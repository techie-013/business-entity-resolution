import os
import time
import pandas as pd


from src.blocking import generate_candidates

SAMPLE_S1 = 2_000
RANDOM_STATE = 42

DATA_DIR = "dataset/train"
S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")


def load_gt():
    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt["source1_entity_id"] = (
        gt["source1_entity_id"].astype(str)
    )

    mapping = {}

    for row in gt.itertuples(index=False):
        raw = row.matched_entity_ids

        if pd.isna(raw):
            mapping[row.source1_entity_id] = set()
        else:
            mapping[row.source1_entity_id] = {
                x.strip()
                for x in str(raw).split(",")
                if x.strip()
            }

    return mapping


def main():
    t0 = time.time()

    print("=" * 72)
    print("ACTUAL LOCAL BLOCKING VALIDATOR")
    print("=" * 72)

    print("Loading S1 sample...", flush=True)

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s1["entity_id"] = s1["entity_id"].astype(str)

    s1_sample = s1.sample(
        n=min(SAMPLE_S1, len(s1)),
        random_state=RANDOM_STATE,
    ).copy()

    print(
        f"S1 sample: {len(s1_sample):,}",
        flush=True
    )

    print("Loading FULL S2...", flush=True)

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s2["entity_id"] = s2["entity_id"].astype(str)

    print(
        f"S2: {len(s2):,}",
        flush=True
    )

    print("Loading FULL S3...", flush=True)

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    s3["entity_id"] = s3["entity_id"].astype(str)

    print(
        f"S3: {len(s3):,}",
        flush=True
    )

    print("Loading ground truth...", flush=True)

    gt = load_gt()

    sample_gt = {
        sid: gt.get(sid, set())
        for sid in s1_sample["entity_id"]
        if gt.get(sid, set())
    }

    print(
        f"S1 with GT matches: {len(sample_gt):,}",
        flush=True
    )

    # This is the same production candidate cap used by
    # predict_submit.py according to the current project setup.
    CAP = 100

    print()
    print("=" * 72)
    print(f"GENERATING CANDIDATES USING ACTUAL src/blocking.py")
    print(f"max_candidates={CAP}")
    print("=" * 72)

    candidates, _, _, _ = generate_candidates(
        s1_sample,
        s2,
        s3,
        max_candidates=CAP,
    )

    hits = 0
    pair_hits = 0
    total_pairs = 0

    missed = []

    for sid, true_ids in sample_gt.items():
        total_pairs += len(true_ids)

        cand = {
            str(x)
            for x in candidates.get(str(sid), set())
        }

        found = true_ids & cand

        if found:
            hits += 1
        else:
            if len(missed) < 20:
                missed.append((sid, true_ids))

        pair_hits += len(found)

    s1_recall = (
        hits / len(sample_gt)
        if sample_gt else 0.0
    )

    pair_recall = (
        pair_hits / total_pairs
        if total_pairs else 0.0
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        f"S1/entity recall: {s1_recall:.4f} "
        f"({hits:,}/{len(sample_gt):,})"
    )

    print(
        f"Pair recall:      {pair_recall:.4f} "
        f"({pair_hits:,}/{total_pairs:,})"
    )

    print(
        f"Candidate cap:    {CAP}"
    )

    if s1_recall >= 0.80:
        print()
        print("PASS: actual local blocking is above 0.80.")
    else:
        print()
        print("FAIL: actual local blocking is still below 0.80.")

    if missed:
        print()
        print("=" * 72)
        print("FIRST MISSED S1 EXAMPLES")
        print("=" * 72)

        s1_lookup = s1_sample.set_index("entity_id")

        for sid, true_ids in missed:
            row = s1_lookup.loc[sid]

            print(
                f"S1: {sid} | {row['business_name']}"
            )
            print(
                f"GT: {next(iter(true_ids))}"
            )
            print(
                f"ADR: {row['business_address']}"
            )
            print("-" * 50)

    print()
    print(
        f"Total validation time: "
        f"{time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
