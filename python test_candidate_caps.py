import os
import time
import pandas as pd

from src.data_loader import load_train, parse_ground_truth
from src.blocking import generate_candidates

S1_SAMPLE_SIZE = 50_000
SAMPLE_S2_S3_SIZE = 500_000
RANDOM_STATE = 42

DATA_DIR = "dataset/train"


def build_target_sample(full_df, true_ids, size, seed):
    df = full_df.copy()
    df["entity_id"] = df["entity_id"].astype(str)

    true_ids = {str(x) for x in true_ids}

    true_rows = df[df["entity_id"].isin(true_ids)]

    remaining = max(0, size - len(true_rows))

    if remaining == 0:
        return true_rows.copy()

    decoys = df[
        ~df["entity_id"].isin(true_ids)
    ].sample(
        n=remaining,
        random_state=seed
    )

    return pd.concat(
        [true_rows, decoys],
        ignore_index=True
    )


def main():
    t0 = time.time()

    print("=" * 70)
    print("CANDIDATE CAP TEST")
    print("=" * 70)

    s1, s2, s3, gt = load_train(DATA_DIR)

    s1_sample = s1.sample(
        n=min(S1_SAMPLE_SIZE, len(s1)),
        random_state=RANDOM_STATE
    ).copy()

    s1_sample["entity_id"] = (
        s1_sample["entity_id"].astype(str)
    )

    gt_map = parse_ground_truth(gt)

    sample_gt = {
        sid: gt_map.get(sid, set())
        for sid in s1_sample["entity_id"]
        if gt_map.get(sid, set())
    }

    true_s2 = set()
    true_s3 = set()

    for tids in sample_gt.values():
        for tid in tids:
            tid = str(tid)
            if tid.startswith("S2-"):
                true_s2.add(tid)
            elif tid.startswith("S3-"):
                true_s3.add(tid)

    s2_sample = build_target_sample(
        s2,
        true_s2,
        SAMPLE_S2_S3_SIZE,
        RANDOM_STATE
    )

    s3_sample = build_target_sample(
        s3,
        true_s3,
        SAMPLE_S2_S3_SIZE,
        RANDOM_STATE + 1
    )

    print(
        f"S1={len(s1_sample):,} "
        f"S2={len(s2_sample):,} "
        f"S3={len(s3_sample):,}"
    )
    print(
        f"S1 with GT matches={len(sample_gt):,}"
    )

    # Test a few caps. Each run is independent.
    caps = [20, 50, 100]

    for cap in caps:
        print(
            f"\n--- Testing max_candidates={cap} ---",
            flush=True
        )

        t = time.time()

        cands, _, _, _ = generate_candidates(
            s1_sample,
            s2_sample,
            s3_sample,
            max_candidates=cap
        )

        hits = 0

        for sid, true_ids in sample_gt.items():
            if true_ids & {
                str(x)
                for x in cands.get(sid, set())
            }:
                hits += 1

        recall = (
            hits / len(sample_gt)
            if sample_gt else 0.0
        )

        print(
            f"cap={cap}: recall={recall:.4f} "
            f"({hits:,}/{len(sample_gt):,}) "
            f"| time={time.time()-t:.1f}s"
        )

    print(
        f"\nTotal: {time.time()-t0:.1f}s"
    )


if __name__ == "__main__":
    main()
