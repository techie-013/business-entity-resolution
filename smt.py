import os
import time
import pandas as pd

from src.blocking import generate_candidates

DATA_DIR = "dataset/train"
S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")

S1_SAMPLE = 500
TARGET_SAMPLE = 100_000
RANDOM_STATE = 42
CAP = 100


def load_ground_truth():
    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        usecols=["source1_entity_id", "matched_entity_ids"]
    )

    gt["source1_entity_id"] = gt["source1_entity_id"].astype(str)

    out = {}

    for row in gt.itertuples(index=False):
        raw = row.matched_entity_ids

        if pd.isna(raw):
            out[row.source1_entity_id] = set()
        else:
            out[row.source1_entity_id] = {
                x.strip()
                for x in str(raw).split(",")
                if x.strip()
            }

    return out


def load_sample(path, wanted_ids=None, n=100_000):
    cols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    if wanted_ids:
        wanted_ids = {str(x) for x in wanted_ids}
        found = []

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=cols,
            chunksize=100_000,
        ):
            chunk["entity_id"] = chunk["entity_id"].astype(str)
            hit = chunk[chunk["entity_id"].isin(wanted_ids)]

            if not hit.empty:
                found.append(hit)

            if sum(len(x) for x in found) >= len(wanted_ids):
                break

        if found:
            return pd.concat(found, ignore_index=True)

        return pd.DataFrame(columns=cols)

    return pd.read_csv(
        path,
        sep="\t",
        usecols=cols,
        nrows=n
    )


def main():
    t0 = time.time()

    print("=" * 72)
    print("LIGHTWEIGHT BLOCKING SMOKE TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. Load a small S1 sample.
    # --------------------------------------------------------
    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        nrows=S1_SAMPLE,
    )

    s1["entity_id"] = s1["entity_id"].astype(str)

    # --------------------------------------------------------
    # 2. Read GT for those S1 rows.
    # --------------------------------------------------------
    gt = load_ground_truth()

    sample_ids = set(s1["entity_id"])

    true_ids = set()
    s1_with_gt = {}

    for sid in sample_ids:
        ids = gt.get(sid, set())
        s1_with_gt[sid] = ids

        for tid in ids:
            true_ids.add(tid)

    print(f"S1 sample: {len(s1):,}")
    print(f"S1 with GT matches: {sum(bool(v) for v in s1_with_gt.values()):,}")
    print(f"Unique true target IDs needed: {len(true_ids):,}")

    # --------------------------------------------------------
    # 3. Load all true target rows plus random decoys.
    #
    # This gives the blocker real matches to recover without
    # constructing blocks across 10M+ rows.
    # --------------------------------------------------------
    print("Loading true S2 targets...", flush=True)

    true_s2_ids = {
        x for x in true_ids
        if str(x).startswith("S2-")
    }

    true_s3_ids = {
        x for x in true_ids
        if str(x).startswith("S3-")
    }

    s2_true = load_sample(
        S2_FILE,
        wanted_ids=true_s2_ids
    )

    s3_true = load_sample(
        S3_FILE,
        wanted_ids=true_s3_ids
    )

    print(f"True S2 rows: {len(s2_true):,}")
    print(f"True S3 rows: {len(s3_true):,}")

    print("Loading random S2/S3 decoys...", flush=True)

    s2_decoy = pd.read_csv(
        S2_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        skiprows=range(
            1,
            1 + 100_000
        ),
        nrows=TARGET_SAMPLE,
    )

    s3_decoy = pd.read_csv(
        S3_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        skiprows=range(
            1,
            1 + 100_000
        ),
        nrows=TARGET_SAMPLE,
    )

    s2 = pd.concat(
        [s2_true, s2_decoy],
        ignore_index=True
    ).drop_duplicates("entity_id")

    s3 = pd.concat(
        [s3_true, s3_decoy],
        ignore_index=True
    ).drop_duplicates("entity_id")

    s2["entity_id"] = s2["entity_id"].astype(str)
    s3["entity_id"] = s3["entity_id"].astype(str)

    print(f"S2 test rows: {len(s2):,}")
    print(f"S3 test rows: {len(s3):,}")

    # --------------------------------------------------------
    # 4. Run the REAL blocking.py.
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("RUNNING ACTUAL src/blocking.py")
    print(f"max_candidates={CAP}")
    print("=" * 72)

    candidates, _, _, _ = generate_candidates(
        s1,
        s2,
        s3,
        max_candidates=CAP,
    )

    # --------------------------------------------------------
    # 5. Check that every returned list obeys the cap and
    # calculate recall on the deliberately small test set.
    # --------------------------------------------------------
    cap_ok = all(
        len(candidates.get(sid, set())) <= CAP
        for sid in sample_ids
    )

    s1_hits = 0
    pair_hits = 0
    total_pairs = sum(len(v) for v in s1_with_gt.values())

    for sid, tids in s1_with_gt.items():
        if not tids:
            continue

        cand = {
            str(x)
            for x in candidates.get(sid, set())
        }

        found = tids & cand

        if found:
            s1_hits += 1

        pair_hits += len(found)

    s1_total = sum(
        bool(v)
        for v in s1_with_gt.values()
    )

    s1_recall = (
        s1_hits / s1_total
        if s1_total else 0.0
    )

    pair_recall = (
        pair_hits / total_pairs
        if total_pairs else 0.0
    )

    print()
    print("=" * 72)
    print("SMOKE TEST RESULT")
    print("=" * 72)

    print(f"Cap respected: {cap_ok}")
    print(
        f"S1 recall on test subset: "
        f"{s1_recall:.4f} "
        f"({s1_hits:,}/{s1_total:,})"
    )
    print(
        f"Pair recall on test subset: "
        f"{pair_recall:.4f} "
        f"({pair_hits:,}/{total_pairs:,})"
    )

    if not cap_ok:
        print("FAIL: candidate cap was violated.")
        return

    if s1_recall > 0:
        print("PASS: blocker runs and recovers true matches.")
    else:
        print(
            "WARNING: blocker ran but recovered no true matches. "
            "Do not train yet."
        )

    print(
        f"Total time: {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
