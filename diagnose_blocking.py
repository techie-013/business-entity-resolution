import os
import time
import random
import pandas as pd

# ------------------------------------------------------------
# FAST BLOCKING-RECALL DIAGNOSTIC
#
# Purpose:
#   Measure whether each blocking key can recover TRUE matches.
#
# Unlike the previous diagnostic, this does NOT build millions
# of block dictionaries and does NOT generate candidate pools.
# It only evaluates the actual ground-truth pairs.
# ------------------------------------------------------------

SAMPLE_S1 = 2_000          # keep this small for a quick first run
RANDOM_STATE = 42

# Adjust only if your project uses different paths.
DATA_DIR = "dataset/train"

S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")

LEGAL = {
    "incorporated", "limited", "company", "corporation", "private",
    "llc", "inc", "ltd", "corp", "pvt", "co", "sarl", "sas", "sa",
    "eurl", "snc", "pllc", "lp", "plc"
}


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    return " ".join(x.split())


def normalize_name(x):
    return normalize_text(x)


def normalize_address(x):
    return normalize_text(x)


def extract_postal(address):
    import re

    # Cached data can contain NaN floats for missing addresses.
    if pd.isna(address):
        return ""

    address = str(address).strip()

    if not address:
        return ""

    # Supports 4-6 digit postal candidates.
    matches = re.findall(r"\b\d{4,6}\b", address)
    return matches[-1] if matches else ""


def prepare_keys(df):
    df = df.copy()

    df["country_l"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_addr"] = df["business_address"].map(normalize_address)
    df["postal"] = df["business_address"].map(extract_postal).fillna("")

    df["postal3"] = df["postal"].str[:3]

    tokens = df["norm_name"].str.split()
    df["first_token"] = tokens.str[0].fillna("")
    df["first3"] = df["norm_name"].str[:3]
    df["first4"] = df["norm_name"].str[:4]

    df["sorted_tokens"] = df["norm_name"].map(
        lambda x: " ".join(sorted(x.split())) if x else ""
    )

    df["core_tokens"] = df["norm_name"].map(
        lambda x: " ".join(
            sorted(t for t in x.split() if t not in LEGAL)
        ) if x else ""
    )

    import re

    df["street_num"] = (
        df["norm_addr"]
        .str.extract(
            r"^\s*(\d+[A-Za-z]?(?:/\d+)?)",
            expand=False
        )
        .fillna("")
        .str.lower()
    )

    df["postal_house"] = (
        df["postal"].fillna("").astype(str)
        .where(
            (df["postal"].fillna("").astype(str) != "")
            & (df["street_num"].fillna("").astype(str) != ""),
            ""
        )
        + "|"
        + df["street_num"].fillna("").astype(str)
    )
    df.loc[
        df["postal"].fillna("").astype(str).eq("")
        | df["street_num"].fillna("").astype(str).eq(""),
        "postal_house"
    ] = ""

    df["addr_sorted_tokens"] = df["norm_addr"].map(
        lambda x: " ".join(sorted(set(x.split()))) if x else ""
    )

    return df


KEYS = [
    ("ph", "postal_house"),
    ("p", "postal"),
    ("k", "core_tokens"),
    ("s", "sorted_tokens"),
    ("a", "addr_sorted_tokens"),
    ("p3", "postal3"),
    ("t", "first_token"),
    ("f4", "first4"),
    ("f3", "first3"),
    ("n", "street_num"),
]


# ------------------------------------------------------------
# Loading helpers
# ------------------------------------------------------------

def load_table(path, columns=None):
    # Prefer the enriched pickle made by your data_loader.
    pkl = path + ".pkl"

    if os.path.exists(pkl):
        print(f"Loading cache: {pkl}", flush=True)
        return pd.read_pickle(pkl)

    print(f"Loading: {path}", flush=True)
    return pd.read_csv(path, sep="\t", usecols=columns)


def parse_gt(gt):
    gt = gt.copy()

    # Make both columns strings.
    gt["source1_entity_id"] = gt["source1_entity_id"].astype(str)

    mapping = {}

    for _, row in gt.iterrows():
        raw = row["matched_entity_ids"]

        if pd.isna(raw):
            mapping[row["source1_entity_id"]] = set()
            continue

        ids = {
            x.strip()
            for x in str(raw).split(",")
            if x.strip()
        }

        mapping[row["source1_entity_id"]] = ids

    return mapping


# ------------------------------------------------------------
# Main diagnostic
# ------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 70)
    print("FAST BLOCKING RECALL DIAGNOSTIC")
    print("=" * 70)

    # We only need IDs/names/addresses/country from S1 and GT.
    s1 = load_table(
        S1_FILE,
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    s1["entity_id"] = s1["entity_id"].astype(str)

    gt_map = parse_gt(gt)

    # Sample S1.
    sample_s1 = s1.sample(
        n=min(SAMPLE_S1, len(s1)),
        random_state=RANDOM_STATE,
    )

    sample_ids = set(sample_s1["entity_id"])

    # Keep only GT rows for our sample.
    sample_gt = {
        sid: gt_map.get(sid, set())
        for sid in sample_ids
    }

    # Collect only TRUE target IDs needed for this sample.
    s2_ids = set()
    s3_ids = set()

    for tids in sample_gt.values():
        for tid in tids:
            if tid.startswith("S2-"):
                s2_ids.add(tid)
            elif tid.startswith("S3-"):
                s3_ids.add(tid)

    print(f"S1 sample: {len(sample_s1):,}")
    print(f"True S2 targets needed: {len(s2_ids):,}")
    print(f"True S3 targets needed: {len(s3_ids):,}")

    # --------------------------------------------------------
    # Load target rows.
    #
    # If pkl caches exist, this is fast.
    # Otherwise read CSV in chunks and keep only target IDs.
    # --------------------------------------------------------

    def get_target_rows(path, wanted_ids, label):
        pkl = path + ".pkl"

        if os.path.exists(pkl):
            print(f"[{label}] reading cache...", flush=True)
            df = pd.read_pickle(pkl)
            df["entity_id"] = df["entity_id"].astype(str)

            return df[df["entity_id"].isin(wanted_ids)].copy()

        print(f"[{label}] scanning CSV in chunks...", flush=True)

        found = []
        found_ids = set()

        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=250_000,
        ):
            chunk["entity_id"] = chunk["entity_id"].astype(str)
            hit = chunk[chunk["entity_id"].isin(wanted_ids)]

            if len(hit):
                found.append(hit)
                found_ids.update(hit["entity_id"])

            if len(found_ids) == len(wanted_ids):
                break

        if found:
            return pd.concat(found, ignore_index=True)

        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        )

    s2 = get_target_rows(S2_FILE, s2_ids, "S2")
    s3 = get_target_rows(S3_FILE, s3_ids, "S3")

    print(f"S2 target rows loaded: {len(s2):,}")
    print(f"S3 target rows loaded: {len(s3):,}")

    # Prepare keys only for the sampled S1 and TRUE targets.
    s1e = prepare_keys(sample_s1)
    s2e = prepare_keys(s2)
    s3e = prepare_keys(s3)

    s2_idx = s2e.set_index("entity_id")
    s3_idx = s3e.set_index("entity_id")
    s1_idx = s1e.set_index("entity_id")

    # Map each S1 ID to its exact S2/S3 target row.
    total_pairs = 0
    block_hits = {tag: 0 for tag, _ in KEYS}
    union_hits = 0

    missed_examples = []

    for sid, tids in sample_gt.items():

        if sid not in s1_idx.index:
            continue

        row = s1_idx.loc[sid]

        for tid in tids:

            if tid.startswith("S2-"):
                target_df = s2_idx
            elif tid.startswith("S3-"):
                target_df = s3_idx
            else:
                continue

            if tid not in target_df.index:
                # GT points to a target that isn't present in the loaded data.
                continue

            target = target_df.loc[tid]
            total_pairs += 1

            pair_hit = False
            hit_tags = []

            for tag, col in KEYS:

                v1 = row[col]
                v2 = target[col]

                if not v1 or not v2:
                    continue

                # This is the exact equality test a blocking key uses.
                if v1 == v2:
                    block_hits[tag] += 1
                    pair_hit = True
                    hit_tags.append(tag)

            if pair_hit:
                union_hits += 1
            elif len(missed_examples) < 20:
                missed_examples.append({
                    "s1_id": sid,
                    "target_id": tid,
                    "name": row["business_name"],
                    "target_name": target["business_name"],
                    "address": row["business_address"],
                    "target_address": target["business_address"],
                })

    print()
    print("=" * 70)
    print("PAIR-LEVEL BLOCKING RECALL")
    print("=" * 70)

    if total_pairs == 0:
        print("No usable ground-truth pairs found.")
        return

    for tag, _ in KEYS:
        hits = block_hits[tag]
        recall = hits / total_pairs
        print(f"{tag:>3}: {recall:.4f} ({hits:,}/{total_pairs:,})")

    union_recall = union_hits / total_pairs

    print("-" * 70)
    print(
        f"UNION: {union_recall:.4f} "
        f"({union_hits:,}/{total_pairs:,})"
    )

    print()
    print("=" * 70)
    print("MISSED TRUE MATCH EXAMPLES")
    print("=" * 70)

    if not missed_examples:
        print("No misses in the sampled true pairs.")
    else:
        for x in missed_examples:
            print(f"S1 : {x['name']}")
            print(f"S2/S3: {x['target_name']}")
            print(f"A1 : {x['address']}")
            print(f"A2 : {x['target_address']}")
            print(f"IDs: {x['s1_id']} -> {x['target_id']}")
            print("-" * 50)

    print()
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
