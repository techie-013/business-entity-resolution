import os
import re
import time
import pandas as pd

# IMPORTANT:
# This experiment imports the CURRENT src.blocking.py.
# It does not modify it.
#
# It reproduces the current production blocker exactly for the baseline,
# then tests adding a small reserved slice for an address block.

from src.blocking import _prepare_keys, LEGAL

SAMPLE_S1 = 2000
CHUNK_SIZE = 250_000
RANDOM_STATE = 42

DATA_DIR = "dataset/train"
S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")


def address_key(address):
    if address is None:
        return ""

    try:
        if pd.isna(address):
            return ""
    except Exception:
        pass

    text = str(address).lower()
    tokens = re.findall(r"\d+[a-zA-Z]?|[a-zA-Z]+", text)

    generic = {
        "road", "street", "avenue", "boulevard", "drive", "lane",
        "highway", "place", "square", "north", "south", "east", "west",
        "number", "no", "floor", "unit", "apt", "apartment", "block",
        "plot", "sector", "village", "district", "county", "city",
        "state", "the", "and"
    }

    for i, token in enumerate(tokens):
        if re.fullmatch(r"\d+[a-zA-Z]?", token):
            digits = re.sub(r"\D", "", token)

            if len(digits) < 2:
                continue

            for nxt in tokens[i + 1:i + 5]:
                if (
                    nxt.isalpha()
                    and nxt.lower() not in generic
                    and len(nxt) >= 3
                ):
                    return f"{token.lower()}|{nxt[:4].lower()}"

            return ""

    return ""


def load_gt():
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


def add_current_blocks(df, blocks, wanted):
    """
    Same key construction as the current src.blocking.py,
    using its actual _prepare_keys().
    """
    c = _prepare_keys(df)

    keys = [
        ("p", "postal3", 1),
        ("t", "first_token", 1),
        ("f", "first3", 1),
        ("s", "sorted_tokens", 1),
        ("k", "core_tokens", 1),
        ("n", "street_num", 3),
    ]

    for tag, col, min_len in keys:
        valid = c[c[col].astype(str).str.len() >= min_len]

        for country, value, eid in valid[
            ["country_l", col, "entity_id"]
        ].itertuples(index=False, name=None):
            if not value:
                continue

            key = f"{country}|{tag}|{value}"
            if key in wanted:
                blocks.setdefault(key, set()).add(str(eid))


def add_address_blocks(df, address_blocks, wanted):
    c = _prepare_keys(df)
    c["address_key"] = c["business_address"].map(address_key)

    valid = c[c["address_key"].astype(str).str.len() >= 3]

    for country, value, eid in valid[
        ["country_l", "address_key", "entity_id"]
    ].itertuples(index=False, name=None):
        if not value:
            continue

        key = f"{country}|h|{value}"
        if key in wanted:
            address_blocks.setdefault(key, set()).add(str(eid))


def baseline_candidates(row, blocks, cap=100):
    # This mirrors the current generate_candidates() ordering exactly.
    strong = set()

    if row.postal3:
        strong |= blocks.get(
            f"{row.country_l}|p|{row.postal3}", set()
        )
    if row.sorted_tokens:
        strong |= blocks.get(
            f"{row.country_l}|s|{row.sorted_tokens}", set()
        )
    if row.core_tokens:
        strong |= blocks.get(
            f"{row.country_l}|k|{row.core_tokens}", set()
        )

    medium = set()

    if row.first_token:
        medium |= blocks.get(
            f"{row.country_l}|t|{row.first_token}", set()
        )
    if row.first3:
        medium |= blocks.get(
            f"{row.country_l}|f|{row.first3}", set()
        )

    weak = set()

    if row.street_num and len(row.street_num) >= 3:
        weak |= blocks.get(
            f"{row.country_l}|n|{row.street_num}", set()
        )

    cand = list(strong)

    if len(cand) < cap:
        cand += [
            x for x in medium
            if x not in strong
        ]

    if len(cand) < cap:
        cand += [
            x for x in weak
            if x not in strong and x not in medium
        ]

    return cand[:cap]


def main():
    start = time.time()

    print("=" * 72)
    print("EXACT CURRENT-BLOCKER EXPERIMENT")
    print("=" * 72)

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )
    s1["entity_id"] = s1["entity_id"].astype(str)

    sample = s1.sample(
        n=min(SAMPLE_S1, len(s1)),
        random_state=RANDOM_STATE
    ).copy()
    sample_prepared = _prepare_keys(sample).set_index("entity_id")

    gt = load_gt()
    sample_gt = {
        sid: gt.get(sid, set())
        for sid in sample_prepared.index
        if gt.get(sid, set())
    }

    total_s1 = len(sample_gt)
    total_pairs = sum(len(v) for v in sample_gt.values())

    wanted_normal = set()
    wanted_address = set()

    for _, row in sample_prepared.iterrows():
        for tag, col, min_len in [
            ("p", "postal3", 1),
            ("t", "first_token", 1),
            ("f", "first3", 1),
            ("s", "sorted_tokens", 1),
            ("k", "core_tokens", 1),
            ("n", "street_num", 3),
        ]:
            value = row[col]
            if not value:
                continue
            if tag == "n" and len(str(value)) < min_len:
                continue
            wanted_normal.add(
                f"{row.country_l}|{tag}|{value}"
            )

        h = address_key(row["business_address"])
        if h:
            wanted_address.add(
                f"{row.country_l}|h|{h}"
            )

    normal_blocks = {}
    address_blocks = {}

    def scan(path, label):
        print(f"[{label}] scanning full target...", flush=True)
        total = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ],
            chunksize=CHUNK_SIZE
        ):
            total += len(chunk)
            add_current_blocks(
                chunk,
                normal_blocks,
                wanted_normal
            )
            add_address_blocks(
                chunk,
                address_blocks,
                wanted_address
            )

            if total % 1_000_000 == 0:
                print(f"  {total:,} rows", flush=True)

        print(f"[{label}] done: {total:,}", flush=True)

    scan(S2_FILE, "S2")
    scan(S3_FILE, "S3")

    print(f"Normal blocks: {len(normal_blocks):,}")
    print(f"Address blocks: {len(address_blocks):,}")

    variants = {
        "baseline": 0,
        "reserve_5": 5,
        "reserve_10": 10,
        "reserve_20": 20,
        "reserve_30": 30,
    }

    s1_hits = {k: 0 for k in variants}
    pair_hits = {k: 0 for k in variants}

    for sid, true_ids in sample_gt.items():
        row = sample_prepared.loc[sid]

        base = baseline_candidates(
            row,
            normal_blocks,
            cap=100
        )

        h = address_key(row["business_address"])
        addr = list(
            address_blocks.get(
                f"{row.country_l}|h|{h}",
                set()
            )
        ) if h else []

        for name, reserve in variants.items():
            if reserve == 0:
                candidates = set(base)
            else:
                keep = 100 - reserve
                chosen = list(base[:keep])

                for eid in addr:
                    if eid not in chosen:
                        chosen.append(eid)
                    if len(chosen) >= 100:
                        break

                candidates = set(chosen[:100])

            found = true_ids & candidates

            if found:
                s1_hits[name] += 1

            pair_hits[name] += len(found)

    print()
    print("=" * 72)
    print("CAP=100 RESULTS")
    print("=" * 72)

    for name in variants:
        print(
            f"{name:>10}: "
            f"S1 recall={s1_hits[name] / total_s1:.4f} "
            f"({s1_hits[name]:,}/{total_s1:,}) "
            f"| pair recall={pair_hits[name] / total_pairs:.4f} "
            f"({pair_hits[name]:,}/{total_pairs:,})"
        )

    print()
    print(
        f"Total time: {time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
