import os
import re
import time
import pandas as pd

# ============================================================
# SAFE BLOCKING EXPERIMENT
#
# Does NOT modify src/blocking.py.
#
# Compares current known-good blocking at cap=100 against
# reserving 5/10/20/30 of the 100 slots for a new address key.
# ============================================================

SAMPLE_S1 = 2_000
CHUNK_SIZE = 250_000
RANDOM_STATE = 42

DATA_DIR = "dataset/train"
S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")

LEGAL = {
    "incorporated", "limited", "company", "corporation",
    "private", "llc", "inc", "ltd", "corp", "pvt", "co",
    "sarl", "sas", "sa", "eurl", "snc", "pllc", "lp", "plc"
}


def norm(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    x = str(x).lower()
    x = re.sub(r"[^\w\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def address_key(x):
    if not x:
        return ""

    tokens = re.findall(
        r"\d+[a-zA-Z]?|[a-zA-Z]+",
        x
    )

    generic = {
        "road", "street", "avenue", "boulevard", "drive",
        "lane", "highway", "place", "square", "north",
        "south", "east", "west", "number", "no", "floor",
        "unit", "apt", "apartment", "block", "plot",
        "sector", "village", "district", "county", "city",
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


def prepare(df):
    df = df.copy()

    df["country_l"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["norm_name"] = df["business_name"].map(norm)
    df["norm_addr"] = df["business_address"].map(norm)

    df["postal"] = df["business_address"].map(
        lambda x: (
            re.findall(r"\b\d{5}\b", str(x))[-1]
            if isinstance(x, str) and re.findall(r"\b\d{5}\b", x)
            else ""
        )
    )

    df["postal3"] = df["postal"].astype(str).str[:3]
    df["first_token"] = (
        df["norm_name"].str.split().str[0].fillna("")
    )
    df["first3"] = df["norm_name"].str[:3]

    df["sorted_tokens"] = df["norm_name"].map(
        lambda x: " ".join(sorted(x.split())) if x else ""
    )

    df["core_tokens"] = df["norm_name"].map(
        lambda x: " ".join(
            sorted(
                t for t in x.split()
                if t not in LEGAL
            )
        ) if x else ""
    )

    df["street_num"] = (
        df["norm_addr"]
        .str.extract(r"(\b\d{2,5}\b)", expand=False)
        .fillna("")
    )

    df["h"] = df["norm_addr"].map(address_key)

    return df


def load_gt():
    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )

    gt["source1_entity_id"] = (
        gt["source1_entity_id"].astype(str)
    )

    out = {}

    for row in gt.itertuples(index=False):
        raw = row.matched_entity_ids
        out[row.source1_entity_id] = (
            set()
            if pd.isna(raw)
            else {
                x.strip()
                for x in str(raw).split(",")
                if x.strip()
            }
        )

    return out


def add_blocks(chunk, relevant, normal_blocks, address_blocks):
    c = prepare(chunk)

    keys = [
        ("p", "postal3"),
        ("s", "sorted_tokens"),
        ("k", "core_tokens"),
        ("t", "first_token"),
        ("f", "first3"),
        ("n", "street_num"),
    ]

    for tag, col in keys:
        min_len = 3 if tag == "n" else 1
        valid = c[
            c[col].astype(str).str.len() >= min_len
        ]

        for country, value, eid in valid[
            ["country_l", col, "entity_id"]
        ].itertuples(index=False, name=None):
            if not value:
                continue

            key = f"{country}|{tag}|{value}"

            if key in relevant:
                normal_blocks.setdefault(
                    key, set()
                ).add(str(eid))

    valid_h = c[
        c["h"].astype(str).str.len() >= 3
    ]

    for country, value, eid in valid_h[
        ["country_l", "h", "entity_id"]
    ].itertuples(index=False, name=None):
        if not value:
            continue

        key = f"{country}|h|{value}"

        if key in relevant:
            address_blocks.setdefault(
                key, set()
            ).add(str(eid))


def scan_target(path, relevant, normal_blocks, address_blocks, label):
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

        add_blocks(
            chunk,
            relevant,
            normal_blocks,
            address_blocks
        )

        if total % 1_000_000 == 0:
            print(f"  {total:,} rows", flush=True)

    print(f"[{label}] done: {total:,}", flush=True)


def baseline_candidates(row, blocks, cap=100):
    strong = set()

    if row.postal3:
        strong |= blocks.get(
            f"{row.country_l}|p|{row.postal3}",
            set()
        )

    if row.sorted_tokens:
        strong |= blocks.get(
            f"{row.country_l}|s|{row.sorted_tokens}",
            set()
        )

    if row.core_tokens:
        strong |= blocks.get(
            f"{row.country_l}|k|{row.core_tokens}",
            set()
        )

    medium = set()

    if row.first_token:
        medium |= blocks.get(
            f"{row.country_l}|t|{row.first_token}",
            set()
        )

    if row.first3:
        medium |= blocks.get(
            f"{row.country_l}|f|{row.first3}",
            set()
        )

    weak = set()

    if row.street_num and len(row.street_num) >= 3:
        weak |= blocks.get(
            f"{row.country_l}|n|{row.street_num}",
            set()
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
            if x not in strong
            and x not in medium
        ]

    return cand[:cap]


def main():
    start = time.time()

    print("=" * 72)
    print("SAFE BLOCKING EXPERIMENT — CAP 100")
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

    sample = prepare(
        s1.sample(
            n=min(SAMPLE_S1, len(s1)),
            random_state=RANDOM_STATE
        )
    ).set_index("entity_id")

    gt = load_gt()

    sample_gt = {
        sid: gt.get(sid, set())
        for sid in sample.index
        if gt.get(sid, set())
    }

    total_s1 = len(sample_gt)
    total_pairs = sum(
        len(ids)
        for ids in sample_gt.values()
    )

    relevant = set()

    for _, row in sample.iterrows():
        keys = [
            ("p", row.postal3),
            ("s", row.sorted_tokens),
            ("k", row.core_tokens),
            ("t", row.first_token),
            ("f", row.first3),
            ("n", row.street_num),
        ]

        for tag, value in keys:
            if not value:
                continue
            if tag == "n" and len(str(value)) < 3:
                continue
            relevant.add(
                f"{row.country_l}|{tag}|{value}"
            )

        if row.h:
            relevant.add(
                f"{row.country_l}|h|{row.h}"
            )

    normal_blocks = {}
    address_blocks = {}

    scan_target(
        S2_FILE,
        relevant,
        normal_blocks,
        address_blocks,
        "S2"
    )

    scan_target(
        S3_FILE,
        relevant,
        normal_blocks,
        address_blocks,
        "S3"
    )

    print(
        f"Normal blocks: {len(normal_blocks):,}"
    )
    print(
        f"Address blocks: {len(address_blocks):,}"
    )

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
        row = sample.loc[sid]

        base = baseline_candidates(
            row,
            normal_blocks,
            cap=100
        )

        addr = list(
            address_blocks.get(
                f"{row.country_l}|h|{row.h}",
                set()
            )
        ) if row.h else []

        for name, reserve in variants.items():
            if reserve == 0:
                candidates = set(base)
            else:
                keep = 100 - reserve

                # Preserve most of the proven baseline candidates.
                candidates_list = base[:keep]

                # Use reserved slots for the new address block.
                for eid in addr:
                    if eid not in candidates_list:
                        candidates_list.append(eid)

                    if len(candidates_list) >= 100:
                        break

                candidates = set(
                    candidates_list[:100]
                )

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

    print(
        f"\nTotal time: {time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
