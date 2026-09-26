import os
import re
import time
import unicodedata
import pandas as pd

# ============================================================
# ACTUAL PRODUCTION BLOCKING-RECALL DIAGNOSTIC
#
# This reproduces the CURRENT blocking.py logic:
#   strong: postal3 + sorted_tokens + core_tokens
#   medium: first_token + first3
#   weak: street_num
#
# It uses FULL S2/S3 files, but only for a 2,000-S1 sample.
# Target rows are scanned in chunks so we don't build giant
# full block dictionaries.
#
# It measures:
#   1) pair-level recall of the current 6 blockers
#   2) actual candidate recall after caps 100/200/300/500
# ============================================================

SAMPLE_S1 = 2_000
CHUNK_SIZE = 250_000
RANDOM_STATE = 42

# Pick the path that exists on your machine.
if os.path.exists("dataset/train/train_source1.tsv"):
    DATA_DIR = "dataset/train"
elif os.path.exists("data/train/train_source1.tsv"):
    DATA_DIR = "data/train"
else:
    raise FileNotFoundError(
        "Could not find dataset/train or data/train."
    )

S1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
S2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
S3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")


LEGAL = {
    "incorporated", "limited", "company", "corporation",
    "private", "llc", "inc", "ltd", "corp", "pvt", "co",
    "sarl", "sas", "sa", "eurl", "snc", "pllc", "lp", "plc"
}

NAME_ABBREVIATIONS = {
    "corp": "corporation", "co": "company", "ltd": "limited",
    "inc": "incorporated", "llc": "limited liability company",
    "pvt": "private", "pte": "private",
    "sarl": "sarl", "sas": "sas", "sa": "sa", "eurl": "eurl",
    "snc": "snc", "ste": "societe", "cie": "compagnie",
    "intl": "international", "svc": "service", "svcs": "services",
    "mfg": "manufacturing", "tech": "technology",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "sq": "square", "ln": "lane",
    "dr": "drive", "hwy": "highway",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "no": "number", "&": "and", "r": "rue", "rue": "rue",
    "bd": "boulevard", "pl": "place", "che": "chemin",
    "imp": "impasse", "all": "allee",
}


def normalize_text(s):
    if s is None:
        return ""
    if isinstance(s, float) and s != s:
        return ""

    text = str(s)
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize(
        "NFKD", text
    )
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def expand_abbreviations(text, mapping):
    if not isinstance(text, str):
        return ""
    return " ".join(
        mapping.get(tok, tok)
        for tok in text.split()
    )


def normalize_name(s):
    return expand_abbreviations(
        normalize_text(s),
        NAME_ABBREVIATIONS
    )


def normalize_address(s):
    return expand_abbreviations(
        normalize_text(s),
        ADDRESS_ABBREVIATIONS
    )


def extract_postal(s):
    # EXACT current repo behavior: 5-digit postal code.
    if s is None or (isinstance(s, float) and s != s):
        return ""

    m = re.search(
        r"\b(\d{5})(?:-?\d{4})?\b",
        str(s)
    )
    return m.group(1) if m else ""


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
    df["norm_addr"] = df["business_address"].map(
        normalize_address
    )
    df["postal"] = df["business_address"].map(extract_postal)

    # Exact current blocking.py key construction.
    df["postal3"] = (
        df["postal"].astype(str).str[:3]
    )

    df["first_token"] = (
        df["norm_name"]
        .astype(str)
        .str.split()
        .str[0]
        .fillna("")
    )

    df["first3"] = (
        df["norm_name"]
        .astype(str)
        .str[:3]
    )

    df["sorted_tokens"] = df["norm_name"].apply(
        lambda x:
        " ".join(sorted(str(x).split()))
        if x else ""
    )

    df["core_tokens"] = df["norm_name"].apply(
        lambda x:
        " ".join(
            sorted(
                t for t in str(x).split()
                if t not in LEGAL
            )
        )
        if x else ""
    )

    df["street_num"] = (
        df["norm_addr"]
        .astype(str)
        .str.extract(
            r"(\b\d{2,5}\b)",
            expand=False
        )
        .fillna("")
    )

    return df


BLOCKS = [
    ("p", "postal3", True),          # strong
    ("s", "sorted_tokens", True),    # strong
    ("k", "core_tokens", True),      # strong
    ("t", "first_token", False),     # medium
    ("f", "first3", False),          # medium
    ("n", "street_num", False),      # weak
]


def add_rows_to_blocks(
    chunk,
    relevant_keys,
    blocks,
):
    """
    Add only rows whose block keys are relevant to our
    sampled S1 queries.

    blocks:
        dict[block_key] -> set(entity_ids)
    """

    c = prepare_keys(chunk)

    for tag, col, _ in BLOCKS:

        # Street-number blocker requires >= 3 chars.
        if tag == "n":
            valid = c[
                c[col].astype(str).str.len() >= 3
            ]
        else:
            valid = c[
                c[col].astype(str).str.len() >= 1
            ]

        if valid.empty:
            continue

        # Only keep keys that at least one sample query uses.
        pairs = valid[
            ["country_l", col, "entity_id"]
        ]

        for country, key, entity_id in pairs.itertuples(
            index=False,
            name=None
        ):
            if not key:
                continue

            block_key = (
                f"{country}|{tag}|{key}"
            )

            if block_key not in relevant_keys:
                continue

            blocks.setdefault(
                block_key, set()
            ).add(str(entity_id))


def query_candidates(
    row,
    blocks,
    max_candidates=None,
):
    c = row["country_l"]

    strong = set()
    medium = set()
    weak = set()

    for tag, col, is_strong in BLOCKS:
        value = row[col]

        if not value:
            continue

        # Current code requires >= 3 chars for street_num.
        if tag == "n" and len(str(value)) < 3:
            continue

        key = f"{c}|{tag}|{value}"
        hits = blocks.get(key, set())

        if is_strong:
            strong |= hits
        elif tag in ("t", "f"):
            medium |= hits
        else:
            weak |= hits

    cand = list(strong)

    if max_candidates is None:
        cand += [
            x for x in medium
            if x not in strong
        ]
        cand += [
            x for x in weak
            if x not in strong
            and x not in medium
        ]
    else:
        if len(cand) < max_candidates:
            cand += [
                x for x in medium
                if x not in strong
            ]

        if len(cand) < max_candidates:
            cand += [
                x for x in weak
                if x not in strong
                and x not in medium
            ]

        cand = cand[:max_candidates]

    return set(cand), len(strong), len(medium), len(weak)


def load_gt():
    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ]
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


def scan_target(
    path,
    wanted_block_keys,
    blocks,
    label,
):
    print(
        f"[{label}] scanning full target file...",
        flush=True
    )

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        chunksize=CHUNK_SIZE,
    ):
        total_rows += len(chunk)

        add_rows_to_blocks(
            chunk,
            wanted_block_keys,
            blocks,
        )

        if total_rows % 1_000_000 == 0:
            print(
                f"  [{label}] scanned {total_rows:,} rows",
                flush=True
            )

    print(
        f"[{label}] finished: {total_rows:,} rows",
        flush=True
    )


def main():
    t0 = time.time()

    print("=" * 72)
    print("ACTUAL PRODUCTION BLOCKING RECALL")
    print("=" * 72)

    print(f"Data directory: {DATA_DIR}")

    # --------------------------------------------------------
    # Load S1 sample + ground truth
    # --------------------------------------------------------

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    )

    s1["entity_id"] = (
        s1["entity_id"].astype(str)
    )

    sample = s1.sample(
        n=min(SAMPLE_S1, len(s1)),
        random_state=RANDOM_STATE
    ).copy()

    sample = prepare_keys(sample)
    sample = sample.set_index("entity_id")

    gt = load_gt()

    sample_gt = {
        sid: gt.get(sid, set())
        for sid in sample.index
        if gt.get(sid, set())
    }

    total_pairs = sum(
        len(ids)
        for ids in sample_gt.values()
    )

    print(f"S1 sample: {len(sample):,}")
    print(f"Ground-truth S1 with matches: {len(sample_gt):,}")
    print(f"TRUE PAIRS: {total_pairs:,}")

    # --------------------------------------------------------
    # Determine exactly which blocks our sample asks for.
    # --------------------------------------------------------

    relevant = set()

    for sid, row in sample.iterrows():
        for tag, col, _ in BLOCKS:

            value = row[col]

            if not value:
                continue

            if tag == "n" and len(str(value)) < 3:
                continue

            relevant.add(
                f"{row['country_l']}|{tag}|{value}"
            )

    print(
        f"Relevant query block keys: {len(relevant):,}"
    )

    blocks = {}

    # --------------------------------------------------------
    # Scan FULL S2 + FULL S3, but retain only rows that can
    # participate in one of the sampled query's block keys.
    # --------------------------------------------------------

    scan_target(
        S2_FILE,
        relevant,
        blocks,
        "S2"
    )

    scan_target(
        S3_FILE,
        relevant,
        blocks,
        "S3"
    )

    print(
        f"Relevant blocks populated: {len(blocks):,}"
    )

    # --------------------------------------------------------
    # Evaluate candidate recall.
    # --------------------------------------------------------

    caps = [100, 200, 300, 500]

    results = {
        cap: {
            "hits": 0,
            "candidate_total": 0,
        }
        for cap in caps
    }

    pair_union_hits = 0
    key_hits = {
        tag: 0
        for tag, _, _ in BLOCKS
    }

    missed_examples = {
        cap: []
        for cap in caps
    }

    for sid, tids in sample_gt.items():

        row = sample.loc[sid]

        # Individual blocker hit check.
        full_cands, strong_n, medium_n, weak_n = (
            query_candidates(
                row,
                blocks,
                max_candidates=None
            )
        )

        for tid in tids:
            # Count which current blocker(s) could recover it.
            # We don't need separate target lookup because the
            # block candidate IDs are the entity IDs themselves.

            if tid in full_cands:
                pair_union_hits += 1

            # individual exact blocker membership
            for tag, col, is_strong in BLOCKS:
                value = row[col]

                if not value:
                    continue
                if tag == "n" and len(str(value)) < 3:
                    continue

                key = (
                    f"{row['country_l']}|{tag}|{value}"
                )

                if tid in blocks.get(key, set()):
                    key_hits[tag] += 1

        for cap in caps:
            cands, _, _, _ = query_candidates(
                row,
                blocks,
                max_candidates=cap
            )

            results[cap]["candidate_total"] += len(
                cands
            )

            row_hit = tids & cands

            # Match the train_model.py definition:
            # S1 is a hit if ANY true target is in candidates.
            if row_hit:
                results[cap]["hits"] += 1
            elif len(missed_examples[cap]) < 20:
                missed_examples[cap].append(
                    (sid, next(iter(tids)), row)
                )

    print()
    print("=" * 72)
    print("INDIVIDUAL BLOCK RECALL")
    print("=" * 72)

    for tag, _, _ in BLOCKS:
        recall = (
            key_hits[tag] / total_pairs
            if total_pairs else 0
        )
        print(
            f"{tag:>3}: {recall:.4f} "
            f"({key_hits[tag]:,}/{total_pairs:,})"
        )

    pair_union = (
        pair_union_hits / total_pairs
        if total_pairs else 0
    )

    print("-" * 72)
    print(
        f"PAIR UNION: {pair_union:.4f} "
        f"({pair_union_hits:,}/{total_pairs:,})"
    )

    print()
    print("=" * 72)
    print("ACTUAL CANDIDATE RECALL")
    print("=" * 72)

    for cap in caps:
        row_hits = results[cap]["hits"]
        s1_total = len(sample_gt)

        recall = (
            row_hits / s1_total
            if s1_total else 0
        )

        avg_candidates = (
            results[cap]["candidate_total"] /
            len(sample_gt)
            if sample_gt else 0
        )

        print(
            f"cap={cap:>3}: "
            f"{recall:.4f} "
            f"({row_hits:,}/{s1_total:,}) "
            f"| avg candidates={avg_candidates:.1f}"
        )

    print()
    print("=" * 72)
    print("IMPORTANT")
    print("=" * 72)
    print(
        "The production train_model.py currently samples "
        "S2/S3 before blocking and uses max_candidates=200."
    )
    print(
        "This diagnostic uses FULL S2/S3 and therefore gives "
        "a much more realistic measurement of actual blocking."
    )

    print()
    print("=" * 72)
    print("MISSED EXAMPLES AT cap=200")
    print("=" * 72)

    for sid, tid, row in missed_examples[200]:
        print(f"S1 : {row['business_name']}")
        print(f"GT : {tid}")
        print(f"ADR: {row['business_address']}")
        print("-" * 50)

    print()
    print(
        f"Total time: {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
