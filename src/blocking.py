import time
import pandas as pd

from src.normalize import normalize_name, normalize_address, extract_postal

LEGAL_SUFFIXES = {
    "incorporated", "limited", "company", "corporation",
    "private", "llc", "inc", "ltd", "corp", "pvt", "co",
    "sarl", "sas", "sa", "eurl", "snc", "pllc", "lp", "plc",
    "societe", "compagnie",
}


def enrich(df, label=""):
    print(f"  [{label}] enriching ({len(df):,} rows)...")
    t0 = time.time()

    df = df.copy()
    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_addr"] = df["business_address"].map(normalize_address)
    df["postal"] = df["business_address"].map(extract_postal)
    df["country_l"] = df["country"].astype(str).str.strip().str.lower()

    print(f"  [{label}] enriched in {time.time()-t0:.1f}s")
    return df


def _prepare_keys(df):
    df = df.copy()

    if "country_l" not in df.columns:
        df["country_l"] = df["country"].astype(str).str.strip().str.lower()
    if "norm_name" not in df.columns:
        df["norm_name"] = df["business_name"].map(normalize_name)
    if "norm_addr" not in df.columns:
        df["norm_addr"] = df["business_address"].map(normalize_address)
    if "postal" not in df.columns:
        df["postal"] = df["business_address"].map(extract_postal)

    df["postal3"] = df["postal"].astype(str).str[:3]
    df["first_token"] = df["norm_name"].astype(str).str.split().str[0].fillna("")
    df["first2"] = df["norm_name"].astype(str).str[:2]

    # FULL sorted tokens — no truncation
    df["sorted_tokens"] = df["norm_name"].apply(
        lambda x: " ".join(sorted(str(x).split())) if x else ""
    )

    # Core tokens — strip legal suffixes
    df["core_tokens"] = df["norm_name"].apply(
        lambda x: " ".join(sorted(t for t in str(x).split() if t not in LEGAL_SUFFIXES))
        if x else ""
    )

    df["street_num"] = (
        df["norm_addr"].astype(str)
        .str.extract(r"(\b\d{2,5}\b)", expand=False)
        .fillna("")
    )

    return df


def build_blocks(df, label="S"):
    print(f"  [{label}] preparing keys...")
    t0 = time.time()

    df = _prepare_keys(df)

    blocks = {}

    print(f"  [{label}] block by postal prefix...")
    valid = df[df["postal3"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "postal3"]):
        blocks[f"{key[0]}|p|{key[1]}"] = set(group["entity_id"])
    print(f"  [{label}]   -> {len(blocks):,} keys")

    print(f"  [{label}] block by first token...")
    valid = df[df["first_token"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "first_token"]):
        blocks[f"{key[0]}|t|{key[1]}"] = set(group["entity_id"])
    print(f"  [{label}]   -> {len(blocks):,} keys")

    print(f"  [{label}] block by first 2 chars...")
    valid = df[df["first2"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "first2"]):
        blocks[f"{key[0]}|c|{key[1]}"] = set(group["entity_id"])
    print(f"  [{label}]   -> {len(blocks):,} keys")

    print(f"  [{label}] block by sorted tokens (full)...")
    valid = df[df["sorted_tokens"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "sorted_tokens"]):
        blocks[f"{key[0]}|s|{key[1]}"] = set(group["entity_id"])
    print(f"  [{label}]   -> {len(blocks):,} keys")

    print(f"  [{label}] block by core tokens...")
    valid = df[df["core_tokens"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "core_tokens"]):
        blocks[f"{key[0]}|k|{key[1]}"] = set(group["entity_id"])
    print(f"  [{label}]   -> {len(blocks):,} keys")

    print(f"  [{label}] block by street number...")
    valid = df[df["street_num"].str.len() >= 3]
    for key, group in valid.groupby(["country_l", "street_num"]):
        blocks[f"{key[0]}|n|{key[1]}"] = set(group["entity_id"])

    print(f"  [{label}] {len(blocks):,} blocks built in {time.time()-t0:.1f}s")
    return blocks


def generate_candidates(s1_df, s2_df, s3_df, max_candidates=500):
    if "norm_name" not in s2_df.columns:
        s2_df = enrich(s2_df, label="S2")
    if "norm_name" not in s3_df.columns:
        s3_df = enrich(s3_df, label="S3")
    if "norm_name" not in s1_df.columns:
        s1_df = enrich(s1_df, label="S1")

    print("Building S2 blocks...")
    b2 = build_blocks(s2_df, label="S2")
    print("Building S3 blocks...")
    b3 = build_blocks(s3_df, label="S3")

    print("Preparing S1 keys...")
    s1 = _prepare_keys(s1_df)

    print(f"Generating candidates for {len(s1):,} S1 entities...")
    t0 = time.time()
    out = {}

    b2_get = b2.get
    b3_get = b3.get

    for i, row in enumerate(s1.itertuples(index=False)):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(s1) - i) / rate if rate > 0 else 0
            print(f"  ... {i:,}/{len(s1):,}  ({elapsed:.1f}s, ETA {eta:.0f}s)")

        sid = row.entity_id
        c = row.country_l

        strong = set()
        if row.postal3:
            strong |= b2_get(f"{c}|p|{row.postal3}", set())
            strong |= b3_get(f"{c}|p|{row.postal3}", set())
        if row.sorted_tokens:
            strong |= b2_get(f"{c}|s|{row.sorted_tokens}", set())
            strong |= b3_get(f"{c}|s|{row.sorted_tokens}", set())
        if row.core_tokens:
            strong |= b2_get(f"{c}|k|{row.core_tokens}", set())
            strong |= b3_get(f"{c}|k|{row.core_tokens}", set())

        medium = set()
        if row.first_token:
            medium |= b2_get(f"{c}|t|{row.first_token}", set())
            medium |= b3_get(f"{c}|t|{row.first_token}", set())

        weak = set()
        if row.first2:
            weak |= b2_get(f"{c}|c|{row.first2}", set())
            weak |= b3_get(f"{c}|c|{row.first2}", set())
        if row.street_num and len(row.street_num) >= 3:
            weak |= b2_get(f"{c}|n|{row.street_num}", set())
            weak |= b3_get(f"{c}|n|{row.street_num}", set())

        # Prioritize: strong → medium → weak
        cand = list(strong)
        if len(cand) < max_candidates:
            cand += [x for x in medium if x not in strong]
        if len(cand) < max_candidates:
            cand += [x for x in weak if x not in strong and x not in medium]

        out[sid] = set(cand[:max_candidates])

    print(f"  Done in {time.time()-t0:.1f}s")
    return out, s1, s2_df, s3_df