import time
import pandas as pd
from src.normalize import normalize_name, normalize_address, extract_postal

LEGAL = {"incorporated", "limited", "company", "corporation",
         "private", "llc", "inc", "ltd", "corp", "pvt", "co",
         "sarl", "sas", "sa", "eurl", "snc", "pllc", "lp", "plc"}


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
    df["first3"] = df["norm_name"].astype(str).str[:3]
    df["sorted_tokens"] = df["norm_name"].apply(
        lambda x: " ".join(sorted(str(x).split())) if x else ""
    )
    df["core_tokens"] = df["norm_name"].apply(
        lambda x: " ".join(sorted(t for t in str(x).split() if t not in LEGAL))
        if x else ""
    )
    df["street_num"] = (
        df["norm_addr"].astype(str)
        .str.extract(r"(\b\d{2,5}\b)", expand=False)
        .fillna("")
    )
    return df


def build_blocks(df, label="S"):
    print(f"  [{label}] preparing keys...", flush=True)
    t0 = time.time()
    df = _prepare_keys(df)

    blocks = {}

    keys = [
        ("p", "postal3", 1),
        ("t", "first_token", 1),
        ("f", "first3", 1),
        ("s", "sorted_tokens", 1),
        ("k", "core_tokens", 1),
        ("n", "street_num", 3),
    ]
    for tag, col, min_len in keys:
        if col == "street_num":
            valid = df[df[col].astype(str).str.len() >= min_len]
        else:
            valid = df[df[col].astype(str).str.len() >= min_len]
        for k, g in valid.groupby(["country_l", col]):
            blocks[f"{k[0]}|{tag}|{k[1]}"] = set(g["entity_id"])
        print(f"  [{label}] after {tag}: {len(blocks):,} keys", flush=True)

    print(f"  [{label}] total: {len(blocks):,} blocks in {time.time()-t0:.1f}s", flush=True)
    return blocks


def generate_candidates(s1_df, s2_df, s3_df, max_candidates=300):
    if "norm_name" not in s2_df.columns:
        s2_df = _prepare_keys(s2_df)
    if "norm_name" not in s3_df.columns:
        s3_df = _prepare_keys(s3_df)

    print("Building S2 blocks...", flush=True)
    b2 = build_blocks(s2_df, "S2")
    print("Building S3 blocks...", flush=True)
    b3 = build_blocks(s3_df, "S3")

    print("Preparing S1 keys...", flush=True)
    s1 = _prepare_keys(s1_df)

    print(f"Generating candidates for {len(s1):,} S1...", flush=True)
    t0 = time.time()
    out = {}

    for i, row in enumerate(s1.itertuples(index=False)):
        if i % 50000 == 0 and i > 0:
            print(f"  ... {i:,}/{len(s1):,} ({time.time()-t0:.1f}s)", flush=True)

        sid = row.entity_id
        c = row.country_l

        strong = set()
        if row.postal3:
            strong |= b2.get(f"{c}|p|{row.postal3}", set())
            strong |= b3.get(f"{c}|p|{row.postal3}", set())
        if row.sorted_tokens:
            strong |= b2.get(f"{c}|s|{row.sorted_tokens}", set())
            strong |= b3.get(f"{c}|s|{row.sorted_tokens}", set())
        if row.core_tokens:
            strong |= b2.get(f"{c}|k|{row.core_tokens}", set())
            strong |= b3.get(f"{c}|k|{row.core_tokens}", set())

        medium = set()
        if row.first_token:
            medium |= b2.get(f"{c}|t|{row.first_token}", set())
            medium |= b3.get(f"{c}|t|{row.first_token}", set())
        if row.first3:
            medium |= b2.get(f"{c}|f|{row.first3}", set())
            medium |= b3.get(f"{c}|f|{row.first3}", set())

        weak = set()
        if row.street_num and len(row.street_num) >= 3:
            weak |= b2.get(f"{c}|n|{row.street_num}", set())
            weak |= b3.get(f"{c}|n|{row.street_num}", set())

        cand = list(strong)
        if len(cand) < max_candidates:
            cand += [x for x in medium if x not in strong]
        if len(cand) < max_candidates:
            cand += [x for x in weak if x not in strong and x not in medium]

        out[sid] = set(cand[:max_candidates])

    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    return out, s1, s2_df, s3_df