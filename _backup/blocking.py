import time
from collections import defaultdict
import pandas as pd


def build_blocks(df, label="S"):
    """Vectorized block builder — no iterrows()."""
    print(f"  [{label}] preparing block keys...")
    t0 = time.time()

    df = df.copy()
    df["country_l"] = df["country"].astype(str).str.strip().str.lower()
    df["postal3"] = df["postal"].astype(str).str[:3]
    df["first_token"] = df["norm_name"].astype(str).str.split().str[0].fillna("")
    df["first2"] = df["norm_name"].astype(str).str[:2]

    blocks = {}

    print(f"  [{label}] block by postal prefix...")
    valid = df[df["postal3"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "postal3"]):
        blocks[f"{key[0]}|p|{key[1]}"] = set(group["entity_id"])

    print(f"  [{label}] block by first token...")
    valid = df[df["first_token"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "first_token"]):
        blocks[f"{key[0]}|t|{key[1]}"] = set(group["entity_id"])

    print(f"  [{label}] block by first 2 chars...")
    valid = df[df["first2"].str.len() > 0]
    for key, group in valid.groupby(["country_l", "first2"]):
        blocks[f"{key[0]}|c|{key[1]}"] = set(group["entity_id"])

    print(f"  [{label}] built {len(blocks):,} blocks in {time.time()-t0:.1f}s")
    return blocks


def generate_candidates(s1_df, s2_df, s3_df, max_candidates=200):
    print("Building S2 blocks...")
    b2 = build_blocks(s2_df, label="S2")
    print("Building S3 blocks...")
    b3 = build_blocks(s3_df, label="S3")

    print("Preparing S1 keys...")
    s1 = s1_df.copy()
    s1["country_l"] = s1["country"].astype(str).str.strip().str.lower()
    s1["postal3"] = s1["postal"].astype(str).str[:3]
    s1["first_token"] = s1["norm_name"].astype(str).str.split().str[0].fillna("")
    s1["first2"] = s1["norm_name"].astype(str).str[:2]

    print(f"Generating candidates for {len(s1):,} S1 entities...")
    t0 = time.time()
    out = {}
    for i, (_, r) in enumerate(s1.iterrows()):
        if i % 500 == 0 and i > 0:
            print(f"  ... {i:,}/{len(s1):,}  ({time.time()-t0:.1f}s)")

        sid = r["entity_id"]
        c = r["country_l"]

        cand = set()
        if r["postal3"]:
            cand |= b2.get(f"{c}|p|{r['postal3']}", set())
            cand |= b3.get(f"{c}|p|{r['postal3']}", set())
        if r["first_token"]:
            cand |= b2.get(f"{c}|t|{r['first_token']}", set())
            cand |= b3.get(f"{c}|t|{r['first_token']}", set())
        if r["first2"]:
            cand |= b2.get(f"{c}|c|{r['first2']}", set())
            cand |= b3.get(f"{c}|c|{r['first2']}", set())

        out[sid] = set(list(cand)[:max_candidates])

    print(f"  Candidate generation done in {time.time()-t0:.1f}s")
    return out