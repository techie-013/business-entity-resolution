"""
Fast end-to-end pipeline.
Loads cache → builds blocks → matches → writes output.
Target: 10-15 min on full test set.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd

from src.normalize import normalize_name, normalize_address, extract_postal
from src.blocking import generate_candidates, prep_keys


T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def load_cached(path, label):
    cache = Path(str(path) + ".pkl")
    if cache.exists():
        log(f"Loading cached {cache.name}...")
        return pd.read_pickle(cache)
    log(f"Reading {path.name}...")
    df = pd.read_csv(path, sep="\t")
    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_addr"] = df["business_address"].map(normalize_address)
    df["postal"] = df["business_address"].map(extract_postal)
    df["country_l"] = df["country"].astype(str).str.strip().str.lower()
    df.to_pickle(cache)
    return df


log("STEP 1: Load test data")
s1 = load_cached("data/test/test_source1.tsv", "S1")
s2 = load_cached("data/test/test_source2.tsv", "S2")
s3 = load_cached("data/test/test_source3.tsv", "S3")
log(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

log("STEP 2: Generate candidates")
matches, s1_enriched = generate_candidates(s1, s2, s3, max_candidates=20)

log("STEP 3: Write output")
out_df = pd.DataFrame({
    "source1_entity_id": s1_enriched["entity_id"].values,
    "matched_entity_ids": matches,
})
out_df.to_csv("output/matching_results.tsv", sep="\t", index=False)
out_df.rename(columns={"matched_entity_ids": "candidate_entity_ids"}).to_csv(
    "output/candidate_pairs.tsv", sep="\t", index=False
)

n_with = (out_df["matched_entity_ids"] != "").sum()
log(f"  Rows: {len(out_df):,}")
log(f"  With matches: {n_with:,} ({n_with/len(out_df)*100:.2f}%)")
log("DONE")