import sys, time
sys.path.insert(0, ".")

print("Loading ground truth only...")
import pandas as pd
gt = pd.read_csv("data/train/train_ground_truth.tsv", sep="\t")
print(f"  GT: {len(gt):,} rows")

print("Loading S1...")
s1 = pd.read_csv("data/train/train_source1.tsv", sep="\t")
print(f"  S1: {len(s1):,} rows")

print("Loading S2 only (skip S3 — we just need one true match)...")
s2 = pd.read_csv("data/train/train_source2.tsv", sep="\t")
print(f"  S2: {len(s2):,} rows")

# Take a small sample of S1 to make enrichment fast
print("Sampling S1 to 100 rows for debug...")
s1_sample = s1.sample(n=100, random_state=42)

print("Enriching S1 sample...")
from src.blocking import enrich
s1_e = enrich(s1_sample, "S1-sample")
print(f"  Enriched S1: {len(s1_e)}")

# Parse ground truth
print("Parsing GT...")
from src.data_loader import parse_ground_truth
gt_map = parse_ground_truth(gt)
print(f"  GT map: {len(gt_map):,} entries")

# Only enrich the S2 rows that matter
print("\nFinding true matches for the S1 sample...")
found = 0
for _, s1_row in s1_e.iterrows():
    s1_id = s1_row["entity_id"]
    true_ids = gt_map.get(s1_id, set())
    if not true_ids:
        continue

    found += 1
    print(f"\n===== S1: {s1_id} =====")
    print(f"  name      : {s1_row['business_name']}")
    print(f"  norm_name : {s1_row['norm_name']}")
    print(f"  addr      : {s1_row['business_address']}")
    print(f"  norm_addr : {s1_row['norm_addr']}")
    print(f"  country   : {s1_row['country']}")
    print(f"  postal    : {s1_row['postal']}")
    print(f"  true_ids  : {true_ids}")

    for tid in list(true_ids)[:2]:
        src = s2 if tid.startswith("S2-") else None
        if src is None:
            print(f"  {tid}: (S3 — skipped in this debug)")
            continue
        row_df = src[src.entity_id == tid]
        if len(row_df) == 0:
            print(f"  {tid}: NOT IN S2")
            continue
        row = row_df.iloc[0]
        print(f"  --- {tid} ---")
        print(f"    name      : {row['business_name']}")
        print(f"    addr      : {row['business_address']}")
        print(f"    country   : {row['country']}")

    if found >= 3:
        break

print(f"\nDone. Found {found} S1 entities with true matches in the sample.")