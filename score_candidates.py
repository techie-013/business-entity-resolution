"""
Optimized ML scoring of candidates.
Batch predicts instead of per-row → 50× faster.
"""
import sys
import time
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import joblib

from src.features import compute_features

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
MAX_PER_S1 = 3       # cap candidates per S1 (was 20)
THR = 0.80           # ML probability threshold
BATCH = 10_000       # batch size for model scoring
MAX_HITS_FINAL = 5   # final matches per S1


# ═══════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════
log("Loading model...")
pkg = joblib.load("model.pkl")
model = pkg["model"]
FEATS = pkg["feats"]
log(f"  Features: {len(FEATS)}")

log("Loading cached test data...")
s1 = pd.read_pickle("data/test/test_source1.tsv.pkl")
s2 = pd.read_pickle("data/test/test_source2.tsv.pkl")
s3 = pd.read_pickle("data/test/test_source3.tsv.pkl")
log(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

log("Loading candidate pairs...")
cand_df = pd.read_csv("output/candidate_pairs.tsv", sep="\t")
log(f"  {len(cand_df):,} S1 rows")


# ═══════════════════════════════════════════════════
# EXPLODE (capped at MAX_PER_S1)
# ═══════════════════════════════════════════════════
log(f"Exploding pairs (capped at {MAX_PER_S1} per S1)...")
rows = []
for s1_id, matched in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"].fillna("")):
    if not matched:
        continue
    for cid in matched.split(",")[:MAX_PER_S1]:
        if cid:
            rows.append((s1_id, cid))
pairs = pd.DataFrame(rows, columns=["s1_id", "cand_id"])
del rows  # free memory
log(f"  {len(pairs):,} pairs to score")


# ═══════════════════════════════════════════════════
# INDEX (fast lookup via dict, not .loc)
# ═══════════════════════════════════════════════════
log("Building lookup dicts...")

def df_to_dict(df, cols):
    """Return dict: entity_id -> dict of cols."""
    return df.set_index("entity_id")[cols].to_dict(orient="index")

s1_lookup = df_to_dict(s1, ["norm_name", "norm_addr", "country_l", "postal"])
s2_lookup = df_to_dict(s2, ["norm_name", "norm_addr", "country_l", "postal"])
s3_lookup = df_to_dict(s3, ["norm_name", "norm_addr", "country_l", "postal"])
log(f"  Dicts built")


# ═══════════════════════════════════════════════════
# COMPUTE FEATURES IN BATCHES
# ═══════════════════════════════════════════════════
log(f"Computing features in batches of {BATCH:,}...")

all_features = []
all_valid_idx = []   # indices in `pairs` where features computed OK

s1_ids = pairs["s1_id"].values
cand_ids = pairs["cand_id"].values
n_pairs = len(pairs)

for start in range(0, n_pairs, BATCH):
    end = min(start + BATCH, n_pairs)
    if start % 500_000 == 0:
        log(f"  ... {start:,}/{n_pairs:,}")

    batch_feats = []
    batch_idx = []

    for i in range(start, end):
        s1_id = s1_ids[i]
        cid = cand_ids[i]

        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue

        if cid.startswith("S2-"):
            cand_row = s2_lookup.get(cid)
        else:
            cand_row = s3_lookup.get(cid)
        if cand_row is None:
            continue

        try:
            f = compute_features(s1_row, cand_row)
            batch_feats.append(f)
            batch_idx.append(i)
        except Exception:
            continue

    if batch_feats:
        all_features.extend(batch_feats)
        all_valid_idx.extend(batch_idx)

log(f"  Computed {len(all_features):,} feature rows")


# ═══════════════════════════════════════════════════
# BATCH PREDICT (ONE call, not 4.5M calls)
# ═══════════════════════════════════════════════════
log("Batch predicting...")
X_all = pd.DataFrame(all_features)

# Keep only features the model knows about
cols = [c for c in FEATS if c in X_all.columns]
missing = [c for c in FEATS if c not in X_all.columns]
if missing:
    log(f"  WARNING: missing features: {missing}")
    for c in missing:
        X_all[c] = 0.0
    cols = FEATS

probs = model.predict_proba(X_all[cols])[:, 1]
log(f"  Predicted {len(probs):,} probabilities")


# ═══════════════════════════════════════════════════
# FILTER + GROUP BY S1
# ═══════════════════════════════════════════════════
log(f"Filtering (threshold={THR})...")
keep_mask = probs >= THR
kept_idx = np.array(all_valid_idx)[keep_mask]
kept_probs = probs[keep_mask]
kept_cands = cand_ids[kept_idx]
kept_s1 = s1_ids[kept_idx]

log(f"  {keep_mask.sum():,} pairs pass threshold")

# Group by S1
log("Grouping by S1...")
from collections import defaultdict
hits_by_s1 = defaultdict(list)
for s1_id, cid, p in zip(kept_s1, kept_cands, kept_probs):
    hits_by_s1[s1_id].append((cid, float(p)))


# ═══════════════════════════════════════════════════
# WRITE OUTPUT (preserve all test S1 rows)
# ═══════════════════════════════════════════════════
log("Writing output...")
all_s1_ids = s1["entity_id"].values
out_rows = []
for sid in all_s1_ids:
    hits = hits_by_s1.get(sid, [])
    hits.sort(key=lambda x: -x[1])
    top = [h[0] for h in hits[:MAX_HITS_FINAL]]
    out_rows.append({
        "source1_entity_id": sid,
        "matched_entity_ids": ",".join(top) if top else "",
    })

out_df = pd.DataFrame(out_rows)
out_df.to_csv("output/matching_results_ml.tsv", sep="\t", index=False)

n_with = (out_df["matched_entity_ids"] != "").sum()
log(f"  Rows: {len(out_df):,}")
log(f"  With matches: {n_with:,} ({n_with/len(out_df)*100:.2f}%)")
log("DONE — output saved to output/matching_results_ml.tsv")