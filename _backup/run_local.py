import sys
import time
sys.path.insert(0, ".")

from src.data_loader import load_train, parse_ground_truth
from src.blocking import generate_candidates

# ─── Load ─────────────────────────────────────
t0 = time.time()
print("Loading data...")
s1, s2, s3, gt = load_train("data/train")
print(f"  Loaded in {time.time() - t0:.1f}s")
print(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,} | GT: {len(gt):,}")

# ─── Reduce S2/S3 for fast local test ─────────
print("Sampling S2/S3 to 300K each...")
s2 = s2.sample(n=min(300_000, len(s2)), random_state=42)
s3 = s3.sample(n=min(300_000, len(s3)), random_state=42)
print(f"  Sampled S2={len(s2):,} | S3={len(s3):,}")

# ─── Ground truth ─────────────────────────────
gt_map = parse_ground_truth(gt)

# ─── Sample S1 ────────────────────────────────
sample_s1 = s1.sample(n=2000, random_state=42)
sample_ids = set(sample_s1["entity_id"])
sample_gt = {k: v for k, v in gt_map.items() if k in sample_ids}
print(f"  S1 sample: {len(sample_s1):,}")

# ─── Generate candidates ──────────────────────
t1 = time.time()
print("Generating candidates...")
cands = generate_candidates(sample_s1, s2, s3, max_candidates=200)
print(f"  Candidate gen took {time.time() - t1:.1f}s")

# ─── Blocking recall ──────────────────────────
hits = sum(
    1 for sid, tids in sample_gt.items()
    if sid in cands and tids & cands[sid]
)
total = sum(1 for sid in sample_gt if len(sample_gt[sid]) > 0)

if total > 0:
    print(f"Blocking recall: {hits / total:.3f}  ({hits}/{total})")
else:
    print("No matches in sample (skip recall)")

print(f"Total time: {time.time() - t0:.1f}s")
print("Done.")