import sys, time
sys.path.insert(0, ".")

from src.data_loader import load_train, parse_ground_truth
from src.blocking import generate_candidates

t0 = time.time()
print("Loading data...")
s1, s2, s3, gt = load_train("data/train")
print(f"  Loaded in {time.time()-t0:.1f}s")

gt_map = parse_ground_truth(gt)
sample_s1 = s1.sample(n=500, random_state=42)
sample_ids = set(sample_s1["entity_id"])
sample_gt = {k: v for k, v in gt_map.items() if k in sample_ids}
print(f"  Sample: {len(sample_s1)} S1, {sum(1 for t in sample_gt.values() if t)} with matches")

t1 = time.time()
cands, s1e, s2e, s3e = generate_candidates(sample_s1, s2, s3, max_candidates=5000)
print(f"  Gen time: {time.time()-t1:.1f}s")

hits = sum(1 for sid, tids in sample_gt.items() if sid in cands and tids & cands[sid])
total = sum(1 for sid in sample_gt if len(sample_gt[sid]) > 0)

if total:
    print(f"Blocking recall: {hits/total:.3f}  ({hits}/{total})")

import numpy as np
counts = [len(c) for c in cands.values()]
print(f"Candidates per S1 — min: {min(counts)} | median: {int(np.median(counts))} | max: {max(counts)}")

print(f"Total time: {time.time()-t0:.1f}s")
print("Done.")