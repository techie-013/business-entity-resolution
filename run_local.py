import sys, time
sys.path.insert(0, ".")

from src.data_loader import load_train, parse_ground_truth
from src.blocking import _prepare_keys, build_blocks

t0 = time.time()
print("Loading data...")
s1, s2, s3, gt = load_train("data/train")
print(f"  Loaded in {time.time()-t0:.1f}s")

gt_map = parse_ground_truth(gt)
sample_s1 = s1.sample(n=500, random_state=42)   # smaller sample
sample_ids = set(sample_s1["entity_id"])
sample_gt = {k: v for k, v in gt_map.items() if k in sample_ids}
print(f"  Sample: {len(sample_s1)} S1, {sum(1 for t in sample_gt.values() if t)} with matches")

print("Building S2 blocks (no cap later)...")
b2 = build_blocks(s2, label="S2")
print("Building S3 blocks...")
b3 = build_blocks(s3, label="S3")

s1e = _prepare_keys(sample_s1)

print("Computing recall with NO cap...")
hits = 0
total = 0
for _, r in s1e.itertuples(index=False):
    sid = r.entity_id
    c = r.country_l

    cand = set()
    if r.postal3:
        cand |= b2.get(f"{c}|p|{r.postal3}", set())
        cand |= b3.get(f"{c}|p|{r.postal3}", set())
    if r.sorted_tokens:
        cand |= b2.get(f"{c}|s|{r.sorted_tokens}", set())
        cand |= b3.get(f"{c}|s|{r.sorted_tokens}", set())
    if r.core_tokens:
        cand |= b2.get(f"{c}|k|{r.core_tokens}", set())
        cand |= b3.get(f"{c}|k|{r.core_tokens}", set())
    if r.first_token:
        cand |= b2.get(f"{c}|t|{r.first_token}", set())
        cand |= b3.get(f"{c}|t|{r.first_token}", set())
    if r.first2:
        cand |= b2.get(f"{c}|c|{r.first2}", set())
        cand |= b3.get(f"{c}|c|{r.first2}", set())
    if r.street_num and len(r.street_num) >= 3:
        cand |= b2.get(f"{c}|n|{r.street_num}", set())
        cand |= b3.get(f"{c}|n|{r.street_num}", set())

    true_ids = sample_gt.get(sid, set())
    if not true_ids:
        continue
    total += 1
    if true_ids & cand:
        hits += 1

if total > 0:
    print(f"Blocking recall (NO cap): {hits/total:.3f}  ({hits}/{total})")
else:
    print("No matches in sample")

print(f"Total time: {time.time()-t0:.1f}s")