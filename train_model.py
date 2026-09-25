import sys, time
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

from src.data_loader import load_train, parse_ground_truth
from src.blocking import generate_candidates

print("=" * 60)
print("STEP 1: Loading training data...")
t0 = time.time()
s1, s2, s3, gt = load_train("data/train")
print(f"  Loaded in {time.time()-t0:.1f}s")

# Sample smaller for speed
s1_sample = s1.sample(n=50000, random_state=42)
s2_sample = s2.sample(n=500000, random_state=42)
s3_sample = s3.sample(n=500000, random_state=42)
print(f"  Sampling: S1={len(s1_sample):,} S2={len(s2_sample):,} S3={len(s3_sample):,}")

gt_map = parse_ground_truth(gt)

print("=" * 60)
print("STEP 2: Generating candidates...")
t0 = time.time()
cands, s1e, s2e, s3e = generate_candidates(s1_sample, s2_sample, s3_sample, max_candidates=200)
print(f"  Candidate gen: {time.time()-t0:.1f}s")

# Compute blocking recall
sample_ids = set(s1_sample["entity_id"])
sample_gt = {k: v for k, v in gt_map.items() if k in sample_ids}
hits = sum(1 for sid, tids in sample_gt.items() if sid in cands and tids & cands[sid])
total = sum(1 for sid in sample_gt if len(sample_gt[sid]) > 0)
print(f"  Blocking recall: {hits/total:.3f}  ({hits}/{total})")

print("=" * 60)
print("STEP 3: Building feature matrix...")
t0 = time.time()

s2_idx = s2e.set_index("entity_id")
s3_idx = s3e.set_index("entity_id")
s1_idx = s1e.set_index("entity_id")

def feat(s1r, cr):
    n1, n2 = str(s1r["norm_name"]), str(cr["norm_name"])
    a1, a2 = str(s1r["norm_addr"]), str(cr["norm_addr"])
    t1n = set(n1.split()); t2n = set(n2.split())
    t1a = set(a1.split()); t2a = set(a2.split())
    def jac(x, y):
        if not x or not y: return 0.0
        return len(x & y) / len(x | y)
    return {
        "name_jaccard": jac(t1n, t2n),
        "name_lev": fuzz.ratio(n1, n2) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_partial": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_token_set": fuzz.token_set_ratio(n1, n2) / 100.0,
        "addr_jaccard": jac(t1a, t2a),
        "addr_lev": fuzz.ratio(a1, a2) / 100.0,
        "addr_token_sort": fuzz.token_sort_ratio(a1, a2) / 100.0,
        "addr_partial": fuzz.partial_ratio(a1, a2) / 100.0,
        "country_match": int(str(s1r["country_l"]) == str(cr["country_l"])),
        "postal_match": int(bool(s1r["postal"]) and str(s1r["postal"]) == str(cr["postal"])),
    }

rows = []
for i, s1_id in enumerate(cands.keys()):
    if i % 5000 == 0:
        print(f"  features {i}/{len(cands)}  ({time.time()-t0:.1f}s)")
    if s1_id not in s1_idx.index:
        continue
    s1r = s1_idx.loc[s1_id]
    true_ids = gt_map.get(s1_id, set())
    for cid in list(cands[s1_id])[:100]:
        src = s2_idx if cid.startswith("S2-") else s3_idx
        if cid not in src.index:
            continue
        cr = src.loc[cid]
        f = feat(s1r, cr)
        f["s1_id"] = s1_id
        f["cand_id"] = cid
        f["label"] = int(cid in true_ids)
        rows.append(f)

X = pd.DataFrame(rows)
print(f"  Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")
print(f"  Positive rate: {X['label'].mean():.3f}")

print("=" * 60)
print("STEP 4: Training LogReg...")
FEATS = [c for c in X.columns if c not in ("s1_id", "cand_id", "label")]
print(f"  Features ({len(FEATS)}): {FEATS}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

Xtr, Xva, ytr, yva = train_test_split(X[FEATS], X["label"], test_size=0.2, random_state=42, stratify=X["label"])

model = Pipeline([
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=200, n_jobs=-1))
])
model.fit(Xtr, ytr)
print("  Trained.")

print("=" * 60)
print("STEP 5: Threshold tuning...")
probs = model.predict_proba(Xva)[:, 1]

best_t, best_f = 0.5, -1
for t in np.arange(0.30, 0.91, 0.02):
    preds = (probs >= t).astype(int)
    tp = ((preds == 1) & (yva == 1)).sum()
    fp = ((preds == 1) & (yva == 0)).sum()
    fn = ((preds == 0) & (yva == 1)).sum()
    if tp == 0:
        continue
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    if p + r == 0:
        continue
    f05 = (1.25 * p * r) / (0.25 * p + r)
    if f05 > best_f:
        best_f, best_t = f05, t

print(f"  Best threshold: {best_t:.2f}  F0.5={best_f:.4f}")

import joblib
joblib.dump({"model": model, "feats": FEATS, "thr": float(best_t)}, "model.pkl")
print("  Saved model.pkl")
print("=" * 60)
print("DONE — Now run predict_submit.py")