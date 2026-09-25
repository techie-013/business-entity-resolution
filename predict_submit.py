import sys, time
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import joblib
from rapidfuzz import fuzz

from src.data_loader import load_test
from src.blocking import generate_candidates

print("Loading model...")
pkg = joblib.load("model.pkl")
model, FEATS, THR = pkg["model"], pkg["feats"], pkg["thr"]
print(f"  Threshold: {THR:.2f}")

print("Loading test data...")
t0 = time.time()
s1, s2, s3 = load_test("data/test")
print(f"  Loaded in {time.time()-t0:.1f}s")

print("Generating candidates...")
t0 = time.time()
cands, s1e, s2e, s3e = generate_candidates(s1, s2, s3, max_candidates=100)
print(f"  Cands: {time.time()-t0:.1f}s")

print("Building features + scoring...")
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

results = []
for i, s1_id in enumerate(cands.keys()):
    if i % 100000 == 0 and i > 0:
        print(f"  ... {i:,}/{len(cands):,}  ({time.time()-t0:.1f}s)")
    if s1_id not in s1_idx.index:
        results.append({"source1_entity_id": s1_id, "matched_entity_ids": ""})
        continue

    s1r = s1_idx.loc[s1_id]
    cids = list(cands[s1_id])[:100]
    hits = []
    for cid in cids:
        src = s2_idx if cid.startswith("S2-") else s3_idx
        if cid not in src.index:
            continue
        try:
            f = feat(s1r, src.loc[cid])
            prob = model.predict_proba(pd.DataFrame([f])[FEATS])[0, 1]
            if prob >= THR:
                hits.append(cid)
        except Exception:
            continue

    results.append({
        "source1_entity_id": s1_id,
        "matched_entity_ids": ",".join(hits[:5]) if hits else ""
    })

print(f"  Scored in {time.time()-t0:.1f}s")

print("Writing outputs...")
df = pd.DataFrame(results)
# Ensure all test S1 present
all_s1 = s1["entity_id"].tolist()
have = set(df["source1_entity_id"])
missing = [x for x in all_s1 if x not in have]
if missing:
    df = pd.concat([df, pd.DataFrame({"source1_entity_id": missing, "matched_entity_ids": ""})])

df.to_csv("output/matching_results.tsv", sep="\t", index=False)
df.rename(columns={"matched_entity_ids": "candidate_entity_ids"}).to_csv(
    "output/candidate_pairs.tsv", sep="\t", index=False
)
n_with = (df["matched_entity_ids"] != "").sum()
print(f"  Rows: {len(df):,} | With matches: {n_with:,} ({n_with/len(df)*100:.1f}%)")
print("=" * 60)