import sys, re, unicodedata, time
sys.path.insert(0, ".")
import pandas as pd

ABBREV = {
    "corp":"corporation","co":"company","ltd":"limited","inc":"incorporated",
    "llc":"llc","pvt":"private","pte":"private",
    "rd":"road","st":"street","ave":"avenue","av":"avenue","blvd":"boulevard",
    "dr":"drive","ln":"lane","sq":"square","hwy":"highway",
    "no":"number","&":"and",
}

def norm(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(ABBREV.get(t, t) for t in s.split())

def postal(s):
    if pd.isna(s): return ""
    m = re.search(r"\b(\d{5})(?:-?\d{4})?\b", str(s))
    return m.group(1) if m else ""

def core(s):
    LEGAL = {"incorporated","limited","company","corporation","private",
             "llc","inc","ltd","corp","pvt","co","sarl","sas","sa","pllc","lp","plc"}
    return " ".join(sorted(t for t in s.split() if t not in LEGAL))

print("=" * 60)
print("STEP 1: Loading test files...")
t0 = time.time()
s1 = pd.read_csv("data/test/test_source1.tsv", sep="\t")
s2 = pd.read_csv("data/test/test_source2.tsv", sep="\t")
s3 = pd.read_csv("data/test/test_source3.tsv", sep="\t")
print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  ({time.time()-t0:.1f}s)")

print("STEP 2: Normalizing names + extracting postals...")
t0 = time.time()
s1["nn"] = s1["business_name"].map(norm)
s2["nn"] = s2["business_name"].map(norm)
s3["nn"] = s3["business_name"].map(norm)
s1["p"] = s1["business_address"].map(postal)
s2["p"] = s2["business_address"].map(postal)
s3["p"] = s3["business_address"].map(postal)
s1["c"] = s1["country"].astype(str).str.lower().str.strip()
s2["c"] = s2["country"].astype(str).str.lower().str.strip()
s3["c"] = s3["country"].astype(str).str.lower().str.strip()
print(f"  Normalized ({time.time()-t0:.1f}s)")

print("STEP 3: Building match keys...")
t0 = time.time()
# Multi-key: (country + full normalized name) AND (country + sorted tokens) AND (country + core tokens)
s2["k_full"] = s2["c"] + "|" + s2["nn"]
s3["k_full"] = s3["c"] + "|" + s3["nn"]
s1["k_full"] = s1["c"] + "|" + s1["nn"]

s2["k_sort"] = s2["c"] + "|" + s2["nn"].apply(lambda x: " ".join(sorted(x.split())) if x else "")
s3["k_sort"] = s3["c"] + "|" + s3["nn"].apply(lambda x: " ".join(sorted(x.split())) if x else "")
s1["k_sort"] = s1["c"] + "|" + s1["nn"].apply(lambda x: " ".join(sorted(x.split())) if x else "")

s2["k_core"] = s2["c"] + "|" + s2["nn"].apply(core)
s3["k_core"] = s3["c"] + "|" + s3["nn"].apply(core)
s1["k_core"] = s1["c"] + "|" + s1["nn"].apply(core)
print(f"  Keys ready ({time.time()-t0:.1f}s)")

print("STEP 4: Building group dictionaries...")
t0 = time.time()
s2_full = s2.groupby("k_full")["entity_id"].apply(list).to_dict()
s3_full = s3.groupby("k_full")["entity_id"].apply(list).to_dict()
s2_sort = s2.groupby("k_sort")["entity_id"].apply(list).to_dict()
s3_sort = s3.groupby("k_sort")["entity_id"].apply(list).to_dict()
s2_core = s2.groupby("k_core")["entity_id"].apply(list).to_dict()
s3_core = s3.groupby("k_core")["entity_id"].apply(list).to_dict()
print(f"  Groups built ({time.time()-t0:.1f}s)")

print("STEP 5: Matching (prioritized by key quality)...")
t0 = time.time()
matches = []
for i, row in enumerate(s1.itertuples(index=False)):
    if i % 200000 == 0 and i > 0:
        print(f"  ... {i:,}/{len(s1):,}  ({time.time()-t0:.1f}s)")

    sid = row.entity_id
    kf = row.k_full
    ks = row.k_sort
    kc = row.k_core

    # Tier 1: exact name match (strongest signal)
    hits = list(s2_full.get(kf, [])) + list(s3_full.get(kf, []))
    # Tier 2: sorted tokens (catches reorders) — only if not enough hits
    if len(hits) < 3:
        extra = list(s2_sort.get(ks, [])) + list(s3_sort.get(ks, []))
        hits += [h for h in extra if h not in hits]
    # Tier 3: core tokens (removes legal suffixes) — only if still sparse
    if len(hits) < 3:
        extra = list(s2_core.get(kc, [])) + list(s3_core.get(kc, []))
        hits += [h for h in extra if h not in hits]

    matches.append({
        "source1_entity_id": sid,
        "matched_entity_ids": ",".join(hits[:5]) if hits else ""
    })

print(f"  Matched ({time.time()-t0:.1f}s)")

print("STEP 6: Writing output files...")
df = pd.DataFrame(matches)
df.to_csv("output/matching_results.tsv", sep="\t", index=False)
df.rename(columns={"matched_entity_ids": "candidate_entity_ids"}).to_csv(
    "output/candidate_pairs.tsv", sep="\t", index=False
)
n_with = (df["matched_entity_ids"] != "").sum()
print(f"  Wrote {len(df):,} rows")
print(f"  Entities with matches: {n_with:,} ({n_with/len(df)*100:.1f}%)")
print("=" * 60)
print("DONE — Now validate and upload:")
print("  python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir data/test")
print("=" * 60)
