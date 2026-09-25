import sys
sys.path.insert(0, ".")

import pandas as pd
from src.normalize import normalize_name, normalize_address, extract_postal

s1 = pd.read_csv("data/train/train_source1.tsv", sep="\t", nrows=20000)
s2 = pd.read_csv("data/train/train_source2.tsv", sep="\t", nrows=20000)
gt = pd.read_csv("data/train/train_ground_truth.tsv", sep="\t", nrows=20000)

gt["has_match"] = (
    gt["matched_entity_ids"].notna() &
    (gt["matched_entity_ids"].astype(str).str.strip() != "")
)
gt2 = gt[gt["has_match"]].head(10)

shown = 0
for _, row in gt2.iterrows():
    s1_id = row["source1_entity_id"]
    match_ids = str(row["matched_entity_ids"]).split(",")
    s2_ids = [m for m in match_ids if m.startswith("S2-")]
    if not s2_ids:
        continue
    s2_id = s2_ids[0]

    s1_row = s1[s1["entity_id"] == s1_id]
    s2_row = s2[s2["entity_id"] == s2_id]
    if len(s1_row) == 0 or len(s2_row) == 0:
        continue

    s1r = s1_row.iloc[0]
    s2r = s2_row.iloc[0]

    s1_norm = normalize_name(s1r["business_name"])
    s2_norm = normalize_name(s2r["business_name"])

    s1_tokens = s1_norm.split()
    s2_tokens = s2_norm.split()

    s1_first = s1_tokens[0] if s1_tokens else ""
    s2_first = s2_tokens[0] if s2_tokens else ""

    s1_sorted = " ".join(sorted(s1_tokens))[:40]
    s2_sorted = " ".join(sorted(s2_tokens))[:40]

    print("=" * 70)
    print(s1_id, " vs ", s2_id)
    print("  S1 country     :", repr(s1r["country"]))
    print("  S2 country     :", repr(s2r["country"]))
    print("  S1 raw name    :", repr(s1r["business_name"]))
    print("  S2 raw name    :", repr(s2r["business_name"]))
    print("  S1 norm name   :", repr(s1_norm))
    print("  S2 norm name   :", repr(s2_norm))
    print("  S1 first token :", repr(s1_first))
    print("  S2 first token :", repr(s2_first))
    print("  S1 sorted(40)  :", repr(s1_sorted))
    print("  S2 sorted(40)  :", repr(s2_sorted))
    print("  Sorted match?  :", s1_sorted == s2_sorted)
    print("  First match?   :", s1_first == s2_first)
    print("  S1 postal      :", repr(extract_postal(s1r["business_address"])))
    print("  S2 postal      :", repr(extract_postal(s2r["business_address"])))

    shown += 1
    if shown >= 3:
        break

print()
print("Shown", shown, "examples")