from rapidfuzz import fuzz
from src.normalize import tokenize


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_features(s1, cand):
    n1, n2 = s1["norm_name"], cand["norm_name"]
    a1, a2 = s1["norm_addr"], cand["norm_addr"]
    t1n, t2n = tokenize(n1), tokenize(n2)
    t1a, t2a = tokenize(a1), tokenize(a2)

    return {
        "name_jaccard": jaccard(t1n, t2n),
        "name_lev": fuzz.ratio(n1, n2) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_partial": fuzz.partial_ratio(n1, n2) / 100.0,
        "addr_jaccard": jaccard(t1a, t2a),
        "addr_lev": fuzz.ratio(a1, a2) / 100.0,
        "addr_token_sort": fuzz.token_sort_ratio(a1, a2) / 100.0,
        "addr_partial": fuzz.partial_ratio(a1, a2) / 100.0,
        "country_match": int(s1["country"] == cand["country"]),
        "postal_match": int(bool(s1["postal"]) and s1["postal"] == cand["postal"]),
        "first_token_match": int(bool(t1n) and bool(t2n) and (sorted(t1n)[0] == sorted(t2n)[0])),
        "name_len_ratio": min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
        "addr_len_ratio": min(len(a1), len(a2)) / max(len(a1), len(a2), 1),
    }


def build_feature_matrix(s1_df, s2_df, s3_df, candidates_dict):
    s1_idx = s1_df.set_index("entity_id")
    s2_idx = s2_df.set_index("entity_id")
    s3_idx = s3_df.set_index("entity_id")

    rows = []
    for s1_id, cand_ids in candidates_dict.items():
        if s1_id not in s1_idx.index:
            continue
        s1_row = s1_idx.loc[s1_id]
        for cid in cand_ids:
            src = s2_idx if cid.startswith("S2-") else s3_idx
            if cid not in src.index:
                continue
            feats = compute_features(s1_row, src.loc[cid])
            feats["s1_id"] = s1_id
            feats["cand_id"] = cid
            rows.append(feats)
    return rows