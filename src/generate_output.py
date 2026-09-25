import pandas as pd


def write_candidate_pairs(candidates: dict, path: str):
    rows = []
    for s1_id, cands in candidates.items():
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(set(cands))) if cands else ""
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"Wrote {len(rows)} rows → {path}")


def write_matching_results(matches: dict, all_s1_ids, path: str):
    rows = []
    for s1_id in all_s1_ids:
        m = sorted(set(matches.get(s1_id, set())))
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(m) if m else ""
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"Wrote {len(rows)} rows → {path}")