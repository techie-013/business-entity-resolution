"""
Optimized blocking:
- 3 keys only (exact name, sorted tokens, first token)
- Drops any block with >MAX_BUCKET members (kills "the", "shri", "john" blowups)
- Cap candidates per S1
- Vectorized where possible
"""
import time
import pandas as pd
from src.normalize import normalize_name, normalize_address, extract_postal

# 🚨 THE KEY CONSTANTS
MAX_BUCKET_SIZE = 50   # reject blocks with more than this many members
MAX_CANDIDATES_PER_S1 = 20
MAX_HITS_PER_S1 = 5


def prep_keys(df):
    """Build 3 block keys."""
    df = df.copy()
    nm = df["norm_name"].astype(str)
    c = df["country_l"]

    df["key_exact"] = c + "|E|" + nm
    df["key_sorted"] = c + "|S|" + nm.str.split().apply(
        lambda t: " ".join(sorted(t)) if isinstance(t, list) and t else ""
    )
    df["key_first"] = c + "|F|" + nm.str.split().str[0].fillna("")

    return df


def build_capped_blocks(df, label="S"):
    """Build inverted indices with hard size cap."""
    print(f"  [{label}] building blocks...", flush=True)
    t0 = time.time()

    blocks = {}
    for key_col in ("key_exact", "key_sorted", "key_first"):
        g = df.groupby(key_col, sort=False)["entity_id"].apply(list)
        total = len(g)
        g = g[g.apply(len) <= MAX_BUCKET_SIZE]
        dropped = total - len(g)
        blocks[key_col] = g.to_dict()
        print(f"  [{label}] {key_col}: kept {len(blocks[key_col]):,} "
              f"(dropped {dropped:,} oversized)", flush=True)

    print(f"  [{label}] done in {time.time()-t0:.1f}s", flush=True)
    return blocks


def generate_candidates(s1_df, s2_df, s3_df,
                        max_candidates=MAX_CANDIDATES_PER_S1):
    """
    Generate candidates for each S1 entity.
    Priority: exact → sorted → first-token.
    """
    print("Preparing S2/S3 blocks...", flush=True)
    s2 = prep_keys(s2_df) if "key_exact" not in s2_df.columns else s2_df
    s3 = prep_keys(s3_df) if "key_exact" not in s3_df.columns else s3_df

    b2 = build_capped_blocks(s2, "S2")
    b3 = build_capped_blocks(s3, "S3")

    print("Preparing S1 keys...", flush=True)
    s1 = prep_keys(s1_df)

    print(f"Generating candidates for {len(s1):,} S1...", flush=True)
    t0 = time.time()

    s1_e = s1["key_exact"].values
    s1_s = s1["key_sorted"].values
    s1_f = s1["key_first"].values

    g2e, g3e = b2["key_exact"].get, b3["key_exact"].get
    g2s, g3s = b2["key_sorted"].get, b3["key_sorted"].get
    g2f, g3f = b2["key_first"].get, b3["key_first"].get

    matches = [""] * len(s1)

    for i in range(len(s1)):
        if i % 200_000 == 0 and i > 0:
            print(f"  ... {i:,}/{len(s1):,}  ({time.time()-t0:.1f}s)", flush=True)

        hits = []

        # Tier 1: exact name
        k = s1_e[i]
        if k:
            v = g2e(k)
            if v: hits.extend(v)
            v = g3e(k)
            if v: hits.extend(v)

        # Tier 2: sorted tokens (word reorder)
        if len(hits) < MAX_HITS_PER_S1:
            k = s1_s[i]
            if k and len(k) > 10:
                v = g2s(k)
                if v: hits.extend(v)
                v = g3s(k)
                if v: hits.extend(v)

        # Tier 3: first token (only if token is meaningful)
        if len(hits) < MAX_HITS_PER_S1:
            k = s1_f[i]
            # require first token to be > 8 chars (avoids "us|F|a", "us|F|the")
            if k and len(k) > 8:
                v = g2f(k)
                if v: hits.extend(v)
                v = g3f(k)
                if v: hits.extend(v)

        # Dedupe + cap
        if hits:
            seen = set()
            final = []
            for h in hits:
                if h not in seen:
                    seen.add(h)
                    final.append(h)
                    if len(final) >= max_candidates:
                        break
            matches[i] = ",".join(final)

    print(f"  Matching done in {time.time()-t0:.1f}s", flush=True)
    return matches, s1