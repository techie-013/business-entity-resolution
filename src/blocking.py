<<<<<<< HEAD
"""
Optimized blocking:
- 3 keys only (exact name, sorted tokens, first token)
- Drops any block with >MAX_BUCKET members (kills "the", "shri", "john" blowups)
- Cap candidates per S1
- Vectorized where possible
"""
=======
import re
>>>>>>> 3deb04d (Update entity resolution project)
import time
import pandas as pd

from src.normalize import normalize_name, normalize_address, extract_postal

<<<<<<< HEAD
# 🚨 THE KEY CONSTANTS
MAX_BUCKET_SIZE = 50   # reject blocks with more than this many members
MAX_CANDIDATES_PER_S1 = 20
MAX_HITS_PER_S1 = 5
=======
LEGAL = {
    "incorporated", "limited", "company", "corporation",
    "private", "llc", "inc", "ltd", "corp", "pvt", "co",
    "sarl", "sas", "sa", "eurl", "snc", "pllc", "lp", "plc"
}

ADDRESS_RESERVE = 5
BLOCK_KEEP = 100


def _extract_address_number_token(address):
    if address is None:
        return "", ""
    try:
        if pd.isna(address):
            return "", ""
    except (TypeError, ValueError):
        pass

    tokens = re.findall(r"\d+[a-zA-Z]?|[a-zA-Z]+", str(address).lower())
    generic = {
        "road", "street", "avenue", "boulevard", "drive", "lane",
        "highway", "place", "square", "north", "south", "east", "west",
        "number", "no", "floor", "unit", "apt", "apartment", "block",
        "plot", "sector", "village", "district", "county", "city",
        "state", "the", "and"
    }

    for i, token in enumerate(tokens):
        if not re.fullmatch(r"\d+[a-zA-Z]?", token):
            continue
        if len(re.sub(r"\D", "", token)) < 2:
            continue

        for nxt in tokens[i + 1:i + 5]:
            if (
                nxt.isalpha()
                and nxt.lower() not in generic
                and len(nxt) >= 3
            ):
                return token.lower(), nxt[:4].lower()

        return token.lower(), ""

    return "", ""
>>>>>>> 3deb04d (Update entity resolution project)


def prep_keys(df):
    """Build 3 block keys."""
    df = df.copy()
<<<<<<< HEAD
    nm = df["norm_name"].astype(str)
    c = df["country_l"]

    df["key_exact"] = c + "|E|" + nm
    df["key_sorted"] = c + "|S|" + nm.str.split().apply(
        lambda t: " ".join(sorted(t)) if isinstance(t, list) and t else ""
    )
    df["key_first"] = c + "|F|" + nm.str.split().str[0].fillna("")
=======

    if "country_l" not in df.columns:
        df["country_l"] = (
            df["country"].astype(str).str.strip().str.lower()
        )

    if "norm_name" not in df.columns:
        df["norm_name"] = df["business_name"].map(normalize_name)

    if "norm_addr" not in df.columns:
        df["norm_addr"] = df["business_address"].map(normalize_address)

    if "postal" not in df.columns:
        df["postal"] = df["business_address"].map(extract_postal)

    df["postal3"] = df["postal"].astype(str).str[:3]
    df["first_token"] = (
        df["norm_name"].astype(str).str.split().str[0].fillna("")
    )
    df["first3"] = df["norm_name"].astype(str).str[:3]
    df["sorted_tokens"] = df["norm_name"].apply(
        lambda x: " ".join(sorted(str(x).split())) if x else ""
    )
    df["core_tokens"] = df["norm_name"].apply(
        lambda x: " ".join(
            sorted(t for t in str(x).split() if t not in LEGAL)
        ) if x else ""
    )
    df["street_num"] = (
        df["norm_addr"].astype(str)
        .str.extract(r"(\b\d{2,5}\b)", expand=False)
        .fillna("")
    )

    extracted = df["norm_addr"].map(_extract_address_number_token)
    df["addr_num"] = extracted.map(lambda x: x[0])
    df["addr_token4"] = extracted.map(lambda x: x[1])
    df["addr_num_token4"] = (
        df["addr_num"].astype(str) + "|" + df["addr_token4"].astype(str)
    )
    df.loc[
        (df["addr_num"] == "") | (df["addr_token4"] == ""),
        "addr_num_token4"
    ] = ""
>>>>>>> 3deb04d (Update entity resolution project)

    return df


<<<<<<< HEAD
def build_capped_blocks(df, label="S"):
    """Build inverted indices with hard size cap."""
    print(f"  [{label}] building blocks...", flush=True)
    t0 = time.time()
=======
BLOCK_DEFS = [
    ("p", "postal3", 1),
    ("t", "first_token", 1),
    ("f", "first3", 1),
    ("s", "sorted_tokens", 1),
    ("k", "core_tokens", 1),
    ("n", "street_num", 3),
    ("h", "addr_num_token4", 3),
]
>>>>>>> 3deb04d (Update entity resolution project)


def _make_query_keys(s1):
    relevant = {}
    for tag, col, min_len in BLOCK_DEFS:
        vals = s1[["country_l", col]].copy()
        vals["country_l"] = vals["country_l"].fillna("").astype(str)
        vals[col] = vals[col].fillna("").astype(str)
        vals = vals[vals[col].str.len() >= min_len]
        relevant[tag] = set(zip(vals["country_l"], vals[col]))
    return relevant


def _build_query_aware_blocks(df, relevant, label):
    t0 = time.time()
    print(f"  [{label}] preparing keys...", flush=True)
    df = _prepare_keys(df)
    blocks = {}
    for key_col in ("key_exact", "key_sorted", "key_first"):
        g = df.groupby(key_col, sort=False)["entity_id"].apply(list)
        total = len(g)
        g = g[g.apply(len) <= MAX_BUCKET_SIZE]
        dropped = total - len(g)
        blocks[key_col] = g.to_dict()
        print(f"  [{label}] {key_col}: kept {len(blocks[key_col]):,} "
              f"(dropped {dropped:,} oversized)", flush=True)

<<<<<<< HEAD
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
=======
    for tag, col, min_len in BLOCK_DEFS:
        stage_t0 = time.time()
        wanted = relevant[tag]
        if not wanted:
            continue

        vals = df[["country_l", col, "entity_id"]].copy()
        vals["country_l"] = vals["country_l"].fillna("").astype(str)
        vals[col] = vals[col].fillna("").astype(str)
        vals = vals[vals[col].str.len() >= min_len]

        if vals.empty:
            continue

        pairs = list(zip(vals["country_l"].to_numpy(), vals[col].to_numpy()))
        mask = [pair in wanted for pair in pairs]

        matched = 0
        if any(mask):
            hits = vals.loc[mask, ["country_l", col, "entity_id"]]
            matched = len(hits)
            counts = {}

            for country, value, entity_id in hits.itertuples(index=False, name=None):
                key = f"{country}|{tag}|{value}"
                count = counts.get(key, 0)
                if count >= BLOCK_KEEP:
                    continue

                bucket = blocks.get(key)
                if bucket is None:
                    blocks[key] = [str(entity_id)]
                else:
                    bucket.append(str(entity_id))
                counts[key] = count + 1

        print(
            f"  [{label}] {tag}: {matched:,} relevant rows | "
            f"{time.time()-stage_t0:.1f}s",
            flush=True
        )

    print(
        f"  [{label}] blocks: {len(blocks):,} | "
        f"{time.time()-t0:.1f}s",
        flush=True
    )
    return blocks, df


def _add_group(output, seen, group, limit):
    if len(output) >= limit:
        return
    for entity_id in group:
        if entity_id in seen:
            continue
        seen.add(entity_id)
        output.append(entity_id)
        if len(output) >= limit:
            return


def _get_base_candidates(row, b2, b3, cap):
    c = row.country_l
    output = []
    seen = set()

    strong = [("p", row.postal3), ("s", row.sorted_tokens), ("k", row.core_tokens)]
    medium = [("t", row.first_token), ("f", row.first3)]

    for tag, value in strong:
        if not value:
            continue
        _add_group(output, seen, b2.get(f"{c}|{tag}|{value}", ()), cap)
        if len(output) >= cap:
            return output
        _add_group(output, seen, b3.get(f"{c}|{tag}|{value}", ()), cap)
        if len(output) >= cap:
            return output

    for tag, value in medium:
        if not value:
            continue
        _add_group(output, seen, b2.get(f"{c}|{tag}|{value}", ()), cap)
        if len(output) >= cap:
            return output
        _add_group(output, seen, b3.get(f"{c}|{tag}|{value}", ()), cap)
        if len(output) >= cap:
            return output

    if row.street_num and len(row.street_num) >= 3:
        _add_group(output, seen, b2.get(f"{c}|n|{row.street_num}", ()), cap)
        if len(output) >= cap:
            return output
        _add_group(output, seen, b3.get(f"{c}|n|{row.street_num}", ()), cap)

    return output


def generate_candidates(s1_df, s2_df, s3_df, max_candidates=300):
    print("Preparing S1 query keys...", flush=True)
    s1 = _prepare_keys(s1_df)
    relevant = _make_query_keys(s1)

    print(
        "Relevant S1 keys: " +
        ", ".join(f"{tag}={len(keys):,}" for tag, keys in relevant.items()),
        flush=True
    )

    print("Building query-aware S2 blocks...", flush=True)
    b2, s2 = _build_query_aware_blocks(s2_df, relevant, "S2")

    print("Building query-aware S3 blocks...", flush=True)
    b3, s3 = _build_query_aware_blocks(s3_df, relevant, "S3")

    reserve = min(ADDRESS_RESERVE, max(0, max_candidates))
    base_cap = max(0, max_candidates - reserve)

    print(
        f"Generating candidates for {len(s1):,} S1 "
        f"(base={base_cap}, address={reserve})...",
        flush=True
    )
>>>>>>> 3deb04d (Update entity resolution project)

    t0 = time.time()

<<<<<<< HEAD
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
=======
    for i, row in enumerate(s1.itertuples(index=False)):
        if i % 100_000 == 0 and i > 0:
            print(
                f"  ... {i:,}/{len(s1):,} "
                f"({time.time()-t0:.1f}s)",
                flush=True
            )

        base = _get_base_candidates(row, b2, b3, base_cap)
        final = base
        seen = set(base)

        if row.addr_num_token4:
            key = f"{row.country_l}|h|{row.addr_num_token4}"

            for entity_id in b2.get(key, ()):
                if entity_id not in seen:
                    final.append(entity_id)
                    seen.add(entity_id)
                if len(final) >= max_candidates:
                    break
                if len(final) >= base_cap + reserve:
                    break

            if len(final) < max_candidates:
                for entity_id in b3.get(key, ()):
                    if entity_id not in seen:
                        final.append(entity_id)
                        seen.add(entity_id)
                    if len(final) >= max_candidates:
                        break
                    if len(final) >= base_cap + reserve:
                        break

        out[row.entity_id] = set(final[:max_candidates])

    print(
        f"  Candidate generation done in {time.time()-t0:.1f}s",
        flush=True
    )
    return out, s1, s2, s3
>>>>>>> 3deb04d (Update entity resolution project)
