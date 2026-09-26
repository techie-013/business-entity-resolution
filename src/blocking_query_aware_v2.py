import re
import time
import pandas as pd

from src.normalize import normalize_name, normalize_address, extract_postal

LEGAL = {
    "incorporated", "limited", "company", "corporation",
    "private", "llc", "inc", "ltd", "corp", "pvt", "co",
    "sarl", "sas", "sa", "eurl", "snc", "pllc", "lp", "plc"
}

# Reserve five of the final candidate slots for the new address blocker.
ADDRESS_RESERVE = 5


def _extract_address_number_token(address):
    if address is None:
        return "", ""

    try:
        if pd.isna(address):
            return "", ""
    except (TypeError, ValueError):
        pass

    tokens = re.findall(
        r"\d+[a-zA-Z]?|[a-zA-Z]+",
        str(address).lower()
    )

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


def _prepare_keys(df):
    df = df.copy()

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

    return df


def build_blocks(df, label="S"):
    print(f"  [{label}] preparing keys...", flush=True)
    t0 = time.time()
    df = _prepare_keys(df)

    blocks = {}
    keys = [
        ("p", "postal3", 1),
        ("t", "first_token", 1),
        ("f", "first3", 1),
        ("s", "sorted_tokens", 1),
        ("k", "core_tokens", 1),
        ("n", "street_num", 3),
        ("h", "addr_num_token4", 3),
    ]

    for tag, col, min_len in keys:
        valid = df[
            df[col].astype(str).str.len() >= min_len
        ]

        for k, group in valid.groupby(["country_l", col]):
            value = k[1]
            if not value:
                continue

            blocks[f"{k[0]}|{tag}|{value}"] = set(
                group["entity_id"]
            )

        print(
            f"  [{label}] after {tag}: {len(blocks):,} keys",
            flush=True
        )

    print(
        f"  [{label}] total: {len(blocks):,} "
        f"blocks in {time.time()-t0:.1f}s",
        flush=True
    )
    return blocks


def _original_candidates(row, block_s2, block_s3, cap):
    c = row.country_l

    strong = set()

    if row.postal3:
        strong |= block_s2.get(f"{c}|p|{row.postal3}", set())
        strong |= block_s3.get(f"{c}|p|{row.postal3}", set())

    if row.sorted_tokens:
        strong |= block_s2.get(f"{c}|s|{row.sorted_tokens}", set())
        strong |= block_s3.get(f"{c}|s|{row.sorted_tokens}", set())

    if row.core_tokens:
        strong |= block_s2.get(f"{c}|k|{row.core_tokens}", set())
        strong |= block_s3.get(f"{c}|k|{row.core_tokens}", set())

    medium = set()

    if row.first_token:
        medium |= block_s2.get(f"{c}|t|{row.first_token}", set())
        medium |= block_s3.get(f"{c}|t|{row.first_token}", set())

    if row.first3:
        medium |= block_s2.get(f"{c}|f|{row.first3}", set())
        medium |= block_s3.get(f"{c}|f|{row.first3}", set())

    weak = set()

    if row.street_num and len(row.street_num) >= 3:
        weak |= block_s2.get(f"{c}|n|{row.street_num}", set())
        weak |= block_s3.get(f"{c}|n|{row.street_num}", set())

    cand = list(strong)

    if len(cand) < cap:
        cand += [x for x in medium if x not in strong]

    if len(cand) < cap:
        cand += [
            x for x in weak
            if x not in strong and x not in medium
        ]

    return cand[:cap]


def generate_candidates(s1_df, s2_df, s3_df, max_candidates=300):
    if "norm_name" not in s2_df.columns:
        s2_df = _prepare_keys(s2_df)

    if "norm_name" not in s3_df.columns:
        s3_df = _prepare_keys(s3_df)

    print("Building S2 blocks...", flush=True)
    b2 = build_blocks(s2_df, "S2")

    print("Building S3 blocks...", flush=True)
    b3 = build_blocks(s3_df, "S3")

    print("Preparing S1 keys...", flush=True)
    s1 = _prepare_keys(s1_df)

    print(
        f"Generating candidates for {len(s1):,} S1...",
        flush=True
    )

    t0 = time.time()
    out = {}

    reserve = min(ADDRESS_RESERVE, max(0, max_candidates))
    base_cap = max(0, max_candidates - reserve)

    for i, row in enumerate(s1.itertuples(index=False)):
        if i % 50000 == 0 and i > 0:
            print(
                f"  ... {i:,}/{len(s1):,} "
                f"({time.time()-t0:.1f}s)",
                flush=True
            )

        base_candidates = _original_candidates(
            row, b2, b3, base_cap
        )

        address_candidates = []

        if row.addr_num_token4:
            key = (
                f"{row.country_l}|h|"
                f"{row.addr_num_token4}"
            )
            seen = set(base_candidates)

            for entity_id in b2.get(key, set()):
                entity_id = str(entity_id)
                if entity_id not in seen:
                    address_candidates.append(entity_id)
                    seen.add(entity_id)

                if len(address_candidates) >= reserve:
                    break

            if len(address_candidates) < reserve:
                for entity_id in b3.get(key, set()):
                    entity_id = str(entity_id)
                    if entity_id not in seen:
                        address_candidates.append(entity_id)
                        seen.add(entity_id)

                    if len(address_candidates) >= reserve:
                        break

        final_candidates = (
            base_candidates + address_candidates[:reserve]
        )

        out[row.entity_id] = set(
            final_candidates[:max_candidates]
        )

    print(
        f"  Done in {time.time()-t0:.1f}s",
        flush=True
    )

    return out, s1, s2_df, s3_df
