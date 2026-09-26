from rapidfuzz import fuzz


LEGAL_SUFFIXES = {
    "incorporated", "limited", "company", "corporation", "private",
    "llc", "inc", "ltd", "corp", "pvt", "co", "sarl", "sas", "sa",
    "eurl", "snc", "pllc", "lp", "plc", "llp", "societe", "compagnie",
}


def safe_similarity(a, b):
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def token_set(text):
    if not text:
        return set()
    return set(text.split())


def jaccard_similarity(a, b):
    ta, tb = token_set(a), token_set(b)
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def token_overlap(a, b):
    ta, tb = token_set(a), token_set(b)
    smaller = min(len(ta), len(tb))
    if smaller == 0:
        return 0.0
    return len(ta & tb) / smaller


def token_diff_count(a, b):
    return len(token_set(a) ^ token_set(b))


def length_ratio(a, b):
    if not a or not b:
        return 0.0
    shorter, longer = min(len(a), len(b)), max(len(a), len(b))
    return shorter / longer


def first_token_match(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0
    return int(ta[0] == tb[0])


def last_token_match(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0
    return int(ta[-1] == tb[-1])


def token_sort_similarity(a, b):
    if not a or not b:
        return 0.0
    sa = " ".join(sorted(a.split()))
    sb = " ".join(sorted(b.split()))
    return fuzz.ratio(sa, sb) / 100.0


def token_set_ratio(a, b):
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def partial_similarity(a, b):
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b) / 100.0


def strip_legal(norm_name):
    """Remove legal suffixes and rejoin — used for 'core' features."""
    if not norm_name:
        return ""
    tokens = [t for t in norm_name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def compute_features(s1_row, cand_row):
    name1 = s1_row.get("norm_name", "")
    name2 = cand_row.get("norm_name", "")

    addr1 = s1_row.get("norm_addr", "")
    addr2 = cand_row.get("norm_addr", "")

    country1 = str(s1_row.get("country", "")).strip().lower()
    country2 = str(cand_row.get("country", "")).strip().lower()

    postal1 = str(s1_row.get("postal", "")).strip()
    postal2 = str(cand_row.get("postal", "")).strip()

    # Core names (legal suffixes stripped)
    core1 = strip_legal(name1)
    core2 = strip_legal(name2)

    name_tokens_1 = token_set(name1)
    name_tokens_2 = token_set(name2)
    common = name_tokens_1 & name_tokens_2

    # Postal prefixes
    p1_3 = postal1[:3] if postal1 else ""
    p2_3 = postal2[:3] if postal2 else ""

    features = {
        # Name features
        "name_jaccard": jaccard_similarity(name1, name2),
        "name_lev": safe_similarity(name1, name2),
        "name_token_sort": token_sort_similarity(name1, name2),
        "name_token_set": token_set_ratio(name1, name2),
        "name_partial": partial_similarity(name1, name2),
        "common_token_count": len(common),
        "name_token_overlap": token_overlap(name1, name2),
        "name_token_diff_count": token_diff_count(name1, name2),
        "name_token_count_diff": abs(len(name_tokens_1) - len(name_tokens_2)),
        "first_token_match": first_token_match(name1, name2),
        "last_token_match": last_token_match(name1, name2),
        "name_len_ratio": length_ratio(name1, name2),

        # Core name features (differ from raw — legal suffixes stripped)
        "core_name_jaccard": jaccard_similarity(core1, core2),
        "core_name_similarity": safe_similarity(core1, core2),
        "core_token_overlap": token_overlap(core1, core2),
        "core_token_diff_count": token_diff_count(core1, core2),
        "core_name_len_ratio": length_ratio(core1, core2),

        # Address features
        "addr_jaccard": jaccard_similarity(addr1, addr2),
        "addr_lev": safe_similarity(addr1, addr2),
        "addr_token_sort": token_sort_similarity(addr1, addr2),
        "addr_token_set": token_set_ratio(addr1, addr2),
        "addr_partial": partial_similarity(addr1, addr2),
        "addr_len_ratio": length_ratio(addr1, addr2),

        # Country / postal
        "country_match": int(bool(country1) and bool(country2) and country1 == country2),
        "postal_match": int(bool(postal1) and bool(postal2) and postal1 == postal2),
        "postal_prefix3_match": int(bool(p1_3) and bool(p2_3) and p1_3 == p2_3),
    }

    return features


if __name__ == "__main__":
    s1 = {
        "norm_name": "rajani enterprises of kannur",
        "norm_addr": "12 mg road kannur",
        "country": "india",
        "postal": "670001",
    }
    candidate = {
        "norm_name": "rajani enterprises kannur of",
        "norm_addr": "12 mg road kannur",
        "country": "india",
        "postal": "670001",
    }
    features = compute_features(s1, candidate)
    print("Feature test:")
    for name, value in features.items():
        print(f"{name:28s}: {value}")