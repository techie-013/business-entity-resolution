from rapidfuzz import fuzz


def safe_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def token_set(text):
    if not text:
        return set()

    return set(text.split())


def jaccard_similarity(a, b):
    tokens_a = token_set(a)
    tokens_b = token_set(b)

    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def token_overlap(a, b):
    tokens_a = token_set(a)
    tokens_b = token_set(b)

    smaller = min(len(tokens_a), len(tokens_b))

    if smaller == 0:
        return 0.0

    return len(tokens_a & tokens_b) / smaller


def token_diff_count(a, b):
    tokens_a = token_set(a)
    tokens_b = token_set(b)

    return len(tokens_a ^ tokens_b)


def length_ratio(a, b):
    if not a or not b:
        return 0.0

    shorter = min(len(a), len(b))
    longer = max(len(a), len(b))

    return shorter / longer


def first_token_match(a, b):
    tokens_a = a.split()
    tokens_b = b.split()

    if not tokens_a or not tokens_b:
        return 0

    return int(tokens_a[0] == tokens_b[0])


def token_sort_similarity(a, b):
    if not a or not b:
        return 0.0

    sorted_a = " ".join(sorted(a.split()))
    sorted_b = " ".join(sorted(b.split()))

    return fuzz.ratio(sorted_a, sorted_b) / 100.0


def partial_similarity(a, b):
    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b) / 100.0


def compute_features(s1_row, cand_row):
    name1 = s1_row.get("norm_name", "")
    name2 = cand_row.get("norm_name", "")

    addr1 = s1_row.get("norm_addr", "")
    addr2 = cand_row.get("norm_addr", "")

    country1 = str(s1_row.get("country", "")).strip().lower()
    country2 = str(cand_row.get("country", "")).strip().lower()

    postal1 = str(s1_row.get("postal", "")).strip()
    postal2 = str(cand_row.get("postal", "")).strip()

    name_tokens_1 = token_set(name1)
    name_tokens_2 = token_set(name2)

    common_tokens = name_tokens_1 & name_tokens_2

    features = {
        "name_jaccard": jaccard_similarity(name1, name2),
        "name_lev": safe_similarity(name1, name2),
        "name_token_sort": token_sort_similarity(name1, name2),
        "name_partial": partial_similarity(name1, name2),
        "common_token_count": len(common_tokens),
        "name_token_overlap": token_overlap(name1, name2),
        "name_token_diff_count": token_diff_count(name1, name2),
        "name_token_count_diff": abs(
            len(name_tokens_1) - len(name_tokens_2)
        ),
        "first_token_match": first_token_match(name1, name2),
        "name_len_ratio": length_ratio(name1, name2),

        "core_name_jaccard": jaccard_similarity(name1, name2),
        "core_name_similarity": safe_similarity(name1, name2),
        "core_token_overlap": token_overlap(name1, name2),
        "core_token_diff_count": token_diff_count(name1, name2),

        "addr_jaccard": jaccard_similarity(addr1, addr2),
        "addr_lev": safe_similarity(addr1, addr2),
        "addr_token_sort": token_sort_similarity(addr1, addr2),
        "addr_partial": partial_similarity(addr1, addr2),
        "addr_len_ratio": length_ratio(addr1, addr2),

        "country_match": int(
            bool(country1)
            and bool(country2)
            and country1 == country2
        ),
        "postal_match": int(
            bool(postal1)
            and bool(postal2)
            and postal1 == postal2
        ),
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
        print(f"{name:25s}: {value}")