import gc
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, ".")

from src.data_loader import load_test
from src.blocking import generate_candidates


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"
MODEL_FILE = "model.pkl"

# Use the cap we just validated experimentally.
MAX_CANDIDATES = 20

# Larger batches reduce pandas overhead. 5,000 * 20 = ~100k pairs.
BATCH_SIZE = 5_000

OUTPUT_MATCH = "output/matching_results.tsv"
OUTPUT_CANDIDATE = "output/candidate_pairs.tsv"

# Features whose calculation is relatively expensive.
EXPENSIVE_FEATURES = {
    "name_partial",
    "addr_partial",
}


def token_jaccard_array(a, b):
    out = np.empty(len(a), dtype=np.float32)

    for i, (x, y) in enumerate(zip(a, b)):
        sx = set(x.split()) if x else set()
        sy = set(y.split()) if y else set()

        union = sx | sy
        out[i] = (
            len(sx & sy) / len(union)
            if union else 0.0
        )

    return out


def cheap_features(s1_rows, candidate_rows):
    """
    Calculate every feature except the two expensive partial-ratio
    features.

    These are enough to determine many predictions without ever
    calculating partial_ratio.
    """
    name1 = (
        s1_rows["norm_name"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    name2 = (
        candidate_rows["norm_name"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    addr1 = (
        s1_rows["norm_addr"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    addr2 = (
        candidate_rows["norm_addr"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    features = {}

    features["name_jaccard"] = token_jaccard_array(
        name1,
        name2,
    )

    features["name_lev"] = (
        process.cpdist(
            name1,
            name2,
            scorer=fuzz.ratio,
            workers=-1,
        ) / 100.0
    ).astype(np.float32)

    features["name_token_sort"] = (
        process.cpdist(
            name1,
            name2,
            scorer=fuzz.token_sort_ratio,
            workers=-1,
        ) / 100.0
    ).astype(np.float32)

    features["name_token_set"] = (
        process.cpdist(
            name1,
            name2,
            scorer=fuzz.token_set_ratio,
            workers=-1,
        ) / 100.0
    ).astype(np.float32)

    features["addr_jaccard"] = token_jaccard_array(
        addr1,
        addr2,
    )

    features["addr_lev"] = (
        process.cpdist(
            addr1,
            addr2,
            scorer=fuzz.ratio,
            workers=-1,
        ) / 100.0
    ).astype(np.float32)

    features["addr_token_sort"] = (
        process.cpdist(
            addr1,
            addr2,
            scorer=fuzz.token_sort_ratio,
            workers=-1,
        ) / 100.0
    ).astype(np.float32)

    country1 = (
        s1_rows["country_l"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    country2 = (
        candidate_rows["country_l"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    postal1 = (
        s1_rows["postal"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    postal2 = (
        candidate_rows["postal"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )

    features["country_match"] = (
        (
            (country1 == country2)
            & (country1 != "")
            & (country2 != "")
        )
    ).astype(np.float32)

    features["postal_match"] = (
        (
            (postal1 == postal2)
            & (postal1 != "")
            & (postal2 != "")
        )
    ).astype(np.float32)

    return features, name1, name2, addr1, addr2


def add_expensive_features(
    features,
    name1,
    name2,
    addr1,
    addr2,
    indices,
):
    """
    Calculate partial_ratio only for candidates that are still
    capable of crossing the decision threshold.

    This is exact for a linear LogisticRegression model because
    the omitted features are bounded in [0, 1].
    """
    idx = np.asarray(indices, dtype=np.int64)

    name_partial = np.zeros(len(name1), dtype=np.float32)
    addr_partial = np.zeros(len(name1), dtype=np.float32)

    if len(idx):
        name_partial[idx] = (
            process.cpdist(
                name1[idx],
                name2[idx],
                scorer=fuzz.partial_ratio,
                workers=-1,
            ) / 100.0
        ).astype(np.float32)

        addr_partial[idx] = (
            process.cpdist(
                addr1[idx],
                addr2[idx],
                scorer=fuzz.partial_ratio,
                workers=-1,
            ) / 100.0
        ).astype(np.float32)

    features["name_partial"] = name_partial
    features["addr_partial"] = addr_partial

    return features


def build_candidate_dataframe(
    pair_candidate_ids,
    s2_idx,
    s3_idx,
):
    cols = [
        "norm_name",
        "norm_addr",
        "country_l",
        "postal",
    ]

    n = len(pair_candidate_ids)

    s2_positions = []
    s2_ids = []
    s3_positions = []
    s3_ids = []

    for i, cid in enumerate(pair_candidate_ids):
        if cid.startswith("S2-"):
            s2_positions.append(i)
            s2_ids.append(cid)
        else:
            s3_positions.append(i)
            s3_ids.append(cid)

    # Construct directly from arrays instead of creating a large
    # empty DataFrame and repeatedly assigning through .iloc.
    candidate_df = pd.DataFrame(
        {
            col: np.empty(n, dtype=object)
            for col in cols
        }
    )

    if s2_ids:
        tmp = (
            s2_idx
            .reindex(s2_ids)[cols]
            .reset_index(drop=True)
        )

        for j, col in enumerate(cols):
            candidate_df.iloc[
                s2_positions, j
            ] = tmp[col].to_numpy()

    if s3_ids:
        tmp = (
            s3_idx
            .reindex(s3_ids)[cols]
            .reset_index(drop=True)
        )

        for j, col in enumerate(cols):
            candidate_df.iloc[
                s3_positions, j
            ] = tmp[col].to_numpy()

    return candidate_df


def exact_threshold_predictions(
    model,
    feats,
    threshold,
    s1_rows,
    candidate_rows,
):
    """
    Fast exact thresholding for LogisticRegression.

    First calculates all cheap features. For the two expensive
    partial-ratio features, use coefficient bounds to identify:

      1. definitely below threshold
      2. definitely above threshold
      3. genuinely ambiguous -> calculate partial ratios

    Therefore the expensive RapidFuzz partial_ratio calls are
    performed only for ambiguous candidates.
    """
    cheap, name1, name2, addr1, addr2 = cheap_features(
        s1_rows,
        candidate_rows,
    )

    n = len(name1)

    # We need a linear model for the bound optimization.
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        full = add_expensive_features(
            cheap,
            name1,
            name2,
            addr1,
            addr2,
            np.arange(n),
        )

        X = pd.DataFrame(full)[feats]
        return model.predict_proba(X)[:, 1] >= threshold

    coef = np.asarray(model.coef_)

    if coef.ndim != 2 or coef.shape[0] != 1:
        full = add_expensive_features(
            cheap,
            name1,
            name2,
            addr1,
            addr2,
            np.arange(n),
        )

        X = pd.DataFrame(full)[feats]
        return model.predict_proba(X)[:, 1] >= threshold

    coef = coef[0]
    intercept = float(np.asarray(model.intercept_).ravel()[0])

    # Build cheap contribution to the linear score.
    score = np.full(
        n,
        intercept,
        dtype=np.float64,
    )

    for j, feature in enumerate(feats):
        if feature in EXPENSIVE_FEATURES:
            continue

        values = cheap.get(feature)

        if values is None:
            # Unexpected model feature: fall back to exact scoring.
            full = add_expensive_features(
                cheap,
                name1,
                name2,
                addr1,
                addr2,
                np.arange(n),
            )
            X = pd.DataFrame(full)[feats]
            return model.predict_proba(X)[:, 1] >= threshold

        score += coef[j] * values

    # Logistic probability >= threshold iff logit >= logit(threshold).
    logit_threshold = np.log(
        threshold / (1.0 - threshold)
    )

    # For each omitted feature x in [0,1]:
    #   if coefficient >= 0: contribution in [0, coef]
    #   else:                contribution in [coef, 0]
    min_extra = 0.0
    max_extra = 0.0

    for feature in EXPENSIVE_FEATURES:
        if feature not in feats:
            continue

        j = feats.index(feature)
        c = float(coef[j])

        if c >= 0:
            max_extra += c
        else:
            min_extra += c

    lower = score + min_extra
    upper = score + max_extra

    definitely_positive = (
        lower >= logit_threshold
    )

    definitely_negative = (
        upper < logit_threshold
    )

    ambiguous = ~(
        definitely_positive
        | definitely_negative
    )

    ambiguous_idx = np.flatnonzero(ambiguous)

    if len(ambiguous_idx):
        full = add_expensive_features(
            cheap,
            name1,
            name2,
            addr1,
            addr2,
            ambiguous_idx,
        )

        # Calculate exact probability only for ambiguous rows.
        X_amb = pd.DataFrame(
            {
                feature: full[feature][ambiguous_idx]
                for feature in feats
            }
        )

        ambiguous_positive = (
            model.predict_proba(X_amb)[:, 1]
            >= threshold
        )

        result = definitely_positive.copy()
        result[ambiguous_idx] = ambiguous_positive
    else:
        result = definitely_positive

    return result


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("FAST BATCHED PREDICTION — CAP 20")
print("=" * 60)

print("Loading model...")
pkg = joblib.load(MODEL_FILE)

model = pkg["model"]
FEATS = list(pkg["feats"])
THR = float(pkg["thr"])

print(f"  Threshold: {THR:.2f}")
print(f"  Features: {len(FEATS)}")

print("Loading test data...")
t0 = time.time()

s1, s2, s3 = load_test(TEST_DIR)

print(
    f"  Loaded in {time.time()-t0:.1f}s"
)

print(
    f"  S1={len(s1):,} "
    f"S2={len(s2):,} "
    f"S3={len(s3):,}"
)

print("Generating candidates...")
t0 = time.time()

cands, s1e, s2e, s3e = generate_candidates(
    s1,
    s2,
    s3,
    max_candidates=MAX_CANDIDATES,
)

print(
    f"  Candidate generation: "
    f"{time.time()-t0:.1f}s"
)

del s1, s2, s3
gc.collect()

print("Preparing indexed candidate data...")

s1_idx = s1e.set_index(
    "entity_id",
    drop=False,
)

s2_idx = s2e.set_index(
    "entity_id",
    drop=False,
)

s3_idx = s3e.set_index(
    "entity_id",
    drop=False,
)

all_s1_ids = (
    s1e["entity_id"]
    .astype(str)
    .tolist()
)

results = []

print(
    f"Building features + scoring in "
    f"batches of {BATCH_SIZE:,} S1..."
)

score_t0 = time.time()
total_s1 = len(all_s1_ids)

for start in range(
    0,
    total_s1,
    BATCH_SIZE
):
    batch_ids = all_s1_ids[
        start:start + BATCH_SIZE
    ]

    pair_s1_ids = []
    pair_candidate_ids = []

    for sid in batch_ids:
        for cid in cands.get(sid, ()):
            pair_s1_ids.append(sid)
            pair_candidate_ids.append(str(cid))

    if pair_candidate_ids:
        candidate_df = build_candidate_dataframe(
            pair_candidate_ids,
            s2_idx,
            s3_idx,
        )

        s1_batch_df = (
            s1_idx
            .loc[pair_s1_ids]
            .reset_index(drop=True)
        )

        positive = exact_threshold_predictions(
            model,
            FEATS,
            THR,
            s1_batch_df,
            candidate_df,
        )

        batch_hits = {
            sid: []
            for sid in batch_ids
        }

        for sid, cid, is_positive in zip(
            pair_s1_ids,
            pair_candidate_ids,
            positive,
        ):
            if is_positive:
                hits = batch_hits[sid]

                # Preserve the old maximum of 5 returned matches.
                if len(hits) < 5:
                    hits.append(cid)

        for sid in batch_ids:
            results.append(
                {
                    "source1_entity_id": sid,
                    "matched_entity_ids": ",".join(
                        batch_hits[sid]
                    ),
                }
            )

        del candidate_df
        del s1_batch_df
        del positive

    else:
        for sid in batch_ids:
            results.append(
                {
                    "source1_entity_id": sid,
                    "matched_entity_ids": "",
                }
            )

    done = min(
        start + BATCH_SIZE,
        total_s1,
    )

    elapsed = time.time() - score_t0
    rate = done / elapsed if elapsed else 0
    remaining = (
        total_s1 - done
    ) / rate if rate else 0

    print(
        f"  {done:,}/{total_s1:,} "
        f"| elapsed {elapsed/60:.1f}m "
        f"| ETA {remaining/60:.1f}m",
        flush=True,
    )

    gc.collect()


print(
    f"  Scoring finished in "
    f"{(time.time()-score_t0)/60:.1f}m"
)

print("Writing outputs...")

out_df = pd.DataFrame(
    results,
    columns=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)

os.makedirs(
    os.path.dirname(OUTPUT_MATCH),
    exist_ok=True,
)

out_df.to_csv(
    OUTPUT_MATCH,
    sep="\t",
    index=False,
)

out_df.rename(
    columns={
        "matched_entity_ids":
            "candidate_entity_ids"
    }
).to_csv(
    OUTPUT_CANDIDATE,
    sep="\t",
    index=False,
)

n_with = (
    out_df["matched_entity_ids"]
    .ne("")
    .sum()
)

print(
    f"  Rows: {len(out_df):,}"
)

print(
    f"  With matches: {n_with:,} "
    f"({n_with / len(out_df) * 100:.1f}%)"
)

print("=" * 60)
print("DONE")
