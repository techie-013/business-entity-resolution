import sys
import time

sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

from src.data_loader import load_train, parse_ground_truth
from src.blocking import generate_candidates


# ============================================================
# CONFIG
# ============================================================

S1_SAMPLE_SIZE = 50_000

# Use the same candidate budget as prediction.
# Match the validated cap-20 blocking experiment.
MAX_CANDIDATES = 20

# Keep this many target rows per source for training.
# All true target rows for the sampled S1s are added first;
# remaining slots are random decoys.
TARGET_SAMPLE_SIZE = 500_000

RANDOM_STATE = 42


print("=" * 60)
print("STEP 1: Loading training data...")
t0 = time.time()

s1, s2, s3, gt = load_train("dataset/train")

print(f"  Loaded in {time.time()-t0:.1f}s")

print("  Sampling S1...")
s1_sample = s1.sample(
    n=min(S1_SAMPLE_SIZE, len(s1)),
    random_state=RANDOM_STATE
).copy()

s1_sample["entity_id"] = (
    s1_sample["entity_id"].astype(str)
)

gt_map = parse_ground_truth(gt)

sample_ids = set(
    s1_sample["entity_id"]
)

sample_gt = {
    sid: gt_map.get(sid, set())
    for sid in sample_ids
}

# ------------------------------------------------------------
# Make the S2/S3 training samples contain ALL true targets
# for the sampled S1 rows.
#
# This fixes the old problem where random S2/S3 sampling could
# exclude a real ground-truth target before blocking even ran.
# ------------------------------------------------------------

true_s2_ids = set()
true_s3_ids = set()

for tids in sample_gt.values():
    for tid in tids:
        tid = str(tid)

        if tid.startswith("S2-"):
            true_s2_ids.add(tid)

        elif tid.startswith("S3-"):
            true_s3_ids.add(tid)

print(
    f"  S1={len(s1_sample):,}"
)
print(
    f"  True S2 targets needed={len(true_s2_ids):,}"
)
print(
    f"  True S3 targets needed={len(true_s3_ids):,}"
)


def build_target_sample(
    full_df,
    true_ids,
    target_size,
    seed
):
    df = full_df.copy()

    df["entity_id"] = (
        df["entity_id"].astype(str)
    )

    true_ids = {
        str(x)
        for x in true_ids
    }

    true_rows = df[
        df["entity_id"].isin(true_ids)
    ].copy()

    remaining = max(
        0,
        target_size - len(true_rows)
    )

    if remaining == 0:
        return true_rows

    decoy_pool = df[
        ~df["entity_id"].isin(true_ids)
    ]

    if len(decoy_pool) <= remaining:
        decoys = decoy_pool
    else:
        decoys = decoy_pool.sample(
            n=remaining,
            random_state=seed
        )

    return pd.concat(
        [true_rows, decoys],
        ignore_index=True
    )


print("  Building S2 training sample...")
s2_sample = build_target_sample(
    s2,
    true_s2_ids,
    TARGET_SAMPLE_SIZE,
    RANDOM_STATE
)

print("  Building S3 training sample...")
s3_sample = build_target_sample(
    s3,
    true_s3_ids,
    TARGET_SAMPLE_SIZE,
    RANDOM_STATE + 1
)

print(
    f"  Sampling: "
    f"S1={len(s1_sample):,} "
    f"S2={len(s2_sample):,} "
    f"S3={len(s3_sample):,}"
)

# ============================================================
# STEP 2: Candidate generation
# ============================================================

print("=" * 60)
print("STEP 2: Generating candidates...")
t0 = time.time()

# IMPORTANT:
# Use the same 100-candidate budget as predict_submit.py.
cands, s1e, s2e, s3e = generate_candidates(
    s1_sample,
    s2_sample,
    s3_sample,
    max_candidates=MAX_CANDIDATES
)

print(
    f"  Candidate gen: {time.time()-t0:.1f}s"
)

# Valid because all true target rows for the sampled S1s
# were deliberately included in S2/S3 above.
hits = 0
total = 0

for sid, tids in sample_gt.items():
    if not tids:
        continue

    total += 1

    candidate_ids = {
        str(x)
        for x in cands.get(sid, set())
    }

    if tids & candidate_ids:
        hits += 1

blocking_recall = (
    hits / total
    if total
    else 0.0
)

print(
    f"  Blocking recall: "
    f"{blocking_recall:.3f} "
    f"({hits}/{total})"
)

# ============================================================
# STEP 3: Feature matrix
# ============================================================

print("=" * 60)
print("STEP 3: Building feature matrix...")
t0 = time.time()

s2_idx = s2e.set_index("entity_id")
s3_idx = s3e.set_index("entity_id")
s1_idx = s1e.set_index("entity_id")


def feat(s1r, cr):
    n1, n2 = (
        str(s1r["norm_name"]),
        str(cr["norm_name"])
    )

    a1, a2 = (
        str(s1r["norm_addr"]),
        str(cr["norm_addr"])
    )

    t1n = set(n1.split())
    t2n = set(n2.split())

    t1a = set(a1.split())
    t2a = set(a2.split())

    def jac(x, y):
        if not x or not y:
            return 0.0

        return len(x & y) / len(x | y)

    return {
        "name_jaccard":
            jac(t1n, t2n),

        "name_lev":
            fuzz.ratio(n1, n2) / 100.0,

        "name_token_sort":
            fuzz.token_sort_ratio(n1, n2) / 100.0,

        "name_partial":
            fuzz.partial_ratio(n1, n2) / 100.0,

        "name_token_set":
            fuzz.token_set_ratio(n1, n2) / 100.0,

        "addr_jaccard":
            jac(t1a, t2a),

        "addr_lev":
            fuzz.ratio(a1, a2) / 100.0,

        "addr_token_sort":
            fuzz.token_sort_ratio(a1, a2) / 100.0,

        "addr_partial":
            fuzz.partial_ratio(a1, a2) / 100.0,

        "country_match":
            int(
                str(s1r["country_l"])
                == str(cr["country_l"])
            ),

        "postal_match":
            int(
                bool(s1r["postal"])
                and
                str(s1r["postal"])
                == str(cr["postal"])
            ),
    }


rows = []

for i, s1_id in enumerate(cands.keys()):

    if i % 5000 == 0:
        print(
            f"  features {i}/{len(cands)} "
            f"({time.time()-t0:.1f}s)"
        )

    if s1_id not in s1_idx.index:
        continue

    s1r = s1_idx.loc[s1_id]

    true_ids = sample_gt.get(
        s1_id,
        set()
    )

    # cands already contains <= 100 candidates.
    for cid in cands[s1_id]:

        cid = str(cid)

        if cid.startswith("S2-"):
            src = s2_idx
        else:
            src = s3_idx

        if cid not in src.index:
            continue

        cr = src.loc[cid]

        f = feat(
            s1r,
            cr
        )

        f["s1_id"] = s1_id
        f["cand_id"] = cid
        f["label"] = int(
            cid in true_ids
        )

        rows.append(f)


X = pd.DataFrame(rows)

print(
    f"  Feature matrix: "
    f"{X.shape}  "
    f"({time.time()-t0:.1f}s)"
)

print(
    f"  Positive rate: "
    f"{X['label'].mean():.3f}"
)

# ============================================================
# STEP 4: Training LogReg
# ============================================================

print("=" * 60)
print("STEP 4: Training LogReg...")

FEATS = [
    c for c in X.columns
    if c not in (
        "s1_id",
        "cand_id",
        "label"
    )
]

print(
    f"  Features ({len(FEATS)}): "
    f"{FEATS}"
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

Xtr, Xva, ytr, yva = train_test_split(
    X[FEATS],
    X["label"],
    test_size=0.2,
    random_state=42,
    stratify=X["label"]
)

model = Pipeline([
    (
        "sc",
        StandardScaler()
    ),
    (
        "clf",
        LogisticRegression(
            class_weight="balanced",
            max_iter=200,
            n_jobs=-1
        )
    )
])

model.fit(
    Xtr,
    ytr
)

print("  Trained.")

# ============================================================
# STEP 5: Threshold tuning
# ============================================================

print("=" * 60)
print("STEP 5: Threshold tuning...")

probs = model.predict_proba(
    Xva
)[:, 1]

best_t = 0.5
best_f = -1

for t in np.arange(
    0.30,
    0.91,
    0.02
):
    preds = (
        probs >= t
    ).astype(int)

    tp = (
        (preds == 1)
        & (yva == 1)
    ).sum()

    fp = (
        (preds == 1)
        & (yva == 0)
    ).sum()

    fn = (
        (preds == 0)
        & (yva == 1)
    ).sum()

    if tp == 0:
        continue

    p = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )

    r = (
        tp / (tp + fn)
        if (tp + fn)
        else 0
    )

    if p + r == 0:
        continue

    f05 = (
        1.25 * p * r
    ) / (
        0.25 * p + r
    )

    if f05 > best_f:
        best_f = f05
        best_t = t

print(
    f"  Best threshold: "
    f"{best_t:.2f} "
    f"F0.5={best_f:.4f}"
)

import joblib

joblib.dump(
    {
        "model": model,
        "feats": FEATS,
        "thr": float(best_t)
    },
    "model.pkl"
)

print("  Saved model.pkl")

print("=" * 60)
print("DONE — Now run predict_submit.py")
