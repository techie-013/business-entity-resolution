import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

FEATURE_COLS = [
    "name_jaccard", "name_lev", "name_token_sort", "name_partial",
    "name_token_overlap", "name_token_count_diff",
    "first_token_match", "name_len_ratio",
    "addr_jaccard", "addr_lev", "addr_token_sort", "addr_partial",
    "addr_len_ratio",
    "country_match", "postal_match",
]


def train(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    model.fit(X[FEATURE_COLS], y)
    return model

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score


# Match the exact features from features.py
FEATURE_COLS = [
    # Name
    "name_jaccard", "name_lev", "name_token_sort", "name_token_set",
    "name_partial", "common_token_count", "name_token_overlap",
    "name_token_diff_count", "name_token_count_diff",
    "first_token_match", "last_token_match", "name_len_ratio",
    # Core name
    "core_name_jaccard", "core_name_similarity",
    "core_token_overlap", "core_token_diff_count", "core_name_len_ratio",
    # Address
    "addr_jaccard", "addr_lev", "addr_token_sort", "addr_token_set",
    "addr_partial", "addr_len_ratio",
    # Country / postal
    "country_match", "postal_match", "postal_prefix3_match",
]


def train(X, y, use_lgbm=False):
    if use_lgbm:
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X[FEATURE_COLS], y)
            return model
        except ImportError:
            print("LightGBM not installed. Falling back to LogReg.")
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500, n_jobs=-1)),
    ])
    model.fit(X[FEATURE_COLS], y)
    return model


def predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X[FEATURE_COLS])[:, 1]
    # Pipeline case
    return model.predict_proba(X[FEATURE_COLS])[:, 1]


def tune_threshold(probs, y_true, grid=np.arange(0.3, 0.91, 0.02)):
    best_t, best_f = 0.5, -1
    for t in grid:
        preds = (probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        if p + r == 0:
            continue
        f05 = (1.25 * p * r) / (0.25 * p + r)
        if f05 > best_f:
            best_f, best_t = f05, t
    return best_t, best_f
def predict_proba(model, X):
    return model.predict_proba(X[FEATURE_COLS])[:, 1]


def tune_threshold(probs, y_true, grid=np.arange(0.3, 0.91, 0.05)):
    best_t, best_f = 0.5, -1
    for t in grid:
        preds = (probs >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        if p + r == 0:
            continue
        f05 = (1.25 * p * r) / (0.25 * p + r)
        if f05 > best_f:
            best_f, best_t = f05, t
    return best_t, best_f