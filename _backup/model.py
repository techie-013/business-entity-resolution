import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

FEATURE_COLS = [
    "name_jaccard", "name_lev", "name_token_sort", "name_partial",
    "addr_jaccard", "addr_lev", "addr_token_sort", "addr_partial",
    "country_match", "postal_match", "first_token_match",
    "name_len_ratio", "addr_len_ratio",
]


def train(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    model.fit(X[FEATURE_COLS], y)
    return model


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