def f05_per_entity(pred_ids: set, true_ids: set) -> float:
    """F0.5 for one S1 entity. Beta=0.5 → precision weighted 2x."""
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    if len(pred_ids) == 0:
        return 0.0
    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0
    p = tp / len(pred_ids)
    r = tp / len(true_ids)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(preds: dict, truth: dict) -> dict:
    """Macro-average F0.5 across all S1 entities in truth."""
    scores = []
    for s1_id, true_ids in truth.items():
        pred_ids = preds.get(s1_id, set())
        scores.append(f05_per_entity(pred_ids, true_ids))
    return {
        "f05": sum(scores) / len(scores),
        "n_entities": len(scores),
        "n_singletons": sum(1 for t in truth.values() if len(t) == 0),
    }