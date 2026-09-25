def f05_per_entity(pred_ids, true_ids):
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


def macro_f05(preds, truth):
    scores = []
    for s1_id, true_ids in truth.items():
        scores.append(f05_per_entity(preds.get(s1_id, set()), true_ids))
    return {
        "f05": sum(scores) / len(scores),
        "n_entities": len(scores),
        "n_singletons": sum(1 for t in truth.values() if len(t) == 0),
    }