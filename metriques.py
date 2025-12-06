import numpy as np
import math
from collections import defaultdict

# ---------- Rating metrics ----------
def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return math.sqrt(((y_true - y_pred) ** 2).mean())

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.abs(y_true - y_pred).mean()

# ---------- Utility: average precision for one user ----------
def apk(actual, predicted, k):
    """
    Average precision at k for a single user.
    actual: set of relevant items
    predicted: list of recommended items (ordered)
    """
    if k == 0:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:  # count only first time
            hits += 1.0
            score += hits / (i + 1.0)
    if len(actual) == 0:
        return 0.0
    return score / min(len(actual), k)

def mapk(actuals, predicteds, k):
    """
    Mean Average Precision at k for lists/dicts
    actuals: dict user -> set(items)
    predicteds: dict user -> list(items)
    """
    ap_sum = 0.0
    n = 0
    for u, actual in actuals.items():
        if u not in predicteds:
            continue
        ap_sum += apk(actual, predicteds[u], k)
        n += 1
    return ap_sum / max(1, n)

# ---------- Precision@k and Recall@k ----------
def precision_at_k(actual, predicted, k):
    predicted_k = predicted[:k]
    if len(predicted_k) == 0:
        return 0.0
    hits = sum(1 for p in predicted_k if p in actual)
    return hits / len(predicted_k)

def recall_at_k(actual, predicted, k):
    if len(actual) == 0:
        return 0.0
    predicted_k = predicted[:k]
    hits = sum(1 for p in predicted_k if p in actual)
    return hits / len(actual)

def mean_precision_recall_at_k(actuals, predicteds, k):
    ps = []
    rs = []
    for u, act in actuals.items():
        if u not in predicteds:
            continue
        pred = predicteds[u]
        ps.append(precision_at_k(act, pred, k))
        rs.append(recall_at_k(act, pred, k))
    return np.mean(ps) if ps else 0.0, np.mean(rs) if rs else 0.0

# ---------- NDCG@k ----------
def dcg_at_k(relevances, k):
    """relevances: list of relevance (1 or 0 or graded) in rank order"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because positions start at 1
    return dcg

def ndcg_at_k(actual, predicted, k):
    # relevance: 1 if item in actual else 0
    rels = [1 if p in actual else 0 for p in predicted[:k]]
    dcg = dcg_at_k(rels, k)
    # ideal DCG
    ideal_rels = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0

def mean_ndcg_at_k(actuals, predicteds, k):
    scores = []
    for u, act in actuals.items():
        if u not in predicteds:
            continue
        scores.append(ndcg_at_k(act, predicteds[u], k))
    return np.mean(scores) if scores else 0.0

# ---------- MRR ----------
def reciprocal_rank(actual, predicted):
    for i, p in enumerate(predicted):
        if p in actual:
            return 1.0 / (i + 1.0)
    return 0.0

def mean_reciprocal_rank(actuals, predicteds):
    rs = []
    for u, act in actuals.items():
        if u not in predicteds:
            continue
        rs.append(reciprocal_rank(act, predicteds[u]))
    return np.mean(rs) if rs else 0.0

# ---------- Coverage, novelty, popularity ----------
def catalog_coverage(predicteds, all_items):
    recommended = set()
    for recs in predicteds.values():
        recommended.update(recs)
    return len(recommended) / len(all_items)

def topk_item_popularity(train_interactions):
    """
    train_interactions: list of (user,item) in training set, or dict item->count
    returns dict item->count
    """
    counts = defaultdict(int)
    for (u, i) in train_interactions:
        counts[i] += 1
    return counts

def novelty_mean_popularity(predicteds, item_popularity, k):
    # lower popularity => more novel (we return average popularity; can invert)
    vals = []
    for recs in predicteds.values():
        topk = recs[:k]
        vals.extend([item_popularity.get(i, 0) for i in topk])
    return np.mean(vals) if vals else 0.0

# ---------- Diversity ----------
def diversity_mean_pairwise(predicteds, item_feature_vectors, k):
    """
    item_feature_vectors: dict item -> vector (numpy array)
    diversity = 1 - mean(similarity) across pairs within each user's top-k
    similarity uses cosine similarity
    """
    def cosine(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    diversities = []
    for recs in predicteds.values():
        topk = recs[:k]
        sims = []
        for i in range(len(topk)):
            for j in range(i+1, len(topk)):
                vi = item_feature_vectors.get(topk[i])
                vj = item_feature_vectors.get(topk[j])
                if vi is None or vj is None:
                    continue
                sims.append(cosine(vi, vj))
        if sims:
            diversities.append(1.0 - np.mean(sims))
    return np.mean(diversities) if diversities else 0.0

# ---------- Personalization (simple) ----------
def personalization(predicteds, k):
    """
    personalization = 1 - average pairwise Jaccard similarity between users' top-k lists
    """
    users = list(predicteds.keys())
    if len(users) < 2:
        return 0.0
    sims = []
    for i in range(len(users)):
        set_i = set(predicteds[users[i]][:k])
        for j in range(i+1, len(users)):
            set_j = set(predicteds[users[j]][:k])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            sims.append(inter / union if union > 0 else 0.0)
    mean_sim = np.mean(sims) if sims else 0.0
    return 1.0 - mean_sim

# ---------- Example aggregator ----------
def evaluate_all(actuals, predicteds, train_interactions=None, all_items=None, item_pop=None,
                 item_features=None, ks=(5,10,20)):
    results = {}
    # ranking metrics
    for k in ks:
        p, r = mean_precision_recall_at_k(actuals, predicteds, k)
        results[f'Precision@{k}'] = p
        results[f'Recall@{k}'] = r
        results[f'MAP@{k}'] = mapk(actuals, predicteds, k)
        results[f'NDCG@{k}'] = mean_ndcg_at_k(actuals, predicteds, k)
    results['MRR'] = mean_reciprocal_rank(actuals, predicteds)
    if all_items and predicteds:
        results['CatalogCoverage'] = catalog_coverage(predicteds, all_items)
    if train_interactions and item_pop:
        for k in ks:
            results[f'Novelty_pop@{k}'] = novelty_mean_popularity(predicteds, item_pop, k)
    if item_features:
        for k in ks:
            results[f'Diversity@{k}'] = diversity_mean_pairwise(predicteds, item_features, k)
    results['Personalization@10'] = personalization(predicteds, 10)
    return results

