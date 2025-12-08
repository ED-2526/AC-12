# -*- coding: utf-8 -*-
"""
SVD Prediction & Recommendation returning INT ratings
"""

import pickle
import numpy as np
from train_svd import MatrixFactorization  


def load_model(model_path="models/svd_model.pkl"):
    """
    Carrega el model SVD des del fitxer pickle.
    Retorna: model, users, items
    """
    with open(model_path, "rb") as f:
        model, users, items = pickle.load(f)
    return model, users, items


# ------------------------------------------------------------
# Helpers per convertir prediccions float → int 1..5
# ------------------------------------------------------------
def clamp_rating(x, min_r=1, max_r=5):
    return max(min_r, min(max_r, x))

def to_int_rating(x):
    if x <= 0:
        return 0
    return int(clamp_rating(round(x)))


# ------------------------------------------------------------
# PREDICCIÓ SVD
# ------------------------------------------------------------
def predict_svd(user_id, item_id, model, users, items):
    if user_id not in users or item_id not in items:
        return 0

    u_idx = list(users).index(user_id)
    i_idx = list(items).index(item_id)

    pred = float(np.dot(model.P[u_idx], model.Q[i_idx]))
    return to_int_rating(pred)


# ------------------------------------------------------------
# RECOMANACIONS TOP-N
# ------------------------------------------------------------
def recommend_svd(user_id, model, users, items, top_n=10,
                  exclude_rated=True, rated_items_idx=None):

    if user_id not in users:
        return []

    u_idx = list(users).index(user_id)

    # Raw scores per tots els ítems
    scores = np.dot(model.P[u_idx], model.Q.T)

    # Excloure items valorats
    if exclude_rated and rated_items_idx is not None:
        scores[rated_items_idx] = -np.inf

    # Top-N índexs
    top_idx = np.argpartition(scores, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    # Convertim cada score a enter
    return [(items[i], to_int_rating(scores[i])) for i in top_idx]


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    model_path = "../models/svd_model.pkl"
    model, users, items = load_model(model_path)

    sample_user = users[0]
    print("\nUser seleccionat:", sample_user)

    example_item = items[0]
    rating_pred = predict_svd(sample_user, example_item, model, users, items)
    print(f"\nPredicció rating per l'item {example_item}: {rating_pred}")

    recs = recommend_svd(sample_user, model, users, items, top_n=10)
    print("\nTop 10 recomanacions:")
    for item, score in recs:
        print(f"{item}: {score}")
