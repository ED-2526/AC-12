# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 11:21:34 2025

@author: laura
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

def predict(user_id, item_id, model, users, items):
    """
    Retorna la predicció de rating de l'usuari per un item.
    """
    if user_id not in users or item_id not in items:
        return 0.0

    u_idx = list(users).index(user_id)
    i_idx = list(items).index(item_id)

    return float(np.dot(model.P[u_idx], model.Q[i_idx]))

def recommend(user_id, model, users, items, top_n=10, exclude_rated=True, rated_items_idx=None):
    """
    Retorna les top_n recomanacions per a un usuari.
    Si exclude_rated=True, es poden passar els índexs dels items ja valorats
    per evitar recomanar-los.
    """
    if user_id not in users:
        return []

    u_idx = list(users).index(user_id)

    # calcular scores per tots els items
    scores = np.dot(model.P[u_idx], model.Q.T)

    # eliminar items ja valorats si cal
    if exclude_rated and rated_items_idx is not None:
        scores[rated_items_idx] = -np.inf

    # agafar els top_n
    top_idx = np.argpartition(scores, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    return [(items[i], scores[i]) for i in top_idx]

if __name__ == "__main__":
    model_path = "../models/svd_model.pkl"  # directori consistent amb train
    model, users, items = load_model(model_path)

    # Exemple amb un usuari real
    sample_user = users[0]
    print("\nUser seleccionat:", sample_user)

    # Predicció per un item concret
    example_item = items[0]
    rating_pred = predict_svd(sample_user, example_item, model, users, items)
    print(f"\nPredicció rating per l'item {example_item}: {rating_pred:.3f}")

    # Recomanacions top 10
    recs = recommend_svd(sample_user, model, users, items, top_n=10)
    print("\nTop 10 recomanacions:")
    for item, score in recs:
        print(f"{item}: {score:.3f}")

