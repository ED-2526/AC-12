# -*- coding: utf-8 -*-
"""
infer_knn.py
Funcions per predir i recomanar amb el model Item-Item KNN.
Prediccions arrodonides a 1..5 i noms coherents amb el main.
"""

import pickle
import numpy as np
from train_knn import ItemItemKNN  # només la classe amb fit i topk_neighbors

def load_model(model_path="models/knn_item_model.pkl"):
    """
    Carrega el model entrenat des de disc.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_knn(user_id, item_id, model):
    """
    Predicció d'un rating (1-5) per un usuari i un item.
    Retorna 0.0 si no hi ha informació.
    """
    if item_id not in model.item_index or user_id not in model.user_index:
        return 0.0

    i_idx = model.item_index[item_id]
    u_idx = model.user_index[user_id]
    user_ratings = model.item_user_matrix[:, u_idx]

    if model.use_topk:
        neighbors = model.topk_neighbors[i_idx]
        if neighbors is None:
            return 0.0
        nbr_idx, nbr_sims = neighbors
        if len(nbr_idx) == 0:
            return 0.0
        nbr_ratings = user_ratings[nbr_idx]
        mask = nbr_ratings > 0
        if mask.sum() == 0:
            return 0.0
        sims = nbr_sims[mask]
        ratings = nbr_ratings[mask]
    else:
        sims = model.topk_neighbors[i_idx]
        rated_mask = user_ratings > 0
        if rated_mask.sum() == 0:
            return 0.0
        sims = sims[rated_mask]
        ratings = user_ratings[rated_mask]

    if sims.sum() == 0:
        return 0.0

    pred = float(np.dot(sims, ratings) / sims.sum())
    # Arrodonim a 1..5
    pred_rounded = float(min(5, max(1, round(pred))))
    return pred_rounded

def recommend_knn(user_id, model, top_n=10):
    """
    Recomanacions top_n per un usuari.
    Retorna llista de tuples (item_id, predicció arrodonida 1..5)
    """
    if user_id not in model.user_index:
        return []

    u_idx = model.user_index[user_id]
    user_ratings = model.item_user_matrix[:, u_idx]
    unseen_items = np.where(user_ratings == 0)[0]

    preds = []
    for i in unseen_items:
        item_id = model.items[i]
        score = predict_knn(user_id, item_id, model)
        if score != 0.0:
            preds.append((i, score))

    # Ordenem per score descendent
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]
    return [(model.items[i], score) for i, score in top]

"""
if __name__ == "__main__":
    model_path = "../models/knn_item_model.pkl"
    model = load_model_knn(model_path)

    # Exemple amb un usuari real
    sample_user = list(model.user_index.keys())[0]
    print("\nUser seleccionat:", sample_user)

    # Predicció per un item concret
    example_item = list(model.item_index.keys())[0]
    rating_pred = predict_knn(sample_user, example_item, model)
    print(f"\nPredicció rating per l'item {example_item}: {rating_pred:.1f}")

    # Recomanacions top 10
    recs = recommend_knn(sample_user, model, top_n=10)
    print("\nTop 10 recomanacions:")
    for item, score in recs:
        print(f"{item}: {score:.1f}")

"""

