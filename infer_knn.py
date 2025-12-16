
#KNN Prediction & Recommendation 

import pickle
import numpy as np
from train_knn import ItemItemKNN  # només la classe amb fit i topk_neighbors


def load_model(model_path="models/knn_item_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def clamp_rating(x, min_r=1, max_r=5):
    return max(min_r, min(max_r, x))


def to_int_rating(x):
    if x == 0:
        return 0
    return int(clamp_rating(round(x)))

# PREDICCIÓ KNN
def predict_knn(user_id, item_id, model):
    if item_id not in model.item_index or user_id not in model.user_index:
        return 0
    i_idx = model.item_index[item_id]
    u_idx = model.user_index[user_id]
    user_ratings = model.item_user_matrix[:, u_idx]
    if model.use_topk:
        neighbors = model.topk_neighbors[i_idx]
        if neighbors is None:
            return 0
        nbr_idx, nbr_sims = neighbors

        if len(nbr_idx) == 0:
            return 0
        nbr_ratings = user_ratings[nbr_idx]
        mask = nbr_ratings > 0
        if mask.sum() == 0:
            return 0
        sims = nbr_sims[mask]
        ratings = nbr_ratings[mask]
        if sims.sum() == 0:
            return 0
        pred = float(np.dot(sims, ratings) / sims.sum())
        return to_int_rating(pred)
    else:
        sims = model.topk_neighbors[i_idx]
        rated_mask = user_ratings > 0
        if rated_mask.sum() == 0:
            return 0
        sims_rated = sims[rated_mask]
        ratings_rated = user_ratings[rated_mask]
        if sims_rated.sum() == 0:
            return 0
        pred = float(np.dot(sims_rated, ratings_rated) / sims_rated.sum())
        return to_int_rating(pred)

# RECOMANACIONS TOP-N
def recommend_knn(user_id, model, top_n=10):
    if user_id not in model.user_index:
        return []
    u_idx = model.user_index[user_id]
    user_ratings = model.item_user_matrix[:, u_idx]
    unseen_items = np.where(user_ratings == 0)[0]
    preds = []
    for i in unseen_items:
        if model.use_topk:
            nbr_idx, nbr_sims = model.topk_neighbors[i]
            if nbr_idx is None or len(nbr_idx) == 0:
                continue
            nbr_ratings = user_ratings[nbr_idx]
            mask = nbr_ratings > 0
            if mask.sum() == 0:
                continue
            sims = nbr_sims[mask]
            ratings = nbr_ratings[mask]
            score = float(np.dot(sims, ratings) / (sims.sum() if sims.sum() != 0 else 1e-8))
        else:
            sims_row = model.topk_neighbors[i]
            rated_mask = user_ratings > 0
            if rated_mask.sum() == 0:
                continue
            sims = sims_row[rated_mask]
            ratings = user_ratings[rated_mask]
            score = float(np.dot(sims, ratings) / (sims.sum() if sims.sum() != 0 else 1e-8))
        preds.append((i, to_int_rating(score)))
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]
    return [(model.items[i], score) for i, score in top]

"""
if __name__ == "__main__":
    model_path = "../models/knn_item_model.pkl"
    model = load_model(model_path)
    sample_user = list(model.user_index.keys())[0]
    print("\nUser seleccionat:", sample_user)
    example_item = list(model.item_index.keys())[0]
    rating_pred = predict_knn(sample_user, example_item, model)
    print(f"\nPredicció rating per l'item {example_item}: {rating_pred}")
    recs = recommend_knn(sample_user, model, top_n=10)
    print("\nTop 10 recomanacions:")
    for item, score in recs:
        print(f"{item}: {score}")
"""

