
#Unified SVD / SVD-Bias Prediction & Recommendation

import pickle
import numpy as np
from train_svd import MatrixFactorization
from train_svd_bias import MatrixFactorizationBias

# LOAD MODEL
def load_model(model_path="models/svd_model.pkl"):
    """
    Carrega model + vocabularis.
    Construeix diccionaris inversos per accés O(1).
    """
    with open(model_path, "rb") as f:
        model, users, items = pickle.load(f)
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: j for j, it in enumerate(items)}
    return model, users, items, user_to_idx, item_to_idx


def clamp_rating(x, min_r=1, max_r=5):
    return max(min_r, min(max_r, x))


def to_int_rating(x):
    return int(clamp_rating(round(float(x))))

# PREDICCIÓ SVD (detecta automàticament si hi ha bias)
def predict_svd(user_id, item_id, model, users, items,
                user_to_idx=None, item_to_idx=None):
    if user_to_idx is None or item_to_idx is None:
        u_idx = list(users).index(user_id)
        i_idx = list(items).index(item_id)
    else:
        if user_id not in user_to_idx or item_id not in item_to_idx:
            return 0
        u_idx = user_to_idx[user_id]
        i_idx = item_to_idx[item_id]
    if isinstance(model, MatrixFactorization): #Model sense bias 
        pred = np.dot(model.P[u_idx], model.Q[i_idx])
    elif isinstance(model, MatrixFactorizationBias): # Model amb bias 
        pred = model.mu + model.b_u[u_idx] + model.b_i[i_idx] + np.dot(model.P[u_idx], model.Q[i_idx])
    else:
        raise ValueError("Model SVD desconegut!")
    return to_int_rating(pred)

# TOP-N RECOMMENDATIONS (automàtic per bias/sense bias)
def recommend_svd(user_id, model, users, items,
                  user_to_idx=None, item_to_idx=None,
                  top_n=10,
                  exclude_rated=True,
                  rated_items_idx=None):
    if user_to_idx is None or item_to_idx is None:
        u_idx = list(users).index(user_id)
    else:
        if user_id not in user_to_idx:
            return []
        u_idx = user_to_idx[user_id]
    if isinstance(model, MatrixFactorization): #Model sense bias 
        scores = np.dot(model.P[u_idx], model.Q.T)
    elif isinstance(model, MatrixFactorizationBias): # Model amb bias 
        scores = model.mu + model.b_u[u_idx] + model.b_i + np.dot(model.P[u_idx], model.Q.T)
    if exclude_rated and rated_items_idx is not None: # Excloure ítems ja valorats
        scores[rated_items_idx] = -np.inf
    top_idx = np.argpartition(scores, -top_n)[-top_n:] # Top N índexs
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(items[i], to_int_rating(scores[i])) for i in top_idx]

"""
if __name__ == "__main__":
    model_path = "../models/svd_model.pkl"           # sense bias
    # model_path = "../models/svd_model_bias.pkl"    # amb bias
    model, users, items, user_to_idx, item_to_idx = load_model(model_path)
    sample_user = users[0]
    example_item = items[0]
    print("\nUser seleccionat:", sample_user)
    pred = predict_svd(sample_user, example_item, model, users, items,
                       user_to_idx, item_to_idx)
    print(f"\nPredicció rating per l'item {example_item}: {pred}")
    recs = recommend_svd(sample_user, model, users, items,
                         user_to_idx, item_to_idx, top_n=10)
    print("\nTop 10 recomanacions:")
    for item, score in recs:
        print(f"{item}: {score}")
"""

