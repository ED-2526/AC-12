# -*- coding: utf-8 -*-
"""
MAIN SURPRISE SCRIPT — KNN Item-Item i SVD (Funk-SVD)
Comparació directa amb els models propis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from surprise import Dataset, Reader, KNNBasic, SVD
from data_cleaner import load_and_clean  # el teu codi


# -----------------------------------------
#       Funcions de mètriques 
# -----------------------------------------

def precision_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def apk(recommended, relevant, k):
    score, hits = 0, 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i+1)
    return score / min(len(relevant), k)

def ndcg_k(recommended, relevant, k):
    dcg = sum(1/np.log2(i+2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# -----------------------------------------
#                 DATA
# -----------------------------------------

PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
df = load_and_clean(PATH)

print("Dataset carregat:", df.shape)

test = df.groupby("userID").tail(1)
train = df.drop(test.index)

test_users  = list(test["userID"])
test_items  = list(test["itemID"])
test_ratings = test.set_index("userID")["rating"].to_dict()

train_items_per_user = train.groupby("userID")["itemID"].apply(set).to_dict()
all_items = set(train["itemID"])

print("Train:", train.shape)
print("Test:", test.shape)


# -----------------------------------------
#     PREPARAR SURPRISE DATASET
# -----------------------------------------

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)
trainset = data.build_full_trainset()



# -----------------------------------------
#               Entrenar MODELS
# -----------------------------------------

print("\n=== ENTRENANT Surprise KNN item-item ===")
knn_algo = KNNBasic(
    k=20,
    sim_options={"name": "cosine", "user_based": False}
)
knn_algo.fit(trainset)


print("\n=== ENTRENANT Surprise SVD ===")
svd_algo = SVD(
    n_factors=20,
    n_epochs=15,
    biased=True,
)
svd_algo.fit(trainset)


# -----------------------------------------
#               TEST + METRIQUES
# -----------------------------------------

K = 10

rmse_knn = rmse_svd = 0
prec_knn = []; rec_knn = []; map_knn = []; ndcg_knn = []
prec_svd = []; rec_svd = []; map_svd = []; ndcg_svd = []

print("\n=== INICIANT TEST... ===")

for u, true_item in tqdm(zip(test_users, test_items), total=len(test_users)):

    true_rating = test_ratings[u]
    relevant = [true_item]

    rated_items = train_items_per_user.get(u, set())
    unseen_items = list(all_items - rated_items)

    # --------------------------
    # KNN
    # --------------------------

    pred_knn = knn_algo.predict(u, true_item).est
    rmse_knn += (pred_knn - true_rating)**2

    preds_knn = [(iid, knn_algo.predict(u, iid).est) for iid in unseen_items]
    preds_knn.sort(key=lambda x: x[1], reverse=True)
    rec_knn_u = [iid for iid, _ in preds_knn[:K]]

    prec_knn.append(precision_k(rec_knn_u, relevant, K))
    rec_knn.append(recall_k(rec_knn_u, relevant, K))
    map_knn.append(apk(rec_knn_u, relevant, K))
    ndcg_knn.append(ndcg_k(rec_knn_u, relevant, K))

    # --------------------------
    # SVD
    # --------------------------

    pred_svd = svd_algo.predict(u, true_item).est
    rmse_svd += (pred_svd - true_rating)**2

    preds_svd = [(iid, svd_algo.predict(u, iid).est) for iid in unseen_items]
    preds_svd.sort(key=lambda x: x[1], reverse=True)
    rec_svd_u = [iid for iid, _ in preds_svd[:K]]

    prec_svd.append(precision_k(rec_svd_u, relevant, K))
    rec_svd.append(recall_k(rec_svd_u, relevant, K))
    map_svd.append(apk(rec_svd_u, relevant, K))
    ndcg_svd.append(ndcg_k(rec_svd_u, relevant, K))


# Final metrics
N = len(test_users)
if N > 0:
    rmse_knn = np.sqrt(rmse_knn / N) 
    rmse_svd = np.sqrt(rmse_svd / N) 
else:
    rmse_knn = 0
    rmse_svd = 0

print("\n============== RESULTATS SURPRISE ==============")
print(f"\n--- KNN ---")
print(f"Precision@{K}: {np.mean(prec_knn):.4f}")
print(f"Recall@{K}:    {np.mean(rec_knn):.4f}")
print(f"MAP@{K}:       {np.mean(map_knn):.4f}")
print(f"NDCG@{K}:      {np.mean(ndcg_knn):.4f}")
print(f"RSME:         {rmse_svd:.4f}" if rmse_knn is not None else "RSME:         N/A")

print(f"\n--- SVD ---")
print(f"Precision@{K}: {np.mean(prec_svd):.4f}")
print(f"Recall@{K}:    {np.mean(rec_svd):.4f}")
print(f"MAP@{K}:       {np.mean(map_svd):.4f}")
print(f"NDCG@{K}:      {np.mean(ndcg_svd):.4f}")
print(f"RSME:           {rmse_svd:.4f}" if rmse_svd is not None else "RSME:         N/A")
print("\n================================================")


