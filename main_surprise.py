# New script: main_surprise.py (using scikit-surprise library for comparison)
# Install surprise with: pip install scikit-surprise
# Run this separately to compare with your custom models.

# -*- coding: utf-8 -*-
"""
MAIN SURPRISE SCRIPT — using scikit-surprise for KNN and SVD comparison
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.accuracy import rmse as surprise_rmse
from data_cleaner import load_and_clean  # Reuse your data cleaner

# Reuse your metric helpers from main_test.py
def precision_k(recommended, relevant, k):
    if len(recommended) == 0:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def apk(recommended, relevant, k):
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    if len(relevant) == 0:
        return 0.0
    return score / min(len(relevant), k)

def ndcg_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

# Load data same as before
PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
df = load_and_clean(PATH)
print("Dataset carregat:", df.shape)

# Leave-One-Out split same
test = df.groupby("userID").tail(1)
train = df.drop(test.index)
test_users = list(test["userID"])
test_items = list(test["itemID"])
test_ratings = test.set_index("userID")["rating"].to_dict()
RELEVANT_THRESHOLD = 4.0
user_relevant_items = train[train["rating"] >= RELEVANT_THRESHOLD].groupby("userID")["itemID"].apply(set).to_dict()
user_rated_items = train.groupby("userID")["itemID"].apply(list).to_dict()
print("Train:", train.shape)
print("Test:", test.shape)

# Prepare Surprise dataset from train
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)
trainset = data.build_full_trainset()

# Train Surprise KNN (item-item)
print("\n=== ENTRENANT Surprise KNN ===")
knn_algo = KNNBasic(k=20, sim_options={'name': 'cosine', 'user_based': False})  # item-based
knn_algo.fit(trainset)

# Train Surprise SVD (biased)
print("\n=== ENTRENANT Surprise SVD ===")
svd_algo = SVD(n_factors=20, n_epochs=15, lr_all=0.01, reg_all=0.1, biased=True)
svd_algo.fit(trainset)

# Test & Metrics
K = 20
rmse_knn_sum = 0
rmse_svd_sum = 0
rmse_count = 0
prec_knn, rec_knn, map_knn, ndcg_knn = [], [], [], []
prec_svd, rec_svd, map_svd, ndcg_svd = [], [], [], []
print("\n=== INICIANT TEST... ===")
for u, true_item in tqdm(zip(test_users, test_items), total=len(test_users), desc="Avaluant"):
    true_rating = test_ratings[u]
    compute_ranking = true_rating >= RELEVANT_THRESHOLD
    relevant_set = user_relevant_items.get(u, set())
    if compute_ranking:
        relevant_set = relevant_set | {true_item}
    relevant = list(relevant_set)
    
    rated = user_rated_items.get(u, [])
    
    # KNN Surprise
    pred_knn = knn_algo.predict(u, true_item).est
    rmse_knn_sum += (pred_knn - true_rating) ** 2
    
    # Get top K for KNN (exclude rated)
    all_items = train['itemID'].unique()
    unseen = [iid for iid in all_items if iid not in rated]
    knn_preds = [(iid, knn_algo.predict(u, iid).est) for iid in unseen]
    knn_preds.sort(key=lambda x: x[1], reverse=True)
    rec_knn_u = [iid for iid, _ in knn_preds[:K]]
    
    # SVD Surprise
    pred_svd = svd_algo.predict(u, true_item).est
    rmse_svd_sum += (pred_svd - true_rating) ** 2
    
    svd_preds = [(iid, svd_algo.predict(u, iid).est) for iid in unseen]
    svd_preds.sort(key=lambda x: x[1], reverse=True)
    rec_svd_u = [iid for iid, _ in svd_preds[:K]]
    
    rmse_count += 1
    
    if compute_ranking:
        prec_knn.append(precision_k(rec_knn_u, relevant, K))
        rec_knn.append(recall_k(rec_knn_u, relevant, K))
        map_knn.append(apk(rec_knn_u, relevant, K))
        ndcg_knn.append(ndcg_k(rec_knn_u, relevant, K))
        
        prec_svd.append(precision_k(rec_svd_u, relevant, K))
        rec_svd.append(recall_k(rec_svd_u, relevant, K))
        map_svd.append(apk(rec_svd_u, relevant, K))
        ndcg_svd.append(ndcg_k(rec_svd_u, relevant, K))

# Results
rmse_knn = np.sqrt(rmse_knn_sum / rmse_count) if rmse_count > 0 else 0
rmse_svd = np.sqrt(rmse_svd_sum / rmse_count) if rmse_count > 0 else 0
print("\n================ RESULTATS SURPRISE ================")
print("\n--- KNN ---")
print(f"Precision@{K}: {np.mean(prec_knn):.4f}")
print(f"Recall@{K}: {np.mean(rec_knn):.4f}")
print(f"MAP@{K}: {np.mean(map_knn):.4f}")
print(f"NDCG@{K}: {np.mean(ndcg_knn):.4f}")
print(f"RMSE: {rmse_knn:.4f}")
print(f"Usuaris avaluats per ranking: {len(prec_knn)}/{len(test_users)}")
print("\n--- SVD ---")
print(f"Precision@{K}: {np.mean(prec_svd):.4f}")
print(f"Recall@{K}: {np.mean(rec_svd):.4f}")
print(f"MAP@{K}: {np.mean(map_svd):.4f}")
print(f"NDCG@{K}: {np.mean(ndcg_svd):.4f}")
print(f"RMSE: {rmse_svd:.4f}")
print(f"Usuaris avaluats per ranking: {len(prec_svd)}/{len(test_users)}")
print("\n====================================================")

# Reuse plot_bar from main_test.py if needed