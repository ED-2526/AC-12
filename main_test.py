# -*- coding: utf-8 -*-
"""
MAIN TEST SCRIPT — amb gràfics
Avalua i compara KNN i SVD amb:
- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- RMSE
Inclou gràfics comparatius.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_cleaner import load_and_clean
from train_knn import train_itemknn
from train_svd import train_svd_model
from infer_knn import predict as knn_predict, recommend as knn_recommend
from infer_svd import predict as svd_predict, recommend as svd_recommend

# ----------------------------------------------------------
#              HELPER: METRIQUES DE RECOMANACIÓ
# ----------------------------------------------------------

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


# ----------------------------------------------------------
#                 PREPARACIÓ DEL DATASET
# ----------------------------------------------------------

PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
df = load_and_clean(PATH)
print("Dataset carregat:", df.shape)

# Leave-One-Out
test = df.groupby("userID").tail(1)
train = df.drop(test.index)

test_users = list(test["userID"])
test_items = list(test["itemID"])

print("Train:", train.shape)
print("Test:", test.shape)


# ----------------------------------------------------------
#                 ENTRENAMENT MODELS
# ----------------------------------------------------------

print("\n=== ENTRENANT KNN ===")
knn_model = train_itemknn(train, k=20, model_path="models/knn_item_model.pkl", use_topk=True)

print("\n=== ENTRENANT SVD ===")
svd_model, svd_users, svd_items = train_svd_model(
    train,
    k=20,
    epochs=15,
    model_path="models/svd_model.pkl"
)


# ----------------------------------------------------------
#                   TEST & METRIQUES
# ----------------------------------------------------------

K = 10

rmse_knn_sum = 0
rmse_svd_sum = 0
rmse_count = 0

prec_knn, rec_knn, map_knn, ndcg_knn = [], [], [], []
prec_svd, rec_svd, map_svd, ndcg_svd = [], [], [], []

print("\n=== INICIANT TEST... ===")

for u, true_item in zip(test_users, test_items):

    # -------------------------
    # KNN
    # -------------------------
    rec_knn_u = [iid for iid, _ in knn_recommend(u, knn_model, top_n=K)]
    prec_knn.append(precision_k(rec_knn_u, [true_item], K))
    rec_knn.append(recall_k(rec_knn_u, [true_item], K))
    map_knn.append(apk(rec_knn_u, [true_item], K))
    ndcg_knn.append(ndcg_k(rec_knn_u, [true_item], K))

    true_rating = float(test.loc[test["userID"] == u, "rating"])
    pred_knn = knn_predict(u, true_item, knn_model)
    if pred_knn != 0:
        rmse_knn_sum += (pred_knn - true_rating) ** 2
        rmse_count += 1

    # -------------------------
    # SVD
    # -------------------------
    rec_svd_u = [iid for iid, _ in svd_recommend(u, svd_model, svd_users, svd_items, top_n=K)]
    prec_svd.append(precision_k(rec_svd_u, [true_item], K))
    rec_svd.append(recall_k(rec_svd_u, [true_item], K))
    map_svd.append(apk(rec_svd_u, [true_item], K))
    ndcg_svd.append(ndcg_k(rec_svd_u, [true_item], K))

    pred_svd = svd_predict(u, true_item, svd_model, svd_users, svd_items)
    if pred_svd != 0:
        rmse_svd_sum += (pred_svd - true_rating) ** 2


# ----------------------------------------------------------
#                 RESULTATS FINALS
# ----------------------------------------------------------

if rmse_count != 0:
    rmse_knn = np.sqrt(rmse_knn_sum / rmse_count)
    rmse_svd = np.sqrt(rmse_svd_sum / rmse_count)
else:
    rmse_knn = 0
    rmse_svd = 0

print("\n================ RESULTATS ================")
print("\n--- KNN ---")
print(f"Precision@{K}: {np.mean(prec_knn):.4f}")
print(f"Recall@{K}:    {np.mean(rec_knn):.4f}")
print(f"MAP@{K}:       {np.mean(map_knn):.4f}")
print(f"NDCG@{K}:      {np.mean(ndcg_knn):.4f}")
print(f"RMSE:          {rmse_knn:.4f}")

print("\n--- SVD ---")
print(f"Precision@{K}: {np.mean(prec_svd):.4f}")
print(f"Recall@{K}:    {np.mean(rec_svd):.4f}")
print(f"MAP@{K}:       {np.mean(map_svd):.4f}")
print(f"NDCG@{K}:      {np.mean(ndcg_svd):.4f}")
print(f"RMSE:          {rmse_svd:.4f}")
print("\n===========================================")


# ----------------------------------------------------------
#              GRÀFICS COMPARATIUS
# ----------------------------------------------------------

def plot_bar(metric_knn, metric_svd, title, ylabel):
    plt.figure(figsize=(6,4))
    plt.bar(["KNN", "SVD"], [metric_knn, metric_svd])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

# --- METRIQUES AGREGADES ---
plot_bar(np.mean(prec_knn), np.mean(prec_svd), "Precision@K Comparison", "Precision@K")
plot_bar(np.mean(rec_knn), np.mean(rec_svd), "Recall@K Comparison", "Recall@K")
plot_bar(np.mean(map_knn), np.mean(map_svd), "MAP@K Comparison", "MAP@K")
plot_bar(np.mean(ndcg_knn), np.mean(ndcg_svd), "NDCG@K Comparison", "NDCG@K")
plot_bar(rmse_knn, rmse_svd, "RMSE Comparison", "RMSE")

print("\nGràfics generats correctament.")

