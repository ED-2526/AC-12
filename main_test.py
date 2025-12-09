# -*- coding: utf-8 -*-
"""
MAIN TEST SCRIPT — amb gràfics
Avalua i compara:
- KNN
- SVD
- SVD amb bias

Mètriques:
- Precision@K
- Recall@K
- MAP@K
- NDCG@K
- RMSE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_cleaner import load_and_clean
from train_knn import train_itemknn
from train_svd import train_svd_model
from train_svd_bias import train_svd_bias

from infer_knn import predict_knn as knn_predict, recommend_knn as knn_recommend
from infer_svd import predict_svd, recommend_svd


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

# Leave-One-Out split per usuari
test = df.groupby("userID").tail(1)
train = df.drop(test.index)

test_users = list(test["userID"])
test_items = list(test["itemID"])

print("Train:", train.shape)
print("Test:", test.shape)


# ----------------------------------------------------------
#                 ENTRENAMENT DELS 3 MODELS
# ----------------------------------------------------------

print("\n=== ENTRENANT KNN ===")
knn_model = train_itemknn(train, k=20, model_path="models/knn_item_model.pkl", use_topk=True)

print("\n=== ENTRENANT SVD (sense bias) ===")
svd_model, svd_users, svd_items = train_svd_model(
    train,
    k=20,
    epochs=15,
    model_path="models/svd_model.pkl"
)
svd_user_to_idx = {u: i for i, u in enumerate(svd_users)}
svd_item_to_idx = {it: j for j, it in enumerate(svd_items)}

print("\n=== ENTRENANT SVD (amb bias) ===")
svd_bias_model, svd_bias_users, svd_bias_items = train_svd_bias(
    train,
    k=20,
    epochs=15,
    model_path="models/svd_model_bias.pkl"
)
svd_bias_user_to_idx = {u: i for i, u in enumerate(svd_bias_users)}
svd_bias_item_to_idx = {it: j for j, it in enumerate(svd_bias_items)}


# ----------------------------------------------------------
#                   TEST & METRIQUES
# ----------------------------------------------------------

K = 10

# Errors RMSE
rmse_knn_sum = 0
rmse_svd_sum = 0
rmse_bias_sum = 0
rmse_count = 0

# Llistes de mètriques
metrics = {
    "knn": {"prec": [], "rec": [], "map": [], "ndcg": []},
    "svd": {"prec": [], "rec": [], "map": [], "ndcg": []},
    "svd_bias": {"prec": [], "rec": [], "map": [], "ndcg": []},
}

print("\n=== INICIANT TEST... ===")

for u, true_item in zip(test_users, test_items):
    true_rating = float(test.loc[test["userID"] == u, "rating"])

    # -------------------------
    # KNN
    # -------------------------
    rec_knn_u = [iid for iid, _ in knn_recommend(u, knn_model, top_n=K)]

    metrics["knn"]["prec"].append(precision_k(rec_knn_u, [true_item], K))
    metrics["knn"]["rec"].append(recall_k(rec_knn_u, [true_item], K))
    metrics["knn"]["map"].append(apk(rec_knn_u, [true_item], K))
    metrics["knn"]["ndcg"].append(ndcg_k(rec_knn_u, [true_item], K))

    pred_knn = knn_predict(u, true_item, knn_model)
    if pred_knn != 0:
        rmse_knn_sum += (pred_knn - true_rating) ** 2
        rmse_count += 1


    # -------------------------
    # SVD (sense bias)
    # -------------------------
    rec_svd_u = [iid for iid, _ in recommend_svd(
        u, svd_model, svd_users, svd_items,
        svd_user_to_idx, svd_item_to_idx, top_n=K)]

    metrics["svd"]["prec"].append(precision_k(rec_svd_u, [true_item], K))
    metrics["svd"]["rec"].append(recall_k(rec_svd_u, [true_item], K))
    metrics["svd"]["map"].append(apk(rec_svd_u, [true_item], K))
    metrics["svd"]["ndcg"].append(ndcg_k(rec_svd_u, [true_item], K))

    pred_svd = predict_svd(
        u, true_item, svd_model,
        svd_users, svd_items,
        svd_user_to_idx, svd_item_to_idx)

    if pred_svd != 0:
        rmse_svd_sum += (pred_svd - true_rating) ** 2


    # -------------------------
    # SVD amb bias
    # -------------------------
    rec_bias_u = [iid for iid, _ in recommend_svd(
        u, svd_bias_model, svd_bias_users, svd_bias_items,
        svd_bias_user_to_idx, svd_bias_item_to_idx, top_n=K)]

    metrics["svd_bias"]["prec"].append(precision_k(rec_bias_u, [true_item], K))
    metrics["svd_bias"]["rec"].append(recall_k(rec_bias_u, [true_item], K))
    metrics["svd_bias"]["map"].append(apk(rec_bias_u, [true_item], K))
    metrics["svd_bias"]["ndcg"].append(ndcg_k(rec_bias_u, [true_item], K))

    pred_bias = predict_svd(
        u, true_item, svd_bias_model,
        svd_bias_users, svd_bias_items,
        svd_bias_user_to_idx, svd_bias_item_to_idx)

    if pred_bias != 0:
        rmse_bias_sum += (pred_bias - true_rating) ** 2



# ----------------------------------------------------------
#                 RESULTATS FINALS
# ----------------------------------------------------------

if rmse_count != 0:
    rmse_knn = np.sqrt(rmse_knn_sum / rmse_count)
    rmse_svd = np.sqrt(rmse_svd_sum / rmse_count)
    rmse_bias = np.sqrt(rmse_bias_sum / rmse_count)
else: # quan es dona divisió per 0 (rsme_count=0), el model no pot generar cap predicció
    rmse_knn = None
    rmse_svd = None
    rmse_bias = None

print("\n================ RESULTATS ================")
for model_name, name_print in [
    ("knn", "KNN"),
    ("svd", "SVD"),
    ("svd_bias", "SVD amb bias")
]:
    print(f"\n--- {name_print} ---")
    print(f"Precision@{K}: {np.mean(metrics[model_name]['prec']):.4f}")
    print(f"Recall@{K}:    {np.mean(metrics[model_name]['rec']):.4f}")
    print(f"MAP@{K}:       {np.mean(metrics[model_name]['map']):.4f}")
    print(f"NDCG@{K}:      {np.mean(metrics[model_name]['ndcg']):.4f}")

print("\n--- RMSE ---")
print(f"KNN:         {rmse_knn:.4f}")
print(f"SVD:         {rmse_svd:.4f}")
print(f"SVD Bias:    {rmse_bias:.4f}")
print("\n===========================================")


# ----------------------------------------------------------
#              GRÀFICS COMPARATIUS
# ----------------------------------------------------------

def plot_bar3(a, b, c, title, ylabel):
    plt.figure(figsize=(6,4))
    plt.bar(["KNN", "SVD", "SVD+B"], [a, b, c])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

plot_bar3(np.mean(metrics["knn"]["prec"]),
          np.mean(metrics["svd"]["prec"]),
          np.mean(metrics["svd_bias"]["prec"]),
          "Precision@K Comparison", "Precision@K")

plot_bar3(np.mean(metrics["knn"]["rec"]),
          np.mean(metrics["svd"]["rec"]),
          np.mean(metrics["svd_bias"]["rec"]),
          "Recall@K Comparison", "Recall@K")

plot_bar3(np.mean(metrics["knn"]["map"]),
          np.mean(metrics["svd"]["map"]),
          np.mean(metrics["svd_bias"]["map"]),
          "MAP@K Comparison", "MAP@K")

plot_bar3(np.mean(metrics["knn"]["ndcg"]),
          np.mean(metrics["svd"]["ndcg"]),
          np.mean(metrics["svd_bias"]["ndcg"]),
          "NDCG@K Comparison", "NDCG@K")

plot_bar3(rmse_knn, rmse_svd, rmse_bias,
          "RMSE Comparison", "RMSE")

print("\nGràfics generats correctament.")



