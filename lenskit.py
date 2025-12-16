import numpy as np
import pandas as pd
from lenskit import batch, topn
from lenskit.algorithms import Recommender
from lenskit.algorithms.item_knn import ItemItem
from data_cleaner import load_and_clean

# -----------------------------------------
#       Funcions de mètriques Top-K
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
            score += hits / (i + 1)
    return score / min(len(relevant), k)

def ndcg_k(recommended, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

# -----------------------------------------
#                 DATA
# -----------------------------------------

df = pd.read_csv("cleaned_data.csv")
df = df.rename(columns={"userID": "user", "itemID": "item", "rating": "rating"})
print("Dataset carregat:", df.shape)


test = df.groupby("user").tail(1)
train = df.drop(test.index)
train_items_per_user = train.groupby("user")["item"].apply(set).to_dict()
all_items = set(train["item"])

print("Train:", train.shape)
print("Test:", test.shape)

# -----------------------------------------
#            MODELS LensKit
# -----------------------------------------

print("\n=== ENTRENANT ItemItem KNN ===")
knn = Recommender.adapt(ItemItem(20))
knn.fit(train)

# MF no se usa en LensKit moderno

# -----------------------------------------
#            TEST + METRIQUES
# -----------------------------------------

K = 10
rmse_knn = 0
n_knn = 0
prec_knn = []; rec_knn = []; map_knn = []; ndcg_knn = []

print("\n=== INICIANT TEST ===")

for _, row in test.iterrows():
    u = row['user']
    true_item = row['item']
    true_rating = row['rating']
    relevant = [true_item]

    rated_items = train_items_per_user.get(u, set())
    unseen_items = list(all_items - rated_items)

    # --------------------------
    # KNN ItemItem
    # --------------------------
    pred_knn = knn.predict_for_user(u, [true_item]).iloc[0]
    if not np.isnan(pred_knn):
        rmse_knn += (pred_knn - true_rating) ** 2
        n_knn += 1

    scores_knn = knn.predict_for_user(u, unseen_items)
    scores_knn = scores_knn.sort_values(ascending=False)
    rec_knn_u = list(scores_knn.head(K).index)

    prec_knn.append(precision_k(rec_knn_u, relevant, K))
    rec_knn.append(recall_k(rec_knn_u, relevant, K))
    map_knn.append(apk(rec_knn_u, relevant, K))
    ndcg_knn.append(ndcg_k(rec_knn_u, relevant, K))

# Final RMSE
rmse_knn = np.sqrt(rmse_knn / n_knn) if n_knn > 0 else np.nan

print("\n=========== RESULTATS LENSKIT ===========")
print("\n--- ItemItem KNN ---")
print(f"Precision@{K}: {np.mean(prec_knn):.4f}")
print(f"Recall@{K}:    {np.mean(rec_knn):.4f}")
print(f"MAP@{K}:       {np.mean(map_knn):.4f}")
print(f"NDCG@{K}:      {np.mean(ndcg_knn):.4f}")
print(f"RMSE:          {rmse_knn:.4f}")
print("\n========================================")
