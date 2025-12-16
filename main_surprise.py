# -*- coding: utf-8 -*-
"""
MAIN SURPRISE SCRIPT — Flexible LOO / K-Fold Cross-validation
Models:
- KNN Item-Item
- SVD (Funk-SVD)
Mètriques:
- Precision@K, Recall@K, MAP@K, NDCG@K, RMSE
"""

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import KFold
from data_cleaner import load_and_clean

# ----------------------------------------------------------
# Mètriques de ranking
# ----------------------------------------------------------
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
    dcg = sum(1/np.log2(i+2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ----------------------------------------------------------
# Configuració
# ----------------------------------------------------------
PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
K = 10
N_FOLDS = 5            # només per cross-validation
EVAL_MODE = "loo"      # "loo" o "crossval"

# ----------------------------------------------------------
# Carregar dataset
# ----------------------------------------------------------
df = load_and_clean(PATH, plot=False)

all_items = set(df["itemID"])
train_items_per_user = df.groupby("userID")["itemID"].apply(set).to_dict()

# ----------------------------------------------------------
# Preparar Surprise dataset
# ----------------------------------------------------------
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID','itemID','rating']], reader)
trainset_full = data.build_full_trainset()

# ----------------------------------------------------------
# Entrenar models sobre tot el trainset
# ----------------------------------------------------------
print("\n=== Entrenant KNN ===")
knn_algo = KNNBasic(k=20, sim_options={"name":"cosine", "user_based":False})
knn_algo.fit(trainset_full)

print("\n=== Entrenant SVD ===")
svd_algo = SVD(n_factors=40, n_epochs=15, biased=True)
svd_algo.fit(trainset_full)

# ----------------------------------------------------------
# Inicialitzar llistes de resultats
# ----------------------------------------------------------
metrics_knn = {"prec":[], "rec":[],"map":[],"ndcg":[]}
metrics_svd = {"prec":[], "rec":[],"map":[],"ndcg":[]}
rmse_knn_total = 0
rmse_svd_total = 0
rmse_count = 0

# ----------------------------------------------------------
# Funció auxiliar per obtenir els ítems no vistos
# ----------------------------------------------------------
def get_unseen_items(user_id, df):
    rated_items = set(df[df["userID"]==user_id]["itemID"])
    return list(all_items - rated_items)

# ----------------------------------------------------------
# Split Leave-One-Out per usuari
# ----------------------------------------------------------
if EVAL_MODE=="loo":
    test = df.groupby("userID").tail(1)
    train = df.drop(test.index)
    test_users = test["userID"].unique()
    
    for u in test_users:
        relevant = list(test[test["userID"]==u]["itemID"])
        unseen_items = list(all_items - set(train[train["userID"]==u]["itemID"]))

        # ---------- KNN ----------
        preds_knn = [(iid, knn_algo.predict(u,iid).est) for iid in unseen_items]
        preds_knn.sort(key=lambda x:x[1], reverse=True)
        rec_knn = [iid for iid,_ in preds_knn[:K]]
        metrics_knn["prec"].append(precision_k(rec_knn,relevant,K))
        metrics_knn["rec"].append(recall_k(rec_knn,relevant,K))
        metrics_knn["map"].append(apk(rec_knn,relevant,K))
        metrics_knn["ndcg"].append(ndcg_k(rec_knn,relevant,K))

        for iid in relevant:
            pred = knn_algo.predict(u,iid).est
            rmse_knn_total += (pred - float(test.loc[(test["userID"]==u)&(test["itemID"]==iid),"rating"]))**2
            rmse_count += 1

        # ---------- SVD ----------
        preds_svd = [(iid, svd_algo.predict(u,iid).est) for iid in unseen_items]
        preds_svd.sort(key=lambda x:x[1], reverse=True)
        rec_svd = [iid for iid,_ in preds_svd[:K]]
        metrics_svd["prec"].append(precision_k(rec_svd,relevant,K))
        metrics_svd["rec"].append(recall_k(rec_svd,relevant,K))
        metrics_svd["map"].append(apk(rec_svd,relevant,K))
        metrics_svd["ndcg"].append(ndcg_k(rec_svd,relevant,K))

        for iid in relevant:
            pred = svd_algo.predict(u,iid).est
            rmse_svd_total += (pred - float(test.loc[(test["userID"]==u)&(test["itemID"]==iid),"rating"]))**2

# ----------------------------------------------------------
# Split K-Fold Cross-validation per usuari
# ----------------------------------------------------------
elif EVAL_MODE=="crossval":
    kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
    for trainset_cv, testset_cv in kf.split(data):
        # Crear diccionari de ratings per usuari en fold
        test_df_cv = pd.DataFrame(testset_cv, columns=["userID","itemID","rating"])
        train_df_cv = pd.DataFrame(trainset_cv.build_testset(), columns=["userID","itemID","rating"])

        for u, group in test_df_cv.groupby("userID"):
            relevant = list(group["itemID"])
            unseen_items = list(all_items - set(train_df_cv[train_df_cv["userID"]==u]["itemID"]))

            # ---------- KNN ----------
            preds_knn = [(iid, knn_algo.predict(u,iid).est) for iid in unseen_items]
            preds_knn.sort(key=lambda x:x[1], reverse=True)
            rec_knn = [iid for iid,_ in preds_knn[:K]]
            metrics_knn["prec"].append(precision_k(rec_knn,relevant,K))
            metrics_knn["rec"].append(recall_k(rec_knn,relevant,K))
            metrics_knn["map"].append(apk(rec_knn,relevant,K))
            metrics_knn["ndcg"].append(ndcg_k(rec_knn,relevant,K))

            for iid in relevant:
                pred = knn_algo.predict(u,iid).est
                rmse_knn_total += (pred - float(group[group["itemID"]==iid]["rating"]))**2
                rmse_count += 1

            # ---------- SVD ----------
            preds_svd = [(iid, svd_algo.predict(u,iid).est) for iid in unseen_items]
            preds_svd.sort(key=lambda x:x[1], reverse=True)
            rec_svd = [iid for iid,_ in preds_svd[:K]]
            metrics_svd["prec"].append(precision_k(rec_svd,relevant,K))
            metrics_svd["rec"].append(recall_k(rec_svd,relevant,K))
            metrics_svd["map"].append(apk(rec_svd,relevant,K))
            metrics_svd["ndcg"].append(ndcg_k(rec_svd,relevant,K))

            for iid in relevant:
                pred = svd_algo.predict(u,iid).est
                rmse_svd_total += (pred - float(group[group["itemID"]==iid]["rating"]))**2

else:
    raise ValueError("EVAL_MODE ha de ser 'loo' o 'crossval'.")

# ----------------------------------------------------------
# Resultats finals
# ----------------------------------------------------------
N = rmse_count if rmse_count>0 else 1

print("\n============== RESULTATS ==============")
print(f"--- KNN ---")
print(f"Precision@{K}: {np.mean(metrics_knn['prec']):.4f}")
print(f"Recall@{K}:    {np.mean(metrics_knn['rec']):.4f}")
print(f"MAP@{K}:       {np.mean(metrics_knn['map']):.4f}")
print(f"NDCG@{K}:      {np.mean(metrics_knn['ndcg']):.4f}")
print(f"RMSE:          {np.sqrt(rmse_knn_total/N):.4f}")

print(f"\n--- SVD ---")
print(f"Precision@{K}: {np.mean(metrics_svd['prec']):.4f}")
print(f"Recall@{K}:    {np.mean(metrics_svd['rec']):.4f}")
print(f"MAP@{K}:       {np.mean(metrics_svd['map']):.4f}")
print(f"NDCG@{K}:      {np.mean(metrics_svd['ndcg']):.4f}")
print(f"RMSE:          {np.sqrt(rmse_svd_total/N):.4f}")
print("\n=======================================")
