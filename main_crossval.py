# -*- coding: utf-8 -*-
"""
MAIN CROSS-VALIDATION SCRIPT
Avalua:
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
from sklearn.model_selection import KFold

from data_cleaner import load_and_clean
from train_knn import train_itemknn
from train_svd import train_svd_model
from train_svd_bias import train_svd_bias

from infer_knn import predict_knn, recommend_knn
from infer_svd import predict_svd, recommend_svd


# ----------------------------------------------------------
#              METRIQUES
# ----------------------------------------------------------

def precision_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k if k > 0 else 0.0

def recall_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant) if relevant else 0.0

def apk(recommended, relevant, k):
    score, hits = 0.0, 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0

def ndcg_k(recommended, relevant, k):
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------
#              MAIN CROSS VALIDATION
# ----------------------------------------------------------

if __name__ == "__main__":

    PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
    df = load_and_clean(PATH)

    N_FOLDS = 5
    K = 10

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Resultats globals
    results = {
        "knn": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
        "svd": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
        "svd_bias": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
    }

    print(f"\n=== INICIANT {N_FOLDS}-FOLD CROSS VALIDATION ===")

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")

        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        # Entrenament
        knn_model = train_itemknn(train, k=20, use_topk=True)

        svd_model, svd_users, svd_items = train_svd_model(
            train, k=20, epochs=15
        )
        svd_u2i = {u: i for i, u in enumerate(svd_users)}
        svd_i2i = {i: j for j, i in enumerate(svd_items)}

        svd_bias_model, svd_b_users, svd_b_items = train_svd_bias(
            train, k=20, epochs=15
        )
        svd_b_u2i = {u: i for i, u in enumerate(svd_b_users)}
        svd_b_i2i = {i: j for j, i in enumerate(svd_b_items)}

        # Per usuari: ítems vistos al train
        train_items_per_user = train.groupby("userID")["itemID"].apply(set).to_dict()
        all_items = set(train["itemID"])

        # Acumuladors RMSE
        rmse = {"knn": [], "svd": [], "svd_bias": []}

        # Loop test
        for _, row in test.iterrows():
            u = row["userID"]
            true_item = row["itemID"]
            true_rating = row["rating"]
            relevant = [true_item]

            rated = train_items_per_user.get(u, set())
            unseen = list(all_items - rated)

            # ---------- KNN ----------
            rec_knn = [i for i, _ in recommend_knn(u, knn_model, top_n=K)]
            results["knn"]["prec"].append(precision_k(rec_knn, relevant, K))
            results["knn"]["rec"].append(recall_k(rec_knn, relevant, K))
            results["knn"]["map"].append(apk(rec_knn, relevant, K))
            results["knn"]["ndcg"].append(ndcg_k(rec_knn, relevant, K))

            pred = predict_knn(u, true_item, knn_model)
            if pred != 0:
                rmse["knn"].append((pred - true_rating) ** 2)

            # ---------- SVD ----------
            rec_svd = [i for i, _ in recommend_svd(
                u, svd_model, svd_users, svd_items,
                svd_u2i, svd_i2i, top_n=K
            )]
            results["svd"]["prec"].append(precision_k(rec_svd, relevant, K))
            results["svd"]["rec"].append(recall_k(rec_svd, relevant, K))
            results["svd"]["map"].append(apk(rec_svd, relevant, K))
            results["svd"]["ndcg"].append(ndcg_k(rec_svd, relevant, K))

            pred = predict_svd(
                u, true_item, svd_model,
                svd_users, svd_items,
                svd_u2i, svd_i2i
            )
            if pred != 0:
                rmse["svd"].append((pred - true_rating) ** 2)

            # ---------- SVD BIAS ----------
            rec_b = [i for i, _ in recommend_svd(
                u, svd_bias_model, svd_b_users, svd_b_items,
                svd_b_u2i, svd_b_i2i, top_n=K
            )]
            results["svd_bias"]["prec"].append(precision_k(rec_b, relevant, K))
            results["svd_bias"]["rec"].append(recall_k(rec_b, relevant, K))
            results["svd_bias"]["map"].append(apk(rec_b, relevant, K))
            results["svd_bias"]["ndcg"].append(ndcg_k(rec_b, relevant, K))

            pred = predict_svd(
                u, true_item, svd_bias_model,
                svd_b_users, svd_b_items,
                svd_b_u2i, svd_b_i2i
            )
            if pred != 0:
                rmse["svd_bias"].append((pred - true_rating) ** 2)

        # RMSE per fold
        for m in rmse:
            if rmse[m]:
                results[m]["rmse"].append(np.sqrt(np.mean(rmse[m])))
            else:
                results[m]["rmse"].append(0.0)

    # ----------------------------------------------------------
    #              RESULTATS FINALS
    # ----------------------------------------------------------

    print("\n================ RESULTATS FINALS (CV) ================")
    for model in ["knn", "svd", "svd_bias"]:
        print(f"\n--- {model.upper()} ---")
        print(f"Precision@{K}: {np.mean(results[model]['prec']):.4f}")
        print(f"Recall@{K}:    {np.mean(results[model]['rec']):.4f}")
        print(f"MAP@{K}:       {np.mean(results[model]['map']):.4f}")
        print(f"NDCG@{K}:      {np.mean(results[model]['ndcg']):.4f}")
        print(f"RMSE:          {np.mean(results[model]['rmse']):.4f}")

    print("\n===============================================")
