# -*- coding: utf-8 -*-
"""
MAIN CROSS-VALIDATION SCRIPT

K-Fold Cross Validation (per usuari) per comparar:
- KNN
- SVD
- SVD amb bias
"""

import numpy as np
import pandas as pd

from data_cleaner import load_and_clean
from train_knn import train_itemknn
from train_svd import train_svd_model
from train_svd_bias import train_svd_bias

from infer_knn import predict_knn, recommend_knn
from infer_svd import predict_svd, recommend_svd


# ----------------------------------------------------------
#               MÈTRIQUES
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
#          CREAR FOLDS PER USUARI
# ----------------------------------------------------------

def user_kfold_split(df, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    folds = []

    for _, user_df in df.groupby("userID"):
        idx = user_df.index.to_numpy()
        rng.shuffle(idx)
        splits = np.array_split(idx, n_splits)

        for i in range(n_splits):
            if len(folds) <= i:
                folds.append([])
            folds[i].extend(splits[i])

    return folds


# ----------------------------------------------------------
#                  MAIN
# ----------------------------------------------------------

PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
K = 10
N_FOLDS = 5

df = load_and_clean(PATH)
print("Dataset:", df.shape)

folds = user_kfold_split(df, n_splits=N_FOLDS)

results = {
    "knn": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
    "svd": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
    "svd_bias": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
}

# ----------------------------------------------------------
#               LOOP FOLDS
# ----------------------------------------------------------

for f in range(N_FOLDS):
    print(f"\n=== FOLD {f+1}/{N_FOLDS} ===")

    test_idx = folds[f]
    test = df.loc[test_idx]
    train = df.drop(test_idx)

    # -------------------------
    # ENTRENAMENT
    # -------------------------

    knn_model = train_itemknn(train, k=20, use_topk=True)

    svd_model, svd_users, svd_items = train_svd_model(
        train, k=20, epochs=30
    )
    svd_u2i = {u: i for i, u in enumerate(svd_users)}
    svd_i2i = {i: j for j, i in enumerate(svd_items)}

    svd_bias_model, sb_users, sb_items = train_svd_bias(
        train, k=20, epochs=30
    )
    sb_u2i = {u: i for i, u in enumerate(sb_users)}
    sb_i2i = {i: j for j, i in enumerate(sb_items)}

    # -------------------------
    # TEST
    # -------------------------

    rmse_knn = rmse_svd = rmse_bias = 0
    count = 0

    for _, row in test.iterrows():
        u, true_item, true_rating = row["userID"], row["itemID"], row["rating"]
        relevant = [true_item]

        # ---------- KNN ----------
        rec_knn = [i for i, _ in recommend_knn(u, knn_model, top_n=K)]
        results["knn"]["prec"].append(precision_k(rec_knn, relevant, K))
        results["knn"]["rec"].append(recall_k(rec_knn, relevant, K))
        results["knn"]["map"].append(apk(rec_knn, relevant, K))
        results["knn"]["ndcg"].append(ndcg_k(rec_knn, relevant, K))

        pred = predict_knn(u, true_item, knn_model)
        if pred != 0:
            rmse_knn += (pred - true_rating) ** 2

        # ---------- SVD ----------
        rec_svd = [i for i, _ in recommend_svd(
            u, svd_model, svd_users, svd_items, svd_u2i, svd_i2i, K)]
        results["svd"]["prec"].append(precision_k(rec_svd, relevant, K))
        results["svd"]["rec"].append(recall_k(rec_svd, relevant, K))
        results["svd"]["map"].append(apk(rec_svd, relevant, K))
        results["svd"]["ndcg"].append(ndcg_k(rec_svd, relevant, K))

        pred = predict_svd(u, true_item, svd_model, svd_users, svd_items, svd_u2i, svd_i2i)
        rmse_svd += (pred - true_rating) ** 2

        # ---------- SVD BIAS ----------
        rec_sb = [i for i, _ in recommend_svd(
            u, svd_bias_model, sb_users, sb_items, sb_u2i, sb_i2i, K)]
        results["svd_bias"]["prec"].append(precision_k(rec_sb, relevant, K))
        results["svd_bias"]["rec"].append(recall_k(rec_sb, relevant, K))
        results["svd_bias"]["map"].append(apk(rec_sb, relevant, K))
        results["svd_bias"]["ndcg"].append(ndcg_k(rec_sb, relevant, K))

        pred = predict_svd(u, true_item, svd_bias_model, sb_users, sb_items, sb_u2i, sb_i2i)
        rmse_bias += (pred - true_rating) ** 2

        count += 1

    results["knn"]["rmse"].append(np.sqrt(rmse_knn / count))
    results["svd"]["rmse"].append(np.sqrt(rmse_svd / count))
    results["svd_bias"]["rmse"].append(np.sqrt(rmse_bias / count))


# ----------------------------------------------------------
#                 RESULTATS FINALS
# ----------------------------------------------------------

print("\n================ CROSS-VALIDATION RESULTS ================")

for model in results:
    print(f"\n--- {model.upper()} ---")
    for m in ["prec", "rec", "map", "ndcg", "rmse"]:
        vals = results[model][m]
        print(f"{m.upper():8}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

print("\n=========================================================")
