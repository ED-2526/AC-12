# -*- coding: utf-8 -*-
"""
MAIN CROSS-VALIDATION SCRIPT (CORREGIT)

K-Fold Cross Validation (per usuari) amb avaluació per USUARI
i múltiples rellevants per fold.

Models:
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
import os
import numpy as np
import pandas as pd

from data_cleaner import load_and_clean
from train_knn import train_itemknn
from train_svd import train_svd_model
from train_svd_bias import train_svd_bias

from infer_knn import predict_knn, recommend_knn
from infer_svd import predict_svd, recommend_svd


# ----------------------------------------------------------
#               MÈTRIQUES DE RANKING
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
    score, hits = 0.0, 0
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
#          MITJANA I DESVIACIÓ (IGNORANT None)
# ----------------------------------------------------------

def mean_std_ignore_none(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return None, None
    return np.mean(vals), np.std(vals)


# ----------------------------------------------------------
#          CREAR FOLDS PER USUARI
# ----------------------------------------------------------

def user_kfold_split(df, n_splits=5, seed=42):
    """
    Divideix els ratings de cada usuari en n_splits,
    assegurant que cada fold tingui exemples de tots els usuaris.
    """
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]

    for _, user_df in df.groupby("userID"):
        idx = user_df.index.to_numpy()
        rng.shuffle(idx)
        splits = np.array_split(idx, n_splits)
        for i in range(n_splits):
            folds[i].extend(splits[i])

    return folds


# ----------------------------------------------------------
#                       MAIN
# ----------------------------------------------------------

PATH = r"ratings_Electronics.csv"
K = 10
N_FOLDS = 10

CLEAN_PATH = "cleaned_data.csv"
if os.path.exists(CLEAN_PATH):
    print("cleaned_data.csv trobat. Carregant dataset netejat...")
    df = pd.read_csv(CLEAN_PATH)
else:
    print("No existeix cleaned_data.csv. Netejant dataset original...")
    df = load_and_clean(PATH) 
print("Dataset:", df.shape)

folds = user_kfold_split(df, n_splits=N_FOLDS)

results = {
    "knn": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
    "svd": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
    "svd_bias": {"prec": [], "rec": [], "map": [], "ndcg": [], "rmse": []},
}

# ----------------------------------------------------------
#               LOOP DE CROSS-VALIDATION
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
        train, k=40, epochs=20
    )
    svd_u2i = {u: i for i, u in enumerate(svd_users)}
    svd_i2i = {i: j for j, i in enumerate(svd_items)}

    svd_bias_model, sb_users, sb_items = train_svd_bias(
        train, k=40, epochs=30
    )
    sb_u2i = {u: i for i, u in enumerate(sb_users)}
    sb_i2i = {i: j for j, i in enumerate(sb_items)}

    # -------------------------
    # TEST PER USUARI
    # -------------------------

    rmse_knn = rmse_svd = rmse_bias = 0.0
    rmse_count = 0

    for u, user_test in test.groupby("userID"):

        relevant = list(user_test["itemID"])

        # ---------- KNN ----------
        rec_knn = [i for i, _ in recommend_knn(u, knn_model, top_n=K)]
        results["knn"]["prec"].append(precision_k(rec_knn, relevant, K))
        results["knn"]["rec"].append(recall_k(rec_knn, relevant, K))
        results["knn"]["map"].append(apk(rec_knn, relevant, K))
        results["knn"]["ndcg"].append(ndcg_k(rec_knn, relevant, K))

        for _, row in user_test.iterrows():
            pred = predict_knn(u, row["itemID"], knn_model)
            if pred != 0:
                rmse_knn += (pred - row["rating"]) ** 2
                rmse_count += 1

        # ---------- SVD ----------
        rec_svd = [i for i, _ in recommend_svd(
            u, svd_model, svd_users, svd_items, svd_u2i, svd_i2i, K)]
        results["svd"]["prec"].append(precision_k(rec_svd, relevant, K))
        results["svd"]["rec"].append(recall_k(rec_svd, relevant, K))
        results["svd"]["map"].append(apk(rec_svd, relevant, K))
        results["svd"]["ndcg"].append(ndcg_k(rec_svd, relevant, K))

        for _, row in user_test.iterrows():
            pred = predict_svd(
                u, row["itemID"], svd_model,
                svd_users, svd_items, svd_u2i, svd_i2i
            )
            rmse_svd += (pred - row["rating"]) ** 2

        # ---------- SVD amb bias ----------
        rec_sb = [i for i, _ in recommend_svd(
            u, svd_bias_model, sb_users, sb_items, sb_u2i, sb_i2i, K)]
        results["svd_bias"]["prec"].append(precision_k(rec_sb, relevant, K))
        results["svd_bias"]["rec"].append(recall_k(rec_sb, relevant, K))
        results["svd_bias"]["map"].append(apk(rec_sb, relevant, K))
        results["svd_bias"]["ndcg"].append(ndcg_k(rec_sb, relevant, K))

        for _, row in user_test.iterrows():
            pred = predict_svd(
                u, row["itemID"], svd_bias_model,
                sb_users, sb_items, sb_u2i, sb_i2i
            )
            rmse_bias += (pred - row["rating"]) ** 2

    # -------------------------
    # RMSE PER FOLD
    # -------------------------

    if rmse_count > 0:
        results["knn"]["rmse"].append(np.sqrt(rmse_knn / rmse_count))
        results["svd"]["rmse"].append(np.sqrt(rmse_svd / rmse_count))
        results["svd_bias"]["rmse"].append(np.sqrt(rmse_bias / rmse_count))
    else:
        results["knn"]["rmse"].append(None)
        results["svd"]["rmse"].append(None)
        results["svd_bias"]["rmse"].append(None)


# ----------------------------------------------------------
#                 RESULTATS FINALS
# ----------------------------------------------------------

print("\n================ CROSS-VALIDATION RESULTS ================")

for model in results:
    print(f"\n--- {model.upper()} ---")
    for m in ["prec", "rec", "map", "ndcg", "rmse"]:
        mean, std = mean_std_ignore_none(results[model][m])
        if mean is None:
            print(f"{m.upper():8}: N/A")
        else:
            print(f"{m.upper():8}: {mean:.4f} ± {std:.4f}")

print("\n=========================================================")

