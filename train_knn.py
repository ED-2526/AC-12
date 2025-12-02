# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:52:55 2025

@author: laura
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

class ItemItemKNN:
    """
    Item-Item KNN:
    - construeix una matriu item x user (items a files, users a columnes)
    - calcula similitud cosine entre items (o guarda només top_k veïns)
    - prediu la valoració d'un usuari per un item fent weighted avg sobre
      les valoracions de l'usuari sobre els items veïns.
    """

    def __init__(self, k=20, use_topk=True):
        self.k = k
        self.use_topk = use_topk
        self.item_user_matrix = None  # numpy array shape (n_items, n_users)
        self.items = None              # array-like: item ids in order of rows
        self.users = None              # array-like: user ids in order of cols
        self.item_index = {}           # item_id -> row index
        self.user_index = {}           # user_id -> col index
        self.topk_neighbors = None     # list of arrays: for each item, (idxs, sims)

    def build_matrix(self, df):
        # Factorize item and user ids to consistent indices
        user_codes, users = pd.factorize(df["userID"])
        item_codes, items = pd.factorize(df["itemID"])

        n_users = len(users)
        n_items = len(items)

        # create item x user matrix
        mat = np.zeros((n_items, n_users), dtype=np.float32)
        for u_code, i_code, r in zip(user_codes, item_codes, df["rating"]):
            mat[i_code, u_code] = r

        self.item_user_matrix = mat
        self.items = items
        self.users = users
        self.item_index = {item: idx for idx, item in enumerate(items)}
        self.user_index = {user: idx for idx, user in enumerate(users)}

    def fit(self, df):
        """
        df: DataFrame with columns userID, itemID, rating
        """
        self.build_matrix(df)
        
        # Limitació de seguretat: si hi ha massa usuaris/items: atura
        if len(self.items) > 30000 or len(self.users) > 30000:
            raise ValueError(
                f"Massa usuaris/items per entrenar KNN. "
                f"Items={len(self.items)}, Users={len(self.users)}. "
                "Redueix dataset amb filtres més estrictes."
            )
            
        # compute item-item similarity (cosine on rows)
        print("Computing item-item similarity (can take time)...")
        sim = cosine_similarity(self.item_user_matrix)  # shape (n_items, n_items)

        if self.use_topk:
            # For each item, keep top-k neighbors (excluding itself)
            n_items = sim.shape[0]
            self.topk_neighbors = [None] * n_items
            for i in range(n_items):
                sims = sim[i].copy()
                sims[i] = -1  # exclude self
                # get indices of top k similarities
                if self.k < len(sims):
                    topk_idx = np.argpartition(sims, -self.k)[-self.k:]
                    # sort those in descending order
                    topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]
                else:
                    # k >= n_items -> take all except self, sorted
                    topk_idx = np.argsort(sims)[::-1]
                topk_sims = sims[topk_idx]
                # keep only positive similarities to avoid weird signs (optional)
                positive_mask = topk_sims > 0
                topk_idx = topk_idx[positive_mask]
                topk_sims = topk_sims[positive_mask]
                self.topk_neighbors[i] = (topk_idx, topk_sims)
            # delete sim matrix to save memory, keep only topk
            del sim
        else:
            # keep full sim matrix (less memory efficient)
            self.topk_neighbors = sim


def train_itemknn(df, k=20, model_path="models/knn_item_model.pkl", use_topk=True):
    model = ItemItemKNN(k=k, use_topk=use_topk)
    model.fit(df)
    # ensure models directory exists
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model
        

if __name__ == "__main__":
    # example run (adjust path)
    from data_cleaner import load_and_clean
    PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
    df = load_and_clean(PATH)
    print("DF shape:", df.shape)
    print("Training item-item KNN (top-k)...")
    model = train_itemknn(df, k=20, model_path="../models/knn_item_model.pkl", use_topk=True)
    print("Saved model to ../models/knn_item_model.pkl")

