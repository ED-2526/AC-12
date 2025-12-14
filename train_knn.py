import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemKNN:
    """
    - construeix una matriu item x user (items=files, users=columnes)
    - construeix una matriu item x item
    - calcula similitud cosine entre items (o guarda només top_k veïns)
    - prediu la valoració d'un usuari per un item fent weighted avg sobre les valoracions de l'usuari sobre els items veïns.
    """

    def __init__(self, k=20, use_topk=True):
        self.k = k
        self.use_topk = use_topk
        self.item_user_matrix = None   # numpy array (items x users)
        self.items = None              # arra: item ids in order of rows
        self.users = None              # array: user ids in order of cols
        self.item_index = {}           # item_id -> row index
        self.user_index = {}           # user_id -> col index
        self.topk_neighbors = None     # list of arrays de similitut d'un item

    def build_matrix(self, df):
        """
        crea matriu item x usuari a partir d'un dataframe (df)
        """
        user_codes, users = pd.factorize(df["userID"]) #crea array d'indexs i array d'usuaris
        item_codes, items = pd.factorize(df["itemID"]) #crea array d'indexs i array d'items
        n_users = len(users)
        n_items = len(items)
        mat = np.zeros((n_items, n_users), dtype=np.float32) #creem matriu de 0
        for u_code, i_code, r in zip(user_codes, item_codes, df["rating"]): #plenem del rating usuari-item
            mat[i_code, u_code] = r
        self.item_user_matrix = mat 
        self.items = items
        self.users = users
        self.item_index = {item: idx for idx, item in enumerate(items)} #crea un index per cada item
        self.user_index = {user: idx for idx, user in enumerate(users)} #crea un index per cada usuari

    def fit(self, df):
        """
        df: DataFrame with columns userID, itemID, rating
        """
        self.build_matrix(df) #creem matriu
        """
        Limitació de seguretat: Si hi ha massa usuaris o items, atura
        #Creiem que ja es limita el dataset amb fraction del data_cleaner
        if len(self.items) > 30000 or len(self.users) > 30000:
            raise ValueError(f"Massa usuaris/items per entrenar KNN. Redueix dataset amb filtres més estrictes.")
        """
        #compara totes les files entre elles, i calcula per parelles com de similars són dos ítems segons les valoracions dels usuaris
        sim = cosine_similarity(self.item_user_matrix) #sim = matriu item x item amb les similituts cosinus (0-1) entre items (diagonal=1)
        if self.use_topk: #For each item, keep top-k neighbors (excluding itself)
            n_items = sim.shape[0] #numero d'items
            self.topk_neighbors = [None] * n_items #[None, None, None, None] tants com items 
            for i in range(n_items):
                sims = sim[i].copy() #sims = matriu 1xn_items (copia la fila)
                sims[i] = -1  # exclude self
                if self.k < len(sims):
                    topk_idx = np.argpartition(sims, -self.k)[-self.k:]
                    topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]] # sort those in descending order
                else:
                    topk_idx = np.argsort(sims)[::-1] #k >= n_items -> take all except self, sorted
                topk_sims = sims[topk_idx]
                # keep only positive similarities to avoid weird signs
                positive_mask = topk_sims > 0
                topk_idx = topk_idx[positive_mask]
                topk_sims = topk_sims[positive_mask]
                self.topk_neighbors[i] = (topk_idx, topk_sims)
            #delete sim matrix to save memory, keep only topk
            del sim
        else:
            # keep full sim matrix (less memory efficient)
            self.topk_neighbors = sim


def train_itemknn(df, k=20, model_path=None, use_topk=True):
    model = ItemItemKNN(k=k, use_topk=use_topk)
    model.fit(df) #entrena el model
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True) # ensure models directory exists
        with open(model_path, "wb") as f:
            pickle.dump(model, f) #guarda el pickle generat
    return model #el codi que cridi la funció podrà seguir utilitzant el model.
  
        

"""
if __name__ == "__main__":
    from data_cleaner import load_and_clean
    PATH = r"/Users/luciarodriguez/Desktop/AC-12/ratings_Electronics(1).csv"
    df = load_and_clean(PATH)
    print("DF shape:", df.shape)
    print("Training item-item KNN (top-k)...")
    model = train_itemknn(df, k=20, model_path="../models/knn_item_model.pkl", use_topk=True)
    print("Saved model to ../models/knn_item_model.pkl")
"""

