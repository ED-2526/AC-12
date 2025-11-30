# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:54:03 2025

@author: laura
"""

import numpy as np
import pandas as pd
import pickle
import os

class MatrixFactorization:
    def __init__(self, num_users, num_items, k=20, lr=0.01, reg=0.1, epochs=20):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        self.P = np.random.normal(scale=0.1, size=(num_users, k))
        self.Q = np.random.normal(scale=0.1, size=(num_items, k))

    def fit(self, data):
        data = np.array(data)
    
        for ep in range(self.epochs):
            # barregem les files cada epoch
            np.random.shuffle(data)
    
            for row in data:
                user = int(row[0])
                item = int(row[1])
                rating = row[2]
    
                p_u = self.P[user]
                q_i = self.Q[item]
    
                pred = np.dot(p_u, q_i)
                error = rating - pred
    
                # SGD update
                self.P[user] += self.lr * (error * q_i - self.reg * p_u)
                self.Q[item] += self.lr * (error * p_u - self.reg * q_i)
    
            print(f"Epoch {ep+1}/{self.epochs} completed.")



def train_svd_model(df, k=20, epochs=20, model_path="models/svd_model.pkl"):
    """
    Entrena un model FunkSVD i el guarda a disk.
    """
    # Encode users and items
    user_codes, users = pd.factorize(df["userID"])
    item_codes, items = pd.factorize(df["itemID"])
    data = list(zip(user_codes, item_codes, df["rating"]))

    # Entrenar el model
    model = MatrixFactorization(
        num_users=len(users),
        num_items=len(items),
        k=k,
        epochs=epochs
    )
    model.fit(data)

    # Crear directori si no existeix
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # Guardar model
    with open(model_path, "wb") as f:
        pickle.dump((model, users, items), f)

    return model, users, items


if __name__ == "__main__":
    from data_cleaner import load_and_clean

    PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
    df = load_and_clean(PATH)
    print("DF shape:", df.shape)
    print("Training FunkSVD...")
    model = train_svd_model(df, k=20, epochs=20, model_path="../models/svd_model.pkl")
    print("Saved model to ../models/svd_model.pkl")
