# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:54:03 2025

@author: laura
"""

import numpy as np
import pandas as pd
import pickle

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

    def train(self, data):
        for ep in range(self.epochs):
            for user, item, rating in data:
                pred = np.dot(self.P[user], self.Q[item])
                error = rating - pred

                # Gradient steps
                self.P[user] += self.lr * (error * self.Q[item] - self.reg * self.P[user])
                self.Q[item] += self.lr * (error * self.P[user] - self.reg * self.Q[item])

            print(f"Epoch {ep+1}/{self.epochs} completed.")

    def predict(self, u, i):
        return np.dot(self.P[u], self.Q[i])


def train_svd_model(df, k=20, epochs=20):
    # Encode users and items
    user_codes, users = pd.factorize(df["userID"])
    item_codes, items = pd.factorize(df["itemID"])

    data = list(zip(user_codes, item_codes, df["rating"]))

    model = MatrixFactorization(
        num_users=len(users),
        num_items=len(items),
        k=k,
        epochs=epochs
    )

    model.train(data)

    return model, users, items


if __name__ == "__main__":
    from data_cleaner import load_and_clean

    PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
    df = load_and_clean(PATH)

    model, users, items = train_svd_model(df)

    with open("svd_model.pkl", "wb") as f:
        pickle.dump((model, users, items), f)

    print("SVD model trained and saved.")

