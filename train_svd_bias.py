# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 19:15:39 2025

@author: laura
"""

"""
Els models sense bias assumeixen que totes les diferències entre ratings s’expliquen EN EXCLUSIVA pels factors latents P i Q.
Però a la pràctica, hi ha patrons molt més simples:

Global bias (μ)

    És la mitjana global de totes les valoracions.
    Exemple: en Amazon, la mitjana sol ser ~4.2.

User bias (bᵤ)

    Alguns usuaris donen sempre més nota (usuari generós) o sempre menys (usuari crític).
    Si un usuari valora tot amb −1 menys que la resta, el model ho ha d’aprendre sense gastar factors latents.

Item bias (bᵢ)

    Alguns ítems són populars i sempre tenen bones notes, i altres són mediocres.
    Això també es pot capturar amb un sol número, no cal gastar 50 dimensions.
"""

import numpy as np
import pandas as pd
import pickle
import os

class MatrixFactorizationBias:
    """
    FunkSVD amb:
    - global bias (mu)
    - user bias (b_u)
    - item bias (b_i)
    """
    def __init__(self, num_users, num_items, k=20, lr=0.01, reg=0.1, epochs=20):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        # Factors latents
        self.P = np.random.normal(scale=0.1, size=(num_users, k))
        self.Q = np.random.normal(scale=0.1, size=(num_items, k))

        # Bias terms
        self.mu = 0.0                      # Global bias
        self.b_u = np.zeros(num_users)      # User bias
        self.b_i = np.zeros(num_items)      # Item bias

    def fit(self, data):
        data = np.array(data)

        # Compute global mean from ratings
        self.mu = np.mean(data[:, 2])

        for ep in range(self.epochs):

            np.random.shuffle(data)

            for row in data:
                user = int(row[0])
                item = int(row[1])
                rating = row[2]

                # Predicció amb bias
                pred = self.mu + self.b_u[user] + self.b_i[item] + np.dot(self.P[user], self.Q[item])

                # Error
                error = rating - pred

                # Guardem referències ràpides
                p_u = self.P[user]
                q_i = self.Q[item]

                # Updates SGD
                # Bias updates
                self.b_u[user] += self.lr * (error - self.reg * self.b_u[user])
                self.b_i[item] += self.lr * (error - self.reg * self.b_i[item])

                # Factors latents
                self.P[user] += self.lr * (error * q_i - self.reg * p_u)
                self.Q[item] += self.lr * (error * p_u - self.reg * q_i)

            print(f"Epoch {ep+1}/{self.epochs} completed.")


def train_svd_bias(df, k=20, epochs=20, model_path="models/svd_bias_model.pkl"):
    """
    Entrena FunkSVD amb bias i guarda:
    model, users, items
    """

    user_codes, users = pd.factorize(df["userID"])
    item_codes, items = pd.factorize(df["itemID"])

    data = list(zip(user_codes, item_codes, df["rating"]))

    model = MatrixFactorizationBias(
        num_users=len(users),
        num_items=len(items),
        k=k,
        epochs=epochs
    )

    model.fit(data)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump((model, users, items), f)

    return model, users, items


if __name__ == "__main__":
    from data_cleaner import load_and_clean

    PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
    df = load_and_clean(PATH)
    
    print("DF shape:", df.shape)
    print("Training FunkSVD with bias...")

    train_svd_bias(df, k=20, epochs=20, model_path="../models/svd_model_bias.pkl")

    print("Saved model to ../models/svd_model_bias.pkl")
