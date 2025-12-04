# -*- coding: utf-8 -*-
"""
Created on Mon Dec 1 9:28:37 2025

@author: marmassanas
""" 
import pickle
from train_knn import ItemItemKNN
import numpy as np

# ---------------------------------------------------------
# 1. Carregar el model KNN entrenat
# ---------------------------------------------------------
MODEL_PATH = "models/knn_item_model.pkl"

with open(MODEL_PATH, "rb") as f:
    knn_model = pickle.load(f)  # directament l'objecte ItemItemKNN

print("Model KNN carregat correctament!\n")

# ---------------------------------------------------------
# 2. Funció per predir el rating d’un usuari per un producte
# ---------------------------------------------------------
def predict_rating(user_id, item_id, K=10):
    if user_id not in knn_model.user_index or item_id not in knn_model.item_index:
        return None
    return knn_model.predict(user_id, item_id)

# ---------------------------------------------------------
# 3. Funció per recomanar TOP-N productes a un usuari
# ---------------------------------------------------------
def recommend_items(user_id, top_n=10):
    return knn_model.recommend(user_id, top_n=top_n)

# ---------------------------------------------------------
# 4. TEST: Exemple d’ús
# ---------------------------------------------------------
TEST_USER = knn_model.users[1]  # primer usuari
TEST_ITEM = knn_model.items[4]  # primer producte

# Imprimir noms reals
print(f"Usuari de prova (ID real): {TEST_USER}")
print(f"Producte de prova (ID real): {TEST_ITEM}")

# Predicció rating
pred = predict_rating(TEST_USER, TEST_ITEM)

print("\n--- Predicció ---")
if pred is None:
    print("No es pot predir per falta de dades.")
else:
    print(f"Predicció del rating → {pred:.3f}")

# Recomanacions Top-10
print("\n--- Recomanacions TOP-10 ---")
recs = recommend_items(TEST_USER, top_n=10)

for i, (item, score) in enumerate(recs, start=1):
    print(f"{i}. Producte: {item}  →  Predicció rating: {score:.3f}")
