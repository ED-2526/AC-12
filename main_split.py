# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 20:01:14 2025

@author: laura
"""


"""
Main complet amb split train/test, ús d'infer_knn i infer_svd, i proves de predicció / recomanació.
"""

import os
import pandas as pd
from data_cleaner import load_and_clean, visualize_dataset
from train_knn import train_itemknn
from train_svd import train_svd_model
import infer_knn
import infer_svd


# Split Train/Test
def split_train_test(df, test_ratio=0.2):
    df = df.sample(frac=1, random_state=42)  # barreja
    cutoff = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:cutoff]
    test_df = df.iloc[cutoff:]
    return train_df, test_df

# MAIN
if __name__ == "__main__":

    # ----- 1. Paths -----
    ROOT = os.path.dirname(__file__)
    DATA_PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
    MODELS_DIR = os.path.join(ROOT, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_item_model.pkl")
    SVD_MODEL_PATH = os.path.join(MODELS_DIR, "svd_model.pkl")

    # ----- 2. Load + clean -----
    print("Loading and cleaning dataset...")
    CLEAN_PATH = "cleaned_data.csv"
    if os.path.exists(CLEAN_PATH):
        print("cleaned_data.csv trobat. Carregant dataset netejat...")
        df = pd.read_csv(CLEAN_PATH)
    else:
        print("No existeix cleaned_data.csv. Netejant dataset original...")
        df = load_and_clean(DATA_PATH)  # ja es guarda automàticament
    print(f"Dataset shape after cleaning: {df.shape}")

    #  ----- 3. Visualize dataset -----
    visualize_dataset(df)

    # ----- 4. Split train/test -----
    print("\nFent Split Train/Test...")
    train_df, test_df = split_train_test(df)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # ----- 5. Train KNN -----
    print("\nTraining Item-Item KNN...")
    _ = train_itemknn(train_df, k=20, model_path=KNN_MODEL_PATH, use_topk=True)
    print(f"KNN model saved at: {KNN_MODEL_PATH}")

    # ----- 6. Train FunkSVD -----
    print("\nTraining FunkSVD...")
    _ = train_svd_model(train_df, k=20, epochs=20, model_path=SVD_MODEL_PATH)
    print(f"SVD model saved at: {SVD_MODEL_PATH}")

    # ----- 7. Load models (infer_knn / infer_svd) -----
    print("\nCarregant models d'inferència...")
    knn_model = infer_knn.load_model(KNN_MODEL_PATH)
    svd_model, svd_users, svd_items = infer_svd.load_model(SVD_MODEL_PATH)

    # ----- 8. Exemple d'usuari -----
    sample_user = svd_users[0]
    print("\nUsuari seleccionat:", sample_user)
    example_item = svd_items[0]

    # ----- 9. Predicció SVD -----
    pred_svd = infer_svd.predict_svd(sample_user, example_item, svd_model, svd_users, svd_items)
    print(f"\n[SVD] Predicció {sample_user} sobre {example_item}: {pred_svd}")

    # ----- 10. Recomanacions SVD -----
    print("\nTop 10 recomanacions (SVD):")
    recs_svd = infer_svd.recommend_svd(sample_user, svd_model, svd_users, svd_items, top_n=10)
    for item, score in recs_svd:
        print(f"{item}: {score}")

    # ----- 11. Predicció KNN -----
    sample_user_knn = list(knn_model.user_index.keys())[0]
    example_item_knn = list(knn_model.item_index.keys())[0]
    print("\n[KNN] User seleccionat:", sample_user_knn)
    pred_knn = infer_knn.predict_knn(sample_user_knn, example_item_knn, knn_model)
    print(f"[KNN] Predicció {sample_user_knn} sobre {example_item_knn}: {pred_knn}")

    # ----- 12. Recomanacions KNN -----
    print("\n[KNN] Top 10 recomanacions:")
    recs_knn = infer_knn.recommend_knn(sample_user_knn, knn_model, top_n=10)
    for item, score in recs_knn:

        print(f"{item}: {score}")
