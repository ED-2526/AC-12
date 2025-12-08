
import pandas as pd
from lenskit import crossfold as xf
from lenskit.algorithms import basic, user_knn, als
from lenskit import batch, topn


#carrega el dataset neet
df = pd.read_csv("cleaned_data.csv")
#df = df[["userID", "itemID", "rating"]]  
df = df.rename(columns={ "userID": "user", "itemID": "item"}) #pq lenskit vol aquests noms
print("Dataset carregat:", df.shape)


#train/Test simple (LensKit recomana crossfold)
train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)
print("Train:", train.shape, "Test:", test.shape)


#User-User KNN
print("\nEntrenando User-User KNN con LensKit...")
knn = user_knn.UserUser(20, center=True)   # k=20
knn.fit(train)


# SVD

print("\nEntrenant MF (BiasedMF)...")
mf = als.BiasedMF(20, iterations=20)       # 20 factors, 20 iteracions
mf.fit(train)

# trie usuari real
sample_user = train["user"].iloc[0]
print("\nUsuari seleccionat:", sample_user)

# Llista d'items possibles
all_items = df["item"].unique()


item_example = all_items[0]
print("\nPrediccin MF sobre l'item triat:")
pred = batch.predict(mf, sample_user, [item_example])
print(pred)

print("\nTop-10 recomanacions amb MF:")

mf_recs = batch.recommend(mf, [sample_user], all_items, n=10)
print(mf_recs)

print("\nTop-10 recomanacions amb User-User KNN:")

knn_recs = batch.recommend(knn, [sample_user], all_items, n=10)
print(knn_recs)
