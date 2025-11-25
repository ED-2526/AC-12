import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from scipy.sparse import load_npz

#Llegir CSV i assignar noms (el dataset no té capçaleres)
columnes = ['user_id', 'product_id', 'rating', 'timestamp']
df = pd.read_csv('/Users/luciarodriguez/Desktop/AC-12/ratings_Electronics(1).csv', header=None, names=columnes, usecols=[0,1,2,3])

#Eliminem les files on hi hagi mínim un valor nul (no serveix la dada)
df = df.dropna(subset=['user_id', 'product_id', 'rating'])

#Si un usuari ha valorat un mateix producte diverses vegades, fem la mitjana
df = df.groupby(['user_id', 'product_id'], as_index=False)['rating'].mean()

# 4️⃣ Assignar un índex numèric per a usuaris i productes
user_idx = {user: i for i, user in enumerate(df['user_id'].unique())}
product_idx = {prod: i for i, prod in enumerate(df['product_id'].unique())}

# 5️⃣ Convertir IDs a índexos numèrics
df['user_index'] = df['user_id'].map(user_idx)
df['product_index'] = df['product_id'].map(product_idx)

# 6️⃣ Crear matriu esparsa Usuari × Producte
R_sparse = csr_matrix((df['rating'], (df['user_index'], df['product_index'])))

print("Nombre d'usuaris:", len(user_idx))
print("Nombre de productes:", len(product_idx))
print("Matriu de", R_sparse.shape, "creada amb èxit: ")

# 7️⃣ Guardar matriu esparsa en fitxer binari .npz
save_npz('R_sparse_avg.npz', R_sparse)
print("Matriu esparsa guardada amb èxit a 'R_sparse_avg.npz'")
#data_cleaned = dataset.copy()

# 🔹 Opcional: guardar CSV amb només ratings existents (ja amb mitjana aplicada)
df[['user_id', 'product_id', 'rating']].to_csv('ratings_sparse_avg.csv', index=False)
print("CSV amb ratings existents i mitjana guardat a 'ratings_sparse_avg.csv'")



R_sparse = load_npz('R_sparse_avg.npz')

# Veure només els primers 10 usuaris i 10 productes
print(R_sparse[:15, :15].toarray())
