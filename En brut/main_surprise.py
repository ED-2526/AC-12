# -*- coding: utf-8 -*-
"""
MAIN SURPRISE SCRIPT — KNN Item-Item i SVD (Funk-SVD)
Comparació directa amb els models propis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNBasic, SVD
from data_cleaner import load_and_clean  


# -----------------------------------------
#       Funcions de mètriques 
# -----------------------------------------

def precision_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def apk(recommended, relevant, k):
    score, hits = 0, 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i+1)
    return score / min(len(relevant), k)

def ndcg_k(recommended, relevant, k):
    dcg = sum(1/np.log2(i+2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# -----------------------------------------
#                 DATA
# -----------------------------------------

PATH = r"ratings_Electronics.csv"
CLEAN_PATH = "cleaned_data.csv"
if os.path.exists(CLEAN_PATH):
    print("cleaned_data.csv trobat. Carregant dataset netejat...")
    df = pd.read_csv(CLEAN_PATH)
else:
    print("No existeix cleaned_data.csv. Netejant dataset original...")
    df = load_and_clean(PATH) 
print("Dataset:", df.shape)

test = df.groupby("userID").tail(1)
train = df.drop(test.index)

test_users  = list(test["userID"])
test_items  = list(test["itemID"])
test_ratings = test.set_index("userID")["rating"].to_dict()

train_items_per_user = train.groupby("userID")["itemID"].apply(set).to_dict()
all_items = set(train["itemID"])

print("Train:", train.shape)
print("Test:", test.shape)


# -----------------------------------------
#     PREPARAR SURPRISE DATASET
# -----------------------------------------

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train[['userID', 'itemID', 'rating']], reader)
trainset = data.build_full_trainset()


# -----------------------------------------
#               Entrenar MODELS
# -----------------------------------------

print("\n=== ENTRENANT Surprise KNN item-item ===")
knn_algo = KNNBasic(
    k=20,
    sim_options={"name": "cosine", "user_based": False}
)
knn_algo.fit(trainset)


print("\n=== ENTRENANT Surprise SVD ===")
svd_algo = SVD(
    n_factors=20,
    n_epochs=15,
    biased=True,
)
svd_algo.fit(trainset)


# -----------------------------------------
#               TEST + METRIQUES
# -----------------------------------------

K = 10

rmse_knn = rmse_svd = 0
prec_knn = []; rec_knn = []; map_knn = []; ndcg_knn = []
prec_svd = []; rec_svd = []; map_svd = []; ndcg_svd = []

print("\n=== INICIANT TEST... ===")

for u, true_item in zip(test_users, test_items):

    true_rating = test_ratings[u]
    relevant = [true_item]

    rated_items = train_items_per_user.get(u, set())
    unseen_items = list(all_items - rated_items)

    # --------------------------
    # KNN
    # --------------------------

    pred_knn = knn_algo.predict(u, true_item).est
    rmse_knn += (pred_knn - true_rating)**2

    preds_knn = [(iid, knn_algo.predict(u, iid).est) for iid in unseen_items]
    preds_knn.sort(key=lambda x: x[1], reverse=True)
    rec_knn_u = [iid for iid, _ in preds_knn[:K]]

    prec_knn.append(precision_k(rec_knn_u, relevant, K))
    rec_knn.append(recall_k(rec_knn_u, relevant, K))
    map_knn.append(apk(rec_knn_u, relevant, K))
    ndcg_knn.append(ndcg_k(rec_knn_u, relevant, K))

    # --------------------------
    # SVD
    # --------------------------

    pred_svd = svd_algo.predict(u, true_item).est
    rmse_svd += (pred_svd - true_rating)**2

    preds_svd = [(iid, svd_algo.predict(u, iid).est) for iid in unseen_items]
    preds_svd.sort(key=lambda x: x[1], reverse=True)
    rec_svd_u = [iid for iid, _ in preds_svd[:K]]

    prec_svd.append(precision_k(rec_svd_u, relevant, K))
    rec_svd.append(recall_k(rec_svd_u, relevant, K))
    map_svd.append(apk(rec_svd_u, relevant, K))
    ndcg_svd.append(ndcg_k(rec_svd_u, relevant, K))


# Final metrics
N = len(test_users)
if N > 0:
    rmse_knn = np.sqrt(rmse_knn / N) 
    rmse_svd = np.sqrt(rmse_svd / N) 
else:
    rmse_knn = None
    rmse_svd = None

# Calcular mitjanes
mean_prec_knn = np.mean(prec_knn)
mean_rec_knn = np.mean(rec_knn)
mean_map_knn = np.mean(map_knn)
mean_ndcg_knn = np.mean(ndcg_knn)

mean_prec_svd = np.mean(prec_svd)
mean_rec_svd = np.mean(rec_svd)
mean_map_svd = np.mean(map_svd)
mean_ndcg_svd = np.mean(ndcg_svd)

print("\n============== RESULTATS SURPRISE ==============")
print(f"\n--- KNN ---")
print(f"Precision@{K}: {mean_prec_knn:.4f}")
print(f"Recall@{K}:    {mean_rec_knn:.4f}")
print(f"MAP@{K}:       {mean_map_knn:.4f}")
print(f"NDCG@{K}:      {mean_ndcg_knn:.4f}")
print(f"RMSE:          {rmse_knn:.4f}" if rmse_knn is not None else "RMSE:         N/A")

print(f"\n--- SVD ---")
print(f"Precision@{K}: {mean_prec_svd:.4f}")
print(f"Recall@{K}:    {mean_rec_svd:.4f}")
print(f"MAP@{K}:       {mean_map_svd:.4f}")
print(f"NDCG@{K}:      {mean_ndcg_svd:.4f}")
print(f"RMSE:          {rmse_svd:.4f}" if rmse_svd is not None else "RMSE:         N/A")
print("\n================================================")

# ==========================================================
#               GRÀFIC ÚNIC DE RESULTATS
# ==========================================================

# Configurar l'estil
plt.style.use('seaborn-v0_8-darkgrid')
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Comparativa de Models Surprise - Resultats Complets', fontsize=16, fontweight='bold')

# Colors per als models
colors = ['#1f77b4', '#ff7f0e']  # KNN: blau, SVD: taronja

# Dades per als models
model_names = ['KNN', 'SVD']
metrics_data = {
    'KNN': [mean_prec_knn, mean_rec_knn, mean_map_knn, mean_ndcg_knn, rmse_knn],
    'SVD': [mean_prec_svd, mean_rec_svd, mean_map_svd, mean_ndcg_svd, rmse_svd]
}

# 1. Precision@10
ax1.bar(model_names, [mean_prec_knn, mean_prec_svd], color=colors, alpha=0.8)
ax1.set_title('Precision@10')
ax1.set_ylabel('Valor')
ax1.grid(True, alpha=0.3)
# Afegir valors sobre les barres
for i, (model, val) in enumerate(zip(model_names, [mean_prec_knn, mean_prec_svd])):
    ax1.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

# 2. Recall@10
ax2.bar(model_names, [mean_rec_knn, mean_rec_svd], color=colors, alpha=0.8)
ax2.set_title('Recall@10')
ax2.set_ylabel('Valor')
ax2.grid(True, alpha=0.3)
for i, (model, val) in enumerate(zip(model_names, [mean_rec_knn, mean_rec_svd])):
    ax2.text(i, val + 0.01, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

# 3. MAP@10
ax3.bar(model_names, [mean_map_knn, mean_map_svd], color=colors, alpha=0.8)
ax3.set_title('MAP@10')
ax3.set_ylabel('Valor')
ax3.grid(True, alpha=0.3)
for i, (model, val) in enumerate(zip(model_names, [mean_map_knn, mean_map_svd])):
    ax3.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

# 4. NDCG@10
ax4.bar(model_names, [mean_ndcg_knn, mean_ndcg_svd], color=colors, alpha=0.8)
ax4.set_title('NDCG@10')
ax4.set_ylabel('Valor')
ax4.grid(True, alpha=0.3)
for i, (model, val) in enumerate(zip(model_names, [mean_ndcg_knn, mean_ndcg_svd])):
    ax4.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

# 5. RMSE
ax5.bar(model_names, [rmse_knn, rmse_svd], color=colors, alpha=0.8)
ax5.set_title('RMSE')
ax5.set_ylabel('Valor')
ax5.grid(True, alpha=0.3)
for i, (model, val) in enumerate(zip(model_names, [rmse_knn, rmse_svd])):
    ax5.text(i, val + 0.05, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

# 6. Comparativa general (gràfic de línies)
metrics_names = ['Precision@10', 'Recall@10', 'MAP@10', 'NDCG@10', 'RMSE']
x_pos = np.arange(len(metrics_names))

# Normalitzar RMSE perquè es vegi bé al mateix gràfic (invertim perquè RMSE més baix és millor)
norm_rmse_knn = 1.0 / (rmse_knn + 0.1)  # Afegim 0.1 per evitar divisió per zero
norm_rmse_svd = 1.0 / (rmse_svd + 0.1)

knn_values = [mean_prec_knn, mean_rec_knn, mean_map_knn, mean_ndcg_knn, norm_rmse_knn]
svd_values = [mean_prec_svd, mean_rec_svd, mean_map_svd, mean_ndcg_svd, norm_rmse_svd]

ax6.plot(x_pos, knn_values, 'o-', linewidth=2, markersize=8, label='KNN', color=colors[0])
ax6.plot(x_pos, svd_values, 's-', linewidth=2, markersize=8, label='SVD', color=colors[1])
ax6.set_title('Comparativa General (RMSE normalitzat)')
ax6.set_xlabel('Mètriques')
ax6.set_ylabel('Valor (RMSE invertit)')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(metrics_names, rotation=45, ha='right')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Ajustar layout
plt.tight_layout()

# Guardar la imatge
plt.savefig('surprise_results_complete.png', dpi=300, bbox_inches='tight')
plt.savefig('surprise_results_complete.pdf', bbox_inches='tight')

print("\nGràfic guardat com 'surprise_results_complete.png' i '.pdf'")

# Mostrar el gràfic
plt.show()

# ==========================================================
#               GRÀFIC ADDICIONAL: COMPARATIVA EN BARRES
# ==========================================================

# Crear un gràfic extra amb totes les mètriques en barres agrupades
fig2, ax = plt.subplots(figsize=(12, 7))

metrics_for_plot = ['Precision@10', 'Recall@10', 'MAP@10', 'NDCG@10', 'RMSE']
x = np.arange(len(metrics_for_plot))
width = 0.35

# Valors reals (per a RMSE, posem el valor directe)
knn_vals_plot = [mean_prec_knn, mean_rec_knn, mean_map_knn, mean_ndcg_knn, rmse_knn]
svd_vals_plot = [mean_prec_svd, mean_rec_svd, mean_map_svd, mean_ndcg_svd, rmse_svd]

bars_knn = ax.bar(x - width/2, knn_vals_plot, width, label='KNN', color=colors[0], alpha=0.8)
bars_svd = ax.bar(x + width/2, svd_vals_plot, width, label='SVD', color=colors[1], alpha=0.8)

# Afegir valors sobre les barres
for bars, vals in zip([bars_knn, bars_svd], [knn_vals_plot, svd_vals_plot]):
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Mètriques')
ax.set_ylabel('Valor')
ax.set_title('Comparativa Completa: Surprise KNN vs SVD')
ax.set_xticks(x)
ax.set_xticklabels(metrics_for_plot)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('surprise_comparison_bars.png', dpi=300, bbox_inches='tight')
print("Gràfic de comparativa en barres guardat com 'surprise_comparison_bars.png'")
plt.show()

# ==========================================================
#               TAULA RESUM DE RESULTATS
# ==========================================================

print("\n" + "="*60)
print("TAULA RESUM DE RESULTATS SURPRISE")
print("="*60)

# Crear taula
headers = ["Model", "Precision@10", "Recall@10", "MAP@10", "NDCG@10", "RMSE"]
table_data = [
    ["KNN", f"{mean_prec_knn:.4f}", f"{mean_rec_knn:.4f}", 
     f"{mean_map_knn:.4f}", f"{mean_ndcg_knn:.4f}", 
     f"{rmse_knn:.4f}" if rmse_knn is not None else "N/A"],
    ["SVD", f"{mean_prec_svd:.4f}", f"{mean_rec_svd:.4f}", 
     f"{mean_map_svd:.4f}", f"{mean_ndcg_svd:.4f}", 
     f"{rmse_svd:.4f}" if rmse_svd is not None else "N/A"]
]

# Imprimir taula
col_widths = [12, 15, 15, 15, 15, 15]
header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
print(header_line)
print("-"*len(header_line))

for row in table_data:
    print("".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)))

print("="*60)
print("\nResum:")
print(f"KNN guanya en: {'Precision' if mean_prec_knn > mean_prec_svd else 'SVD guanya'}")
print(f"SVD guanya en: {'Recall' if mean_rec_svd > mean_rec_knn else 'KNN guanya'}")
print(f"RMSE més baix: {'KNN' if rmse_knn < rmse_svd else 'SVD'}")
print("="*60)


