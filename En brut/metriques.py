import numpy as np
import math
import os
import pandas as pd
import pickle
from collections import defaultdict
from inferencia import KNNPredictions, SVDPredictions

# ---------- Rating metrics ----------
def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return math.sqrt(((y_true - y_pred) ** 2).mean())

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.abs(y_true - y_pred).mean()

# ---------- Utility: average precision for one user ----------
def apk(actual, predicted, k):
    """
    Average precision at k for a single user.
    actual: set of relevant items
    predicted: list of recommended items (ordered)
    """
    if k == 0:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:  # count only first time
            hits += 1.0
            score += hits / (i + 1.0)
    if len(actual) == 0:
        return 0.0
    return score / min(len(actual), k)

def mapk(actuals, predicteds, k):
    """
    Mean Average Precision at k for lists/dicts
    actuals: dict user -> set(items)
    predicteds: dict user -> list(items)
    """
    ap_sum = 0.0
    n = 0
    for u, actual in actuals.items():
        if u not in predicteds:
            continue
        ap_sum += apk(actual, predicteds[u], k)
        n += 1
    return ap_sum / max(1, n)

# ---------- Precision@k and Recall@k ----------
def precision_at_k(actual, predicted, k):
    predicted_k = predicted[:k]
    if len(predicted_k) == 0:
        return 0.0
    hits = sum(1 for p in predicted_k if p in actual)
    return hits / len(predicted_k)

def recall_at_k(actual, predicted, k):
    if len(actual) == 0:
        return 0.0
    predicted_k = predicted[:k]
    hits = sum(1 for p in predicted_k if p in actual)
    return hits / len(actual)

def mean_precision_recall_at_k(actuals, predicteds, k):
    ps = []
    rs = []
    for u, act in actuals.items():
        if u not in predicteds:
            continue
        pred = predicteds[u]
        ps.append(precision_at_k(act, pred, k))
        rs.append(recall_at_k(act, pred, k))
    return np.mean(ps) if ps else 0.0, np.mean(rs) if rs else 0.0

# ---------- NDCG@k ----------
def dcg_at_k(relevances, k):
    """relevances: list of relevance (1 or 0 or graded) in rank order"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)  # i+2 because positions start at 1
    return dcg

def ndcg_at_k(actual, predicted, k):
    # relevance: 1 if item in actual else 0
    rels = [1 if p in actual else 0 for p in predicted[:k]]
    dcg = dcg_at_k(rels, k)
    # ideal DCG
    ideal_rels = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0

def mean_ndcg_at_k(actuals, predicteds, k):
    scores = []
    for u, act in actuals.items():
        if u not in predicteds:
            continue
        scores.append(ndcg_at_k(act, predicteds[u], k))
    return np.mean(scores) if scores else 0.0

# ---------- MRR ----------
def reciprocal_rank(actual, predicted):
    for i, p in enumerate(predicted):
        if p in actual:
            return 1.0 / (i + 1.0)
    return 0.0

def mean_reciprocal_rank(actuals, predicteds):
    rs = []
    for u, act in actuals.items():
        if u not in predicteds:
            continue
        rs.append(reciprocal_rank(act, predicteds[u]))
    return np.mean(rs) if rs else 0.0

# ---------- Coverage, novelty, popularity ----------
def catalog_coverage(predicteds, all_items):
    recommended = set()
    for recs in predicteds.values():
        recommended.update(recs)
    return len(recommended) / len(all_items)

def topk_item_popularity(train_interactions):
    """
    train_interactions: list of (user,item) in training set, or dict item->count
    returns dict item->count
    """
    counts = defaultdict(int)
    for (u, i) in train_interactions:
        counts[i] += 1
    return counts

def novelty_mean_popularity(predicteds, item_popularity, k):
    # lower popularity => more novel (we return average popularity; can invert)
    vals = []
    for recs in predicteds.values():
        topk = recs[:k]
        vals.extend([item_popularity.get(i, 0) for i in topk])
    return np.mean(vals) if vals else 0.0

# ---------- Diversity ----------
def diversity_mean_pairwise(predicteds, item_feature_vectors, k):
    """
    item_feature_vectors: dict item -> vector (numpy array)
    diversity = 1 - mean(similarity) across pairs within each user's top-k
    similarity uses cosine similarity
    """
    def cosine(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    diversities = []
    for recs in predicteds.values():
        topk = recs[:k]
        sims = []
        for i in range(len(topk)):
            for j in range(i+1, len(topk)):
                vi = item_feature_vectors.get(topk[i])
                vj = item_feature_vectors.get(topk[j])
                if vi is None or vj is None:
                    continue
                sims.append(cosine(vi, vj))
        if sims:
            diversities.append(1.0 - np.mean(sims))
    return np.mean(diversities) if diversities else 0.0

# ---------- Personalization (simple) ----------
def personalization(predicteds, k):
    """
    personalization = 1 - average pairwise Jaccard similarity between users' top-k lists
    """
    users = list(predicteds.keys())
    if len(users) < 2:
        return 0.0
    sims = []
    for i in range(len(users)):
        set_i = set(predicteds[users[i]][:k])
        for j in range(i+1, len(users)):
            set_j = set(predicteds[users[j]][:k])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            sims.append(inter / union if union > 0 else 0.0)
    mean_sim = np.mean(sims) if sims else 0.0
    return 1.0 - mean_sim

# ---------- Example aggregator ----------
def evaluate_all(actuals, predicteds, train_interactions=None, all_items=None, item_pop=None,
                 item_features=None, ks=(5,10,20)):
    results = {}
    # ranking metrics
    for k in ks:
        p, r = mean_precision_recall_at_k(actuals, predicteds, k)
        results[f'Precision@{k}'] = p
        results[f'Recall@{k}'] = r
        results[f'MAP@{k}'] = mapk(actuals, predicteds, k)
        results[f'NDCG@{k}'] = mean_ndcg_at_k(actuals, predicteds, k)
    results['MRR'] = mean_reciprocal_rank(actuals, predicteds)
    if all_items and predicteds:
        results['CatalogCoverage'] = catalog_coverage(predicteds, all_items)
    if train_interactions and item_pop:
        for k in ks:
            results[f'Novelty_pop@{k}'] = novelty_mean_popularity(predicteds, item_pop, k)
    if item_features:
        for k in ks:
            results[f'Diversity@{k}'] = diversity_mean_pairwise(predicteds, item_features, k)
    results['Personalization@10'] = personalization(predicteds, 10)
    return results

# ---------- MAIN EVALUATION ----------
if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    
    print("="*60)
    print("EVALUACIÓ DE MODELS SVD i KNN")
    print("="*60)
    
    # ----- 1. Configuració de paths -----
    ROOT = os.path.dirname(__file__)  # Directori actual
    DATA_PATH = os.path.join(ROOT, "cleaned_data.csv")
    MODELS_DIR = os.path.join(ROOT, "models")
    KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_item_model.pkl")
    SVD_MODEL_PATH = os.path.join(MODELS_DIR, "svd_model.pkl")
    
    # ----- 2. Carregar dataset -----
    print("\n[1] Carregant dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: No es troba {DATA_PATH}")
        print("Assegura't que tens el cleaned_data.csv al mateix directori")
        exit(1)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset carregat: {df.shape[0]} files, {df.shape[1]} columnes")
    
    # Mostrar primeres files per verificar
    print("\nPrimeres files del dataset:")
    print(df.head())
    
    # Verificar columnes necessàries
    required_cols = ['userID', 'itemID', 'rating']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Falten columnes al dataset: {missing_cols}")
        exit(1)
    
    # ----- 3. Carregar models amb classes d'inferència -----
    print("\n[2] Carregant models amb classes d'inferència...")
    
    # Crear instàncies de les classes
    knn_inf = KNNPredictions()
    svd_inf = SVDPredictions()
    
    # Carregar model KNN
    if not os.path.exists(KNN_MODEL_PATH):
        print(f"ERROR: No es troba {KNN_MODEL_PATH}")
        print("Assegura't que el model KNN està al directori /models")
        exit(1)
    
    knn_model_loaded = knn_inf.load_model(KNN_MODEL_PATH)
    print(f"✓ KNN model carregat")
    print(f"  Usuaris: {len(knn_inf.model.user_index)}")
    print(f"  Items: {len(knn_inf.model.item_index)}")
    
    # Carregar model SVD
    if not os.path.exists(SVD_MODEL_PATH):
        print(f"ERROR: No es troba {SVD_MODEL_PATH}")
        print("Assegura't que el model SVD està al directori /models")
        exit(1)
    
    svd_model_loaded = svd_inf.load_model(SVD_MODEL_PATH)
    print(f"✓ SVD model carregat")
    print(f"  Usuaris: {len(svd_inf.users)}")
    print(f"  Items: {len(svd_inf.items)}")
    
    # ----- 4. Preparar dades per avaluació -----
    print("\n[3] Preparant dades per avaluació...")
    
    # Agrupar items per usuari (items que ja ha valorat)
    user_items = df.groupby('userID')['itemID'].apply(set).to_dict()
    user_ratings = df.groupby('userID').apply(lambda x: dict(zip(x['itemID'], x['rating']))).to_dict()
    
    # Tots els items únics del dataset
    all_items = set(df['itemID'])
    print(f"Total usuaris al dataset: {len(user_items)}")
    print(f"Total items al dataset: {len(all_items)}")
    
    # ----- 5. Calcular mètriques de RATING (RMSE, MAE) -----
    print("\n[4] Calculant mètriques de RATING...")
    
    # Recollir ratings reals i predits per cada model
    ratings_true_knn = []
    ratings_pred_knn = []
    ratings_true_svd = []
    ratings_pred_svd = []
    
    # Filtrar usuaris/items que els models coneixen
    knn_users = set(knn_inf.model.user_index.keys())
    knn_items = set(knn_inf.model.item_index.keys())
    svd_users = set(svd_inf.users)
    svd_items = set(svd_inf.items)
    
    print(f"Usuaris coneguts per KNN: {len(knn_users)}")
    print(f"Items coneguts per KNN: {len(knn_items)}")
    print(f"Usuaris coneguts per SVD: {len(svd_users)}")
    print(f"Items coneguts per SVD: {len(svd_items)}")
    
    # Mostrem progress (agafem una mostra per no trigar massa)
    sample_size = min(1000, len(df))
    print(f"Evaluant sobre mostra de {sample_size} interaccions...")
    
    sample_df = df.sample(sample_size, random_state=42)
    
    print("\nCalculant prediccions...")
    for idx, row in sample_df.iterrows():
        user = row['userID']
        item = row['itemID']
        true_rating = row['rating']
        
        # Predicció KNN (només si el model coneix usuari i item)
        if user in knn_users and item in knn_items:
            pred_knn = knn_inf.predict(user, item)
            ratings_true_knn.append(true_rating)
            ratings_pred_knn.append(pred_knn)
        
        # Predicció SVD (només si el model coneix usuari i item)
        if user in svd_users and item in svd_items:
            pred_svd = svd_inf.predict(user, item)
            ratings_true_svd.append(true_rating)
            ratings_pred_svd.append(pred_svd)
    
    # Calcular mètriques de rating
    print("\n--- Mètriques de RATING ---")
    if ratings_true_knn and ratings_pred_knn:
        rmse_knn = rmse(ratings_true_knn, ratings_pred_knn)
        mae_knn_val = mae(ratings_true_knn, ratings_pred_knn)
        print(f"KNN:")
        print(f"  • RMSE: {rmse_knn:.4f}")
        print(f"  • MAE:  {mae_knn_val:.4f}")
        print(f"  • Mostra: {len(ratings_true_knn)} interaccions")
    else:
        rmse_knn = mae_knn_val = np.nan
        print(f"KNN: No hi ha dades vàlides per calcular mètriques")
    
    if ratings_true_svd and ratings_pred_svd:
        rmse_svd = rmse(ratings_true_svd, ratings_pred_svd)
        mae_svd_val = mae(ratings_true_svd, ratings_pred_svd)
        print(f"\nSVD:")
        print(f"  • RMSE: {rmse_svd:.4f}")
        print(f"  • MAE:  {mae_svd_val:.4f}")
        print(f"  • Mostra: {len(ratings_true_svd)} interaccions")
    else:
        rmse_svd = mae_svd_val = np.nan
        print(f"\nSVD: No hi ha dades vàlides per calcular mètriques")
    
    # ----- 6. Calcular mètriques de RECOMANACIÓ -----
    print("\n[5] Calculant mètriques de RECOMANACIÓ...")
    
    # Per a recomanacions, necessitem definir quins items són "rellevants" per cada usuari
    # Utilitzem threshold de 4 (com has dit)
    threshold = 4.0
    
    # Crear diccionari d'items rellevants per usuari
    relevant_items = {}
    for user, items_dict in user_ratings.items():
        rel_items = {item for item, rating in items_dict.items() if rating >= threshold}
        if rel_items:  # Només usuaris amb items rellevants
            relevant_items[user] = rel_items
    
    print(f"Usuaris amb items rellevants (rating ≥ {threshold}): {len(relevant_items)}")
    
    # Agafem una mostra d'usuaris per avaluar (que els models coneixin)
    # Prioritzem usuaris que estiguin en tots dos models
    common_users = [u for u in relevant_items.keys() 
                   if u in knn_users and u in svd_users]
    
    if not common_users:
        print("ERROR: No hi ha usuaris comuns entre models i items rellevants")
        exit(1)
    
    sample_users = common_users[:50]  # Primer 50 usuaris comuns
    print(f"Evaluant recomanacions per a {len(sample_users)} usuaris comuns...")
    
    # Generar recomanacions per cada model
    ks = (5, 10, 20)
    knn_predictions = {}
    svd_predictions = {}
    
    print("\nGenerant recomanacions...")
    for user in sample_users:
        # Items que l'usuari JA HA VIST (no podem recomanar-los)
        seen_items = user_items.get(user, set())
        
        # Recomanacions KNN
        try:
            knn_recs_tuples = knn_inf.recommend(user, top_n=max(ks))
            knn_recs = [item for item, _ in knn_recs_tuples]
            # Filtrar items que l'usuari ja ha vist
            knn_recs_filtered = [item for item in knn_recs if item not in seen_items]
            knn_predictions[user] = knn_recs_filtered[:max(ks)]
        except Exception as e:
            print(f"Error en KNN recommend per usuari {user}: {e}")
            knn_predictions[user] = []
        
        # Recomanacions SVD
        try:
            svd_recs_tuples = svd_inf.recommend(user, top_n=max(ks))
            svd_recs = [item for item, _ in svd_recs_tuples]
            # Filtrar items que l'usuari ja ha vist
            svd_recs_filtered = [item for item in svd_recs if item not in seen_items]
            svd_predictions[user] = svd_recs_filtered[:max(ks)]
        except Exception as e:
            print(f"Error en SVD recommend per usuari {user}: {e}")
            svd_predictions[user] = []
    
    # Verificar que tenim recomanacions
    knn_with_recs = sum(1 for recs in knn_predictions.values() if len(recs) > 0)
    svd_with_recs = sum(1 for recs in svd_predictions.values() if len(recs) > 0)
    
    print(f"Usuaris amb recomanacions KNN vàlides: {knn_with_recs}/{len(sample_users)}")
    print(f"Usuaris amb recomanacions SVD vàlides: {svd_with_recs}/{len(sample_users)}")
    
    # Calcular mètriques per KNN
    if knn_with_recs > 0:
        print("\n--- Mètriques de RECOMANACIÓ per KNN ---")
        knn_results = evaluate_all(
            actuals={u: relevant_items[u] for u in sample_users if u in relevant_items},
            predicteds=knn_predictions,
            ks=ks
        )
        
        for k in ks:
            print(f"\nTop-{k}:")
            print(f"  Precision@{k}: {knn_results.get(f'Precision@{k}', 0):.4f}")
            print(f"  Recall@{k}:    {knn_results.get(f'Recall@{k}', 0):.4f}")
            print(f"  MAP@{k}:       {knn_results.get(f'MAP@{k}', 0):.4f}")
            print(f"  NDCG@{k}:      {knn_results.get(f'NDCG@{k}', 0):.4f}")
    else:
        print("\n⚠️  No es poden calcular mètriques de recomanació per KNN")
        knn_results = {}
    
    # Calcular mètriques per SVD
    if svd_with_recs > 0:
        print("\n--- Mètriques de RECOMANACIÓ per SVD ---")
        svd_results = evaluate_all(
            actuals={u: relevant_items[u] for u in sample_users if u in relevant_items},
            predicteds=svd_predictions,
            ks=ks
        )
        
        for k in ks:
            print(f"\nTop-{k}:")
            print(f"  Precision@{k}: {svd_results.get(f'Precision@{k}', 0):.4f}")
            print(f"  Recall@{k}:    {svd_results.get(f'Recall@{k}', 0):.4f}")
            print(f"  MAP@{k}:       {svd_results.get(f'MAP@{k}', 0):.4f}")
            print(f"  NDCG@{k}:      {svd_results.get(f'NDCG@{k}', 0):.4f}")
    else:
        print("\n⚠️  No es poden calcular mètriques de recomanació per SVD")
        svd_results = {}
    
    # ----- 7. Mètriques addicionals -----
    print("\n[6] Calculant mètriques addicionals...")
    
    # Coverage
    if knn_predictions and all_items and knn_with_recs > 0:
        knn_coverage = catalog_coverage(knn_predictions, all_items)
        print(f"\nCoverage KNN: {knn_coverage:.4f} ({knn_coverage*100:.1f}% dels items)")
    
    if svd_predictions and all_items and svd_with_recs > 0:
        svd_coverage = catalog_coverage(svd_predictions, all_items)
        print(f"Coverage SVD: {svd_coverage:.4f} ({svd_coverage*100:.1f}% dels items)")
    
    # Personalization
    if len(knn_predictions) >= 2 and knn_with_recs > 0:
        knn_pers = personalization(knn_predictions, 10)
        print(f"\nPersonalization@10 KNN: {knn_pers:.4f}")
    
    if len(svd_predictions) >= 2 and svd_with_recs > 0:
        svd_pers = personalization(svd_predictions, 10)
        print(f"Personalization@10 SVD: {svd_pers:.4f}")
    
    # MRR
    if 'MRR' in knn_results:
        print(f"\nMRR KNN: {knn_results['MRR']:.4f}")
    if 'MRR' in svd_results:
        print(f"MRR SVD: {svd_results['MRR']:.4f}")
    
    # ----- 8. Comparativa final -----
    print("\n" + "="*60)
    print("COMPARATIVA FINAL")
    print("="*60)
    
    print("\n🔍 RESUM:")
    print(f"- Dataset: {df.shape[0]} interaccions, {len(user_items)} usuaris, {len(all_items)} items")
    print(f"- Threshold per items rellevants: ≥{threshold}")
    print(f"- Usuaris avaluats: {len(sample_users)} (comuns entre models)")
    print(f"- Mostra rating: {sample_size} interaccions")
    
    print("\n🏆 COMPARATIVA PER MÈTRIQUES CLAU:")
    
    # Comparar RMSE si disponibles
    if not np.isnan(rmse_knn) and not np.isnan(rmse_svd):
        if rmse_knn < rmse_svd:
            diff = rmse_svd - rmse_knn
            print(f"  • RMSE: KNN guanya per {diff:.4f} ({rmse_knn:.4f} vs {rmse_svd:.4f})")
        else:
            diff = rmse_knn - rmse_svd
            print(f"  • RMSE: SVD guanya per {diff:.4f} ({rmse_svd:.4f} vs {rmse_knn:.4f})")
    
    # Comparar Precision@10 si disponibles
    if 'Precision@10' in knn_results and 'Precision@10' in svd_results:
        prec_knn_10 = knn_results['Precision@10']
        prec_svd_10 = svd_results['Precision@10']
        if prec_knn_10 > prec_svd_10:
            diff = prec_knn_10 - prec_svd_10
            print(f"  • Precision@10: KNN guanya per {diff:.4f} ({prec_knn_10:.4f} vs {prec_svd_10:.4f})")
        else:
            diff = prec_svd_10 - prec_knn_10
            print(f"  • Precision@10: SVD guanya per {diff:.4f} ({prec_svd_10:.4f} vs {prec_knn_10:.4f})")
    
    print("\n" + "="*60)
    print("AVALUACIÓ COMPLETADA")
    print("="*60)
    
    # ----- 9. Notes i limitacions -----
    print("\n📝 NOTES I LIMITACIONS:")
    print("1. ✅ Models carregats correctament amb classes d'inferència")
    print("2. ⚠️  Avaluació sobre mateixes dades d'entrenament (pot inflar resultats)")
    print("3. ✅ Filtrem items que l'usuari ja ha vist de les recomanacions")
    print("4. ✅ Considerem items amb rating ≥4.0 com a rellevants")
    print("5. 🔄 Per avaluació més realista, considerar split train-test")
    print("6. 📊 Resultats basats en mostra de 50 usuaris comuns i 1000 interaccions")