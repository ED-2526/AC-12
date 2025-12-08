"""
Versió optiimitzada i utilotzant el predict_knn
"""

def recommend_knn(user_id, model, top_n=10):
    """
    Recomanacions top_n per un usuari.
    Reutilitza predict_knn per calcular les puntuacions.
    Retorna llista de tuples (item_id, predicció arrodonida 1..5)
    """
    if user_id not in model.user_index:
        return []

    u_idx = model.user_index[user_id]
    user_ratings = model.item_user_matrix[:, u_idx]
    unseen_items = np.where(user_ratings == 0)[0]

    # Cridem predict_knn per cada item no valorat
    preds = [(i, predict_knn(user_id, model.items[i], model)) for i in unseen_items]
    # Eliminem prediccions amb 0.0
    preds = [(i, score) for i, score in preds if score != 0.0]

    # Ordenem i agafem top_n
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]

    # Convertim indices a item_id
    return [(model.items[i], score) for i, score in top]
