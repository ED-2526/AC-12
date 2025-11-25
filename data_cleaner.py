import pandas as pd

def load_and_clean(path, min_ratings_per_user=20, min_ratings_per_item=100, fraction=None, seed=42):
    """
    Carrega i neteja el dataset.

    Args:
        path (str): path al CSV.
        min_ratings_per_user (int): mínim nombre de valoracions per usuari.
        min_ratings_per_item (int): mínim nombre de valoracions per item.
        fraction (float): si és diferent de None, agafar només una fracció aleatòria del dataset.
        seed (int): random seed per sampling.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["userID", "itemID", "rating", "timestamp"]
    )

    df = df.dropna(subset=['userID', 'itemID', 'rating'])

    df['userID'] = df['userID'].astype(str)
    df['itemID'] = df['itemID'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # TRACTAR DUPLICATS: (userID, itemID): fem la mitjana
    df = df.groupby(["userID", "itemID"], as_index=False).agg({
        "rating": "mean",
        "timestamp": "max"
    })

    # Filtrar usuaris amb poques valoracions totals
    df = df.groupby("userID").filter(lambda x: len(x) >= min_ratings_per_user)

    # Filtrar items amb poques valoracions totals
    df = df.groupby("itemID").filter(lambda x: len(x) >= min_ratings_per_item)

    # Sampling opcional
    if fraction is not None:
        df = df.sample(frac=fraction, random_state=seed)

    return df
