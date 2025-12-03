import pandas as pd


def load_and_clean(path, min_ratings_per_user=20, min_ratings_per_item=100, fraction=None, seed=42):
    """
    Carrega i neteja el dataset.

    Args:
        path (str): path al CSV.
        min_ratings_per_user (int): mínim nombre de valoracions per usuari.
        min_ratings_per_item (int): mínim nombre de valoracions per item.
        fraction (float): si és True, agafar només una fracció aleatòria del dataset.
        seed (int): random seed per sampling.
    """
    df = pd.read_csv(path, header=None, names=["userID", "itemID", "rating", "timestamp"]) #llegeix l'arxiu csv i posa nom a cada columna (no es veu)
    #print(df.head(10)) 10 primeres del dataset sencer
    df = df.dropna(subset=['userID', 'itemID', 'rating']) #eliminem les files on algun d'aquests camps sigui nul
    df['userID'] = df['userID'].astype(str) #passem a string l'userID
    df['itemID'] = df['itemID'].astype(str) #passem a string l'itemID
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce') #converteix ratings a numeric (int o float) i si no es pot es NaN
    df = df.dropna(subset=['rating']) #Eliminem les files on sigui NaN
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') #converteix timestap a numeric (int o float) i si no es pot es NaN
    #DUPLICATS (userID, itemID): fem la mitjana
    df = df.groupby(["userID", "itemID"], as_index=False).agg({"rating": "mean", "timestamp": "max"}) #agrupem mateix user-item, segueixen sent columnes, fem la mitjana del rating i el max timestamp
    # Filtrar usuaris amb poques valoracions totals
    df = df.groupby("userID").filter(lambda x: len(x) >= min_ratings_per_user)
    # Filtrar items amb poques valoracions totals
    df = df.groupby("itemID").filter(lambda x: len(x) >= min_ratings_per_item)  
    #Sampling opcional
    if fraction is True:
        df = df.sample(frac=fraction, random_state=seed)
    return df


def visualize_dataset(df):
    """
    Mostrar la informació més rellevant d’un dataset ja netejat.

    Args:
        df (pandas.DataFrame): dataset netejat a visualitzar
    """
    print("INFORMACIÓ DEL DATASET NETEJAT")
    print("\nPrimeres 10 files:") #del sample, ordenat alfabeticament
    print(df.head(10))
    print("\nMida del dataset netejat:", df.shape)
    print("\nNombre d’usuaris:", df["userID"].nunique()) #selecciona la col usuaris i compta valors unics
    print("Nombre d’items:", df["itemID"].nunique())  
    #print("\nEstadístiques dels usuaris: "df["userID"].value_counts().describe())
    #print("\nEstadístiques dels items: "df["itemID"].value_counts().describe())
    print("\nUsuaris més actius:", df["userID"].value_counts().head(10)) #de més gran a més petit
    print("\nItems més valorats:", df["itemID"].value_counts().head(10))
