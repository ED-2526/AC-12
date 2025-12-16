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
    df = pd.read_csv(path, header=None, names=["userID", "itemID", "rating", "timestamp"]) 
    df = df.dropna(subset=['userID', 'itemID', 'rating']) 
    df['userID'] = df['userID'].astype(str)
    df['itemID'] = df['itemID'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce') 
    df = df.dropna(subset=['rating']) 
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') 
    df = df.groupby(["userID", "itemID"], as_index=False).agg({"rating": "mean", "timestamp": "max"}) #DUPLICATS: fem la mitjana
    df = df.groupby("userID").filter(lambda x: len(x) >= min_ratings_per_user)
    df = df.groupby("itemID").filter(lambda x: len(x) >= min_ratings_per_item)  
    if fraction is not None:
        df = df.sample(frac=fraction, random_state=seed)
    df.to_csv("cleaned_data.csv", index=False)
    return df

def visualize_dataset(df):
    """
    Mostrar la informació més rellevant d’un dataset ja netejat.

    Args:
        df (pandas.DataFrame): dataset netejat a visualitzar
    """
    print("INFORMACIÓ DEL DATASET NETEJAT")
    print("\nPrimeres 10 files: ") 
    print(df.head(10))
    print("\nMida del dataset netejat:", df.shape)
    print("\nNombre d’usuaris:", df["userID"].nunique()) 
    print("Nombre d’items:", df["itemID"].nunique())  
    #print("\nEstadístiques dels usuaris: "df["userID"].value_counts().describe())
    #print("\nEstadístiques dels items: "df["itemID"].value_counts().describe())
    print("\nUsuaris més actius:", df["userID"].value_counts().head(10))
    print("\nItems més valorats:", df["itemID"].value_counts().head(10))



