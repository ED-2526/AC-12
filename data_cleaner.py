import pandas as pd
import matplotlib.pyplot as plt



def plot_rating_distribution(df, title):
    """
    Dibuixa un histograma de la distribució de ratings.
    """
    plt.figure()
    plt.hist(df["rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    plt.xlabel("Rating")
    plt.ylabel("Nombre de valoracions")
    plt.title(title)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(axis="y")
    plt.show()


def load_and_clean(path, min_ratings_per_user=50, min_ratings_per_item=110, fraction=None, seed=42, plot=True):
    """
    Carrega i neteja el dataset.

    Args:
        path (str): path al CSV.
        min_ratings_per_user (int): mínim nombre de valoracions per usuari.
        min_ratings_per_item (int): mínim nombre de valoracions per item.
        fraction (float): si és diferent de None, agafar només una fracció aleatòria del dataset.
        seed (int): random seed per sampling.
        plot (bool): si True, mostra gràfics abans/després.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["userID", "itemID", "rating", "timestamp"]
    )
    
    # Gràfic ABANS de netejar
    if plot: 
        plot_rating_distribution(df, "Distribució de ratings — ABANS de netejar")


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
        
    # Gràfic DESPRÉS de netejar
    if plot:
        plot_rating_distribution(df, "Distribució de ratings — DESPRÉS de netejar")

    df.to_csv("cleaned_data.csv", index=False)
    
    return df


def visualize_dataset(df):
    print("INFORMACIÓ DEL DATASET NETEJAT")

    print("\nPrimeres 10 files:")
    print(df.head(10))

    print("\nShape final del dataset:", df.shape)
    #files: df.shape[0], columnes: df.shape[1]

    # Comptar valoracions per usuari i item
    user_counts = df["userID"].value_counts() #quantes vegades apareix cada usuari
    item_counts = df["itemID"].value_counts() #quantes vegades apareix cada ítem

    print("\nNombre d’usuaris únics:", df["userID"].nunique())
    print("Nombre d’items únics:", df["itemID"].nunique())

    print("\nEstadístiques de valoracions per usuari:")
    print(user_counts.describe())

    print("\nEstadístiques de valoracions per item:")
    print(item_counts.describe())

    print("\nExemple d’usuaris més actius:")
    print(user_counts.head())

    print("\nExemple d’items més valorats:")
    print(item_counts.head())
    

