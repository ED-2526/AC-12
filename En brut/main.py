from data_cleaner import load_and_clean

def visualize_dataset(df):
    print("INFORMACIÓ DEL DATASET NETEJAT")

    print("\nPrimeres 10 files:")
    print(df.head(10))

    print("\nShape final del dataset:", df.shape)

    # Comptar valoracions per usuari i item
    user_counts = df["userID"].value_counts()
    item_counts = df["itemID"].value_counts()

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


if __name__ == "__main__":
    PATH = r"C:\Users\laura\Desktop\AC\projecte\archive\ratings_Electronics.csv"
    
    # carregar i netejar
    df = load_and_clean(PATH)
    
    # mostrar informació del dataset
    visualize_dataset(df)
    
    