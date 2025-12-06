import pandas as pd
from lenskit import batch


DATA_PATH = os.path.join(ROOT, "ratings_Electronics(1).csv")

ratings = load_and_clean(DATA_PATH)
ratings = ratings.rename(columns={"userID": "user", "itemID": "item", "rating": "rating"}) #LensKit necessita un DataFrame user, item, rating amb aquests noms exactes.



def make_train_test(df):
    test = df.groupby("user").tail(1)
    train = df.drop(test.index)
    return train, test

train_lk, test_lk = make_train_test(lk_df)


model = train_itemknn(train_original_df)


train_original = train_lk.rename(columns={
    "user": "userID",
    "item": "itemID"
})
