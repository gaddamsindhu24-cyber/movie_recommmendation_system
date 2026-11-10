from surprise import Dataset, Reader, CoClustering
import pandas as pd


def train_coclustering(
    train_df,
    n_cltr_u=3,
    n_cltr_i=3,
    n_epochs=20,
    random_state=42
):
    """
    Train CoClustering recommendation model.
    """

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        train_df[["user_id", "movie_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    model = CoClustering(
        n_cltr_u=n_cltr_u,
        n_cltr_i=n_cltr_i,
        n_epochs=n_epochs,
        random_state=random_state
    )

    model.fit(trainset)

    print("CoClustering model trained successfully!")

    return model


def predict_coclustering(model, test_df):
    """
    Predict ratings using the trained CoClustering model.
    """

    predictions = [
        model.predict(
            int(row.user_id),
            int(row.movie_id)
        ).est
        for row in test_df.itertuples(index=False)
    ]

    return pd.Series(predictions, index=test_df.index)
