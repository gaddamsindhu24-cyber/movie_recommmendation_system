from surprise import Dataset, Reader, NMF
import pandas as pd


def train_nmf(
    train_df,
    n_factors=20,
    n_epochs=30,
    random_state=42
):
    """
    Train Non-negative Matrix Factorization
    using the MovieLens training ratings.
    """

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        train_df[["user_id", "movie_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    nmf_model = NMF(
        n_factors=n_factors,
        n_epochs=n_epochs,
        random_state=random_state
    )

    nmf_model.fit(trainset)

    print("NMF model trained successfully!")

    return nmf_model


def predict_nmf(nmf_model, test_df):
    """
    Predict ratings for the test dataset.
    """

    predictions = [
        nmf_model.predict(
            int(row.user_id),
            int(row.movie_id)
        ).est
        for row in test_df.itertuples(index=False)
    ]

    return pd.Series(predictions, index=test_df.index)
