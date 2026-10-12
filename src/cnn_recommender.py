import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Reshape, Multiply,
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
ratings_path = BASE_DIR / "data/ml-1m/ratings.dat"
movies_path = BASE_DIR / "data/ml-1m/movies.dat"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cnn_model.keras"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
ratings = pd.read_csv(
    ratings_path,
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)

movies = pd.read_csv(
    movies_path,
    sep="::",
    engine="python",
    names=["MovieID", "Title", "Genres"],
    encoding="latin-1"
)

# Convert IDs to indexes
user_ids = sorted(ratings["UserID"].unique())
movie_ids = sorted(ratings["MovieID"].unique())

user_to_index = {
    user_id: index for index, user_id in enumerate(user_ids)
}
movie_to_index = {
    movie_id: index for index, movie_id in enumerate(movie_ids)
}

ratings["UserIndex"] = ratings["UserID"].map(user_to_index)
ratings["MovieIndex"] = ratings["MovieID"].map(movie_to_index)

X_user = ratings["UserIndex"].to_numpy(dtype=np.int32)
X_movie = ratings["MovieIndex"].to_numpy(dtype=np.int32)
y = ratings["Rating"].to_numpy(dtype=np.float32)


# Build the CNN architecture
def build_cnn_model():
    embedding_size = 16

    user_input = Input(shape=(1,), name="user_input")
    movie_input = Input(shape=(1,), name="movie_input")

    user_embedding = Embedding(
        input_dim=len(user_ids),
        output_dim=embedding_size,
        name="user_embedding"
    )(user_input)

    movie_embedding = Embedding(
        input_dim=len(movie_ids),
        output_dim=embedding_size,
        name="movie_embedding"
    )(movie_input)

    user_embedding = Reshape((embedding_size,))(user_embedding)
    movie_embedding = Reshape((embedding_size,))(movie_embedding)

    user_embedding = Reshape((embedding_size, 1))(user_embedding)
    movie_embedding = Reshape((1, embedding_size))(movie_embedding)

    interaction = Multiply()([user_embedding, movie_embedding])
    interaction = Reshape(
        (embedding_size, embedding_size, 1)
    )(interaction)

    x = Conv2D(
        32, kernel_size=(3, 3),
        activation="relu", padding="same"
    )(interaction)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(
        64, kernel_size=(3, 3),
        activation="relu", padding="same"
    )(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    output = Dense(1, activation="linear", name="rating")(x)

    model = Model(
        inputs=[user_input, movie_input],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# Load the model only. Never train automatically here.
if MODEL_PATH.exists():
    print("Loading saved CNN model...")
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model loaded successfully.")
else:
    cnn_model = None
    print("CNN model not trained yet. Run: python -m src.train_cnn")


# Generate recommendations
def get_cnn_recommendations(user_id, n=10, genre=None):
    """
    Generate recommendations using trained Deep CNN interaction model.
    Pre-filters candidates by genre if requested, and excludes already-watched movies.
    """
    if cnn_model is None:
        raise RuntimeError(
            "CNN model is not trained yet. "
            "Run: python -m src.train_cnn"
        )

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return pd.DataFrame(
            columns=["movie_id", "title", "genres", "PredictedRating"]
        )

    if user_id not in user_to_index:
        return pd.DataFrame(
            columns=["movie_id", "title", "genres", "PredictedRating"]
        )

    watched_movies = set(
        ratings.loc[ratings["UserID"] == user_id, "MovieID"]
    )

    candidates = movies[
        ~movies["MovieID"].isin(watched_movies)
    ].copy()

    candidates = candidates[
        candidates["MovieID"].isin(movie_to_index)
    ].copy()

    # Pre-filter candidate pool by genre if requested
    if genre and genre != "All Genres":
        candidates = candidates[
            candidates["Genres"].str.contains(genre, case=False, na=False)
        ].copy()

    if candidates.empty:
        return pd.DataFrame(
            columns=["movie_id", "title", "genres", "PredictedRating"]
        )

    user_index = user_to_index[user_id]

    user_array = np.full(
        len(candidates), user_index, dtype=np.int32
    )

    movie_array = candidates["MovieID"].map(
        movie_to_index
    ).to_numpy(dtype=np.int32)

    predictions = cnn_model.predict(
        [user_array, movie_array],
        batch_size=256,
        verbose=0
    ).flatten()

    candidates["PredictedRating"] = predictions

    candidates = candidates.sort_values(
        "PredictedRating", ascending=False
    ).head(n)

    result = candidates.rename(
        columns={
            "MovieID": "movie_id",
            "Title": "title",
            "Genres": "genres"
        }
    )

    return result[
        ["movie_id", "title", "genres", "PredictedRating"]
    ].reset_index(drop=True)
