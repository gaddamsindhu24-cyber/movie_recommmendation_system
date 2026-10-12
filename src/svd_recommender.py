from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

RATINGS_PATH = BASE_DIR / "data/ml-1m/ratings.dat"
MOVIES_PATH = BASE_DIR / "data/ml-1m/movies.dat"
MODEL_PATH = BASE_DIR / "models/svd_model.pkl"

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)

movies = pd.read_csv(
    MOVIES_PATH,
    sep="::",
    engine="python",
    names=["MovieID", "Title", "Genres"],
    encoding="latin-1"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Saved SVD model not found. Run: python -m src.train_svd"
    )

print("Loading saved SVD model...")
model = joblib.load(MODEL_PATH)
print("SVD model loaded successfully.")


def get_svd_recommendations(user_id, n=10, genre=None):
    """
    Generate recommendations using trained SVD matrix factorization.
    Supports candidate filtering by genre and strictly excludes movies already rated by the user.
    """
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return pd.DataFrame(
            columns=["movie_id", "title", "genres", "PredictedRating"]
        )

    if user_id not in ratings["UserID"].values:
        return pd.DataFrame(
            columns=["movie_id", "title", "genres", "PredictedRating"]
        )

    watched_movies = set(
        ratings.loc[ratings["UserID"] == user_id, "MovieID"]
    )

    candidates = movies[
        ~movies["MovieID"].isin(watched_movies)
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

    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in candidates["MovieID"]
    ]

    predictions_df = pd.DataFrame(
        predictions,
        columns=["MovieID", "PredictedRating"]
    )

    predictions_df = predictions_df.nlargest(
        n, "PredictedRating"
    )

    result = predictions_df.merge(
        movies, on="MovieID", how="left"
    )

    return result.rename(
        columns={
            "MovieID": "movie_id",
            "Title": "title",
            "Genres": "genres"
        }
    )[
        ["movie_id", "title", "genres", "PredictedRating"]
    ].reset_index(drop=True)
