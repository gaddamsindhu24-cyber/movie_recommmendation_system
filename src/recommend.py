from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

# Load MovieLens data
movies = pd.read_csv(
    BASE_DIR / "data/ml-1m/movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

ratings = pd.read_csv(
    BASE_DIR / "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
    encoding="latin-1"
)

# Create User-Movie Matrix
user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# Item-item similarity
movie_similarity = cosine_similarity(user_movie_matrix.T)


def get_recommendations(user_id, n=10, genre=None):
    """
    Generate item-based collaborative filtering (KNN/Cosine Similarity) recommendations.
    Filters by genre if specified, and strictly excludes movies already rated by the user.
    """
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return pd.DataFrame(columns=["movie_id", "title", "genres", "recommendation_score"])

    if user_id not in user_movie_matrix.index:
        return pd.DataFrame(columns=["movie_id", "title", "genres", "recommendation_score"])

    # Movies rated by the user
    user_ratings = ratings[ratings["user_id"] == user_id]
    watched_movie_ids = set(user_ratings["movie_id"])

    # Movies the user liked (rating >= 4, or top rated if none >= 4)
    liked_movies = user_ratings[user_ratings["rating"] >= 4]
    if liked_movies.empty:
        liked_movies = user_ratings[user_ratings["rating"] >= 3]
    if liked_movies.empty:
        liked_movies = user_ratings

    scores = {}

    for _, row in liked_movies.iterrows():
        movie_id = row["movie_id"]
        rating = row["rating"]

        if movie_id not in user_movie_matrix.columns:
            continue

        movie_index = user_movie_matrix.columns.get_loc(movie_id)
        similarities = movie_similarity[movie_index]

        for i, similarity in enumerate(similarities):
            recommended_movie_id = user_movie_matrix.columns[i]

            # Don't recommend movies already rated or with non-positive similarity
            if recommended_movie_id in watched_movie_ids or similarity <= 0:
                continue

            scores[recommended_movie_id] = (
                scores.get(recommended_movie_id, 0) + similarity * rating
            )

    if not scores:
        return pd.DataFrame(columns=["movie_id", "title", "genres", "recommendation_score"])

    result = pd.DataFrame(
        list(scores.items()),
        columns=["movie_id", "recommendation_score"]
    )

    # Add movie information
    result = result.merge(
        movies,
        on="movie_id",
        how="left"
    )

    # Filter by genre if requested
    if genre and genre != "All Genres":
        result = result[
            result["genres"].str.contains(genre, case=False, na=False)
        ]

    # Sort recommendations by score
    result = result.sort_values(
        "recommendation_score",
        ascending=False
    ).head(n)

    return result[
        ["movie_id", "title", "genres", "recommendation_score"]
    ].reset_index(drop=True)
