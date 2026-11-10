
from pathlib import Path

import numpy as np
import pandas as pd

from src.svd_recommender import ratings, movies, model


ALPHA = 0.6
HALF_LIFE_DAYS = 180


def get_hybrid_recommendations(user_id, n=10, genre=None):
    """Generate recommendations using SVD predictions and genre preferences."""

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return pd.DataFrame()

    user_ratings = ratings[
        ratings["UserID"] == user_id
    ].copy()

    if user_ratings.empty:
        return pd.DataFrame()

    # Exclude movies already rated by this user.
    watched_movies = set(user_ratings["MovieID"].astype(int))

    candidates = movies[
        ~movies["MovieID"].isin(watched_movies)
    ].copy()

    # Apply the selected genre filter.
    if genre and genre != "All Genres":
        candidates = candidates[
            candidates["Genres"].str.contains(
                genre, case=False, na=False
            )
        ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Calculate the user's average rating.
    user_mean = float(user_ratings["Rating"].mean())

    # Calculate recency weights using the user's latest rating.
    user_ratings["Timestamp"] = pd.to_numeric(
        user_ratings["Timestamp"]
    )

    latest_timestamp = user_ratings["Timestamp"].max()

    user_ratings["age_days"] = (
        latest_timestamp - user_ratings["Timestamp"]
    ) / 86400

    user_ratings["time_weight"] = (
        0.5 ** (
            user_ratings["age_days"] / HALF_LIFE_DAYS
        )
    )

    # Attach movie genres to the user's rating history.
    history = user_ratings.merge(
        movies[["MovieID", "Genres"]],
        on="MovieID",
        how="left"
    )

    history["genre"] = (
        history["Genres"]
        .fillna("")
        .str.split("|")
    )

    history = history.explode("genre")

    history = history[
        (history["genre"] != "") &
        (history["genre"] != "(no genres listed)")
    ]

    history["weighted_rating"] = (
        history["Rating"] * history["time_weight"]
    )

    grouped = history.groupby("genre").agg(
        weighted_sum=("weighted_rating", "sum"),
        total_weight=("time_weight", "sum")
    )

    # Smooth genre preferences toward the user's average rating.
    genre_preferences = {}

    for movie_genre, row in grouped.iterrows():
        genre_preferences[movie_genre] = (
            row["weighted_sum"] + 5 * user_mean
        ) / (
            row["total_weight"] + 5
        )

    # Calculate SVD predictions and hybrid scores.
    svd_scores = []
    hybrid_scores = []

    for _, movie in candidates.iterrows():
        movie_id = int(movie["MovieID"])

        svd_score = model.predict(
            user_id, movie_id
        ).est

        movie_genres = str(movie["Genres"]).split("|")

        preference_values = [
            genre_preferences[g]
            for g in movie_genres
            if g in genre_preferences
        ]

        genre_score = (
            float(np.mean(preference_values))
            if preference_values
            else user_mean
        )

        hybrid_score = (
            ALPHA * svd_score
            + (1 - ALPHA) * genre_score
        )

        svd_scores.append(svd_score)
        hybrid_scores.append(hybrid_score)

    candidates["PredictedRating"] = svd_scores
    candidates["HybridScore"] = hybrid_scores
    candidates["recommendation_score"] = hybrid_scores

    # Rank by the hybrid score, not the original SVD prediction.
    result = candidates.nlargest(
        n, "HybridScore"
    )

    return result.rename(
        columns={
            "MovieID": "movie_id",
            "Title": "title",
            "Genres": "genres"
        }
    )[
        [
            "movie_id",
            "title",
            "genres",
            "PredictedRating",
            "HybridScore",
            "recommendation_score"
        ]
    ].reset_index(drop=True)
