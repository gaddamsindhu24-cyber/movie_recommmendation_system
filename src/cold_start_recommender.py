from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RATINGS_PATH = BASE_DIR / "data" / "ml-1m" / "ratings.dat"
MOVIES_PATH = BASE_DIR / "data" / "ml-1m" / "movies.dat"


def get_cold_start_recommendations(
    selected_genres,
    n=10,
    min_ratings=50,
    profile_feedback=None
):
    """
    Recommend movies to new users using:
    1. Selected genres
    2. Bayesian popularity score
    3. Previously submitted Like/Dislike and rating feedback
    """

    columns = [
        "movie_id", "title", "genres",
        "AverageRating", "RatingCount",
        "PopularityScore", "PersonalizedScore",
        "recommendation_score"
    ]

    if not selected_genres:
        return pd.DataFrame(columns=columns)

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

    movie_stats = ratings.groupby("MovieID").agg(
        AverageRating=("Rating", "mean"),
        RatingCount=("Rating", "count")
    ).reset_index()

    result = movies.merge(
        movie_stats,
        on="MovieID",
        how="inner"
    )

    result = result[result["RatingCount"] >= min_ratings].copy()

    global_average = ratings["Rating"].mean()
    m = min_ratings

    result["PopularityScore"] = (
        (result["RatingCount"] / (result["RatingCount"] + m))
        * result["AverageRating"]
        +
        (m / (result["RatingCount"] + m))
        * global_average
    )

    selected_genres = set(selected_genres)

    result = result[
        result["Genres"].apply(
            lambda genres: bool(
                selected_genres.intersection(str(genres).split("|"))
            )
        )
    ].copy()

    # Feedback-based personalization
    liked_genres = set()
    disliked_genres = set()

    if profile_feedback is not None and not profile_feedback.empty:

        for _, feedback_row in profile_feedback.iterrows():

            movie_id = feedback_row["movie_id"]

            movie_match = movies[movies["MovieID"] == movie_id]

            if movie_match.empty:
                continue

            movie_genres = set(
                str(movie_match.iloc[0]["Genres"]).split("|")
            )

            choice = feedback_row.get("feedback")
            rating = feedback_row.get("rating")

            try:
                rating = int(rating) if pd.notna(rating) else None
            except (ValueError, TypeError):
                rating = None

            if choice == "Dislike" or (rating is not None and rating <= 2):
                disliked_genres.update(movie_genres)

            if choice == "Like" or (rating is not None and rating >= 4):
                liked_genres.update(movie_genres)

            # Do not recommend a movie the user has already rated
            result = result[result["MovieID"] != movie_id]

    def calculate_personalized_score(movie_genres, popularity_score):
        genres = set(str(movie_genres).split("|"))

        liked_match = (
            len(genres.intersection(liked_genres)) / len(genres)
            if genres else 0
        )

        disliked_match = (
            len(genres.intersection(disliked_genres)) / len(genres)
            if genres else 0
        )

        return (
            popularity_score
            + (0.15 * liked_match)
            - (0.10 * disliked_match)
        )

    result["PersonalizedScore"] = result.apply(
        lambda row: calculate_personalized_score(
            row["Genres"],
            row["PopularityScore"]
        ),
        axis=1
    )

    # Exclude movies explicitly marked Dislike
    if profile_feedback is not None and not profile_feedback.empty:
        disliked_ids = profile_feedback.loc[
            profile_feedback["feedback"] == "Dislike",
            "movie_id"
        ].dropna().astype(int).tolist()

        result = result[~result["MovieID"].isin(disliked_ids)]

    result = result.sort_values(
        by=["PersonalizedScore", "RatingCount"],
        ascending=False
    )

    result = result.rename(
        columns={
            "MovieID": "movie_id",
            "Title": "title",
            "Genres": "genres"
        }
    )

    result["recommendation_score"] = result["PersonalizedScore"]

    return result[columns].head(n).reset_index(drop=True)
