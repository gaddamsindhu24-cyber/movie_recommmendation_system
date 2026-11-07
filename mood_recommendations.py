from src.svd_recommender import get_svd_recommendations

MOOD_GENRES = {
    "Happy 😊": ["Comedy", "Animation", "Musical"],
    "Sad 😢": ["Drama", "Romance"],
    "Romantic ❤️": ["Romance", "Drama"],
    "Excited 🤩": ["Action", "Adventure", "Sci-Fi"],
    "Bored 😴": ["Thriller", "Adventure", "Mystery"]
}


def get_mood_recommendations(user_id, mood, n=10, genre=None):

    if mood not in MOOD_GENRES:
        return None

    # Generate personalized recommendations using SVD
    recommendations = get_svd_recommendations(
        user_id,
        n=3883,
        genre=genre
    )

    if recommendations.empty:
        return recommendations

    # Get genres related to the selected mood
    selected_genres = MOOD_GENRES[mood]

    pattern = "|".join(selected_genres)

    # Filter recommendations according to mood
    mood_movies = recommendations[
        recommendations["genres"].str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
    ].copy()

    # Return top N movies based on SVD predicted ratings
    return mood_movies.head(n).reset_index(drop=True)

def add_recommendation_explanations(recommendations, user_id, mood=None):

    from src.svd_recommender import ratings, movies

    if recommendations is None or recommendations.empty:
        return recommendations

    liked_movies = ratings[
        (ratings["UserID"] == int(user_id)) &
        (ratings["Rating"] >= 4)
    ]

    liked_movies = liked_movies.merge(
        movies[["MovieID", "Genres"]],
        on="MovieID",
        how="left"
    )

    preferred_genres = set()

    for genres in liked_movies["Genres"].dropna():
        preferred_genres.update(genres.split("|"))

    def explain(row):

        movie_genres = set(str(row["genres"]).split("|"))
        matching_genres = movie_genres.intersection(preferred_genres)

        reasons = []

        if matching_genres:
            reasons.append(
                "You previously rated "
                + ", ".join(sorted(matching_genres))
                + " movies highly."
            )
        else:
            reasons.append(
                "This movie introduces you to a different genre."
            )

        if mood and mood in MOOD_GENRES:
            mood_matches = movie_genres.intersection(MOOD_GENRES[mood])

            if mood_matches:
                reasons.append(
                    "It matches your selected "
                    + mood
                    + " mood through its "
                    + ", ".join(sorted(mood_matches))
                    + " genre."
                )

        if "PredictedRating" in row.index:
            reasons.append(
                "SVD predicted a rating of "
                + str(round(row["PredictedRating"], 2))
                + " out of 5."
            )

        return " ".join(reasons)

    recommendations = recommendations.copy()

    recommendations["Why This Movie?"] = recommendations.apply(
        explain,
        axis=1
    )

    return recommendations

