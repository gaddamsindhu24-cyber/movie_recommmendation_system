import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load movies
movies = pd.read_csv(
    "data/ml-1m/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"]
)

# Load ratings
ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# Movie similarity
movie_similarity = cosine_similarity(user_movie_matrix.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

# Target user
target_user = 1

user_ratings = ratings[
    ratings["user_id"] == target_user
]

liked_movies = user_ratings[
    user_ratings["rating"] >= 4
]

# Calculate recommendation scores
recommendation_scores = {}

for _, row in liked_movies.iterrows():

    movie_id = row["movie_id"]
    user_rating = row["rating"]

    similar_movies = movie_similarity_df[movie_id]

    for similar_movie_id, similarity_score in similar_movies.items():

        if similar_movie_id not in user_ratings["movie_id"].values:

            score = similarity_score * user_rating

            recommendation_scores[similar_movie_id] = (
                recommendation_scores.get(similar_movie_id, 0)
                + score
            )

# Top 10
recommended_movie_ids = sorted(
    recommendation_scores,
    key=recommendation_scores.get,
    reverse=True
)[:10]

recommendations = movies[
    movies["movie_id"].isin(recommended_movie_ids)
].copy()

recommendations["recommendation_score"] = (
    recommendations["movie_id"].map(recommendation_scores)
)

recommendations = recommendations.sort_values(
    "recommendation_score",
    ascending=False
)

# Save recommendations
recommendations.to_csv(
    "data/recommendations.csv",
    index=False
)

print("\nRecommendations saved to:")
print("data/recommendations.csv")

print("\nTop 10 Recommendations:")
print(
    recommendations[
        ["movie_id", "title", "genres", "recommendation_score"]
    ].to_string(index=False)
)
