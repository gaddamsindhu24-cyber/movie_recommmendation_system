
import pandas as pd
from src.cold_start_recommender import get_cold_start_recommendations

result = get_cold_start_recommendations(
    selected_genres=["Comedy", "Drama"],
    n=10
)

print("\n========== BAYESIAN POPULARITY RANKING ==========\n")

print(result[
    ["title", "AverageRating", "RatingCount", "PopularityScore"]
].to_string(index=False))

print("\n========== RAW AVERAGE RATING COMPARISON ==========\n")

ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)

movies = pd.read_csv(
    "data/ml-1m/movies.dat",
    sep="::",
    engine="python",
    names=["MovieID", "Title", "Genres"],
    encoding="latin-1"
)

stats = ratings.groupby("MovieID").agg(
    AverageRating=("Rating", "mean"),
    RatingCount=("Rating", "count")
).reset_index()

comparison = movies.merge(stats, on="MovieID")

print(
    comparison.sort_values(
        "AverageRating", ascending=False
    )[["Title", "AverageRating", "RatingCount"]]
    .head(10).to_string(index=False)
)

