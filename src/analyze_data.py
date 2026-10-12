import pandas as pd

# Load data
movies = pd.read_csv(
    "data/ml-1m/movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"]
)

ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

users = pd.read_csv(
    "data/ml-1m/users.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["user_id", "gender", "age", "occupation", "zip_code"]
)

# 1. Average rating
print("\nAverage Rating:")
print(ratings["rating"].mean())

# 2. Most-rated movies
movie_rating_counts = ratings["movie_id"].value_counts().head(10)

most_rated = movie_rating_counts.reset_index()
most_rated.columns = ["movie_id", "rating_count"]

most_rated = most_rated.merge(
    movies[["movie_id", "title"]],
    on="movie_id"
)

print("\nTop 10 Most Rated Movies:")
print(most_rated[["title", "rating_count"]])

# 3. Ratings per user
ratings_per_user = ratings.groupby("user_id").size()

print("\nAverage Ratings Per User:")
print(ratings_per_user.mean())

# 4. Rating distribution
print("\nRating Distribution:")
print(ratings["rating"].value_counts().sort_index())

# 5. Most common genres
genre_counts = {}

for genres in movies["genres"]:
    for genre in genres.split("|"):
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

genre_counts = sorted(
    genre_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 10 Genres:")
for genre, count in genre_counts[:10]:
    print(genre, ":", count)
