import pandas as pd

# File paths
movies_path = "data/ml-1m/movies.dat"
ratings_path = "data/ml-1m/ratings.dat"
users_path = "data/ml-1m/users.dat"

# Load movies
movies = pd.read_csv(
    movies_path,
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"]
)

# Load ratings
ratings = pd.read_csv(
    ratings_path,
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Load users
users = pd.read_csv(
    users_path,
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["user_id", "gender", "age", "occupation", "zip_code"]
)

# Display information
print("Movies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())

print("\nUsers:")
print(users.head())

print("\nDataset sizes:")
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)
