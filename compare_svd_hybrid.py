
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from surprise import Dataset, Reader, SVD

BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "data/ml-1m/ratings.dat"
MOVIES_PATH = BASE_DIR / "data/ml-1m/movies.dat"

# Load datasets
ratings = pd.read_csv(
    RATINGS_PATH, sep="::", engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    MOVIES_PATH, sep="::", engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

ratings = ratings[["user_id", "movie_id", "rating"]]

# Split: 70% training, 10% validation, 20% testing
train_val, test_df = train_test_split(
    ratings, test_size=0.20, random_state=123
)

train_df, val_df = train_test_split(
    train_val, test_size=0.125, random_state=42
)

reader = Reader(rating_scale=(1, 5))


def train_svd(data):
    dataset = Dataset.load_from_df(
        data[["user_id", "movie_id", "rating"]], reader
    )
    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=100,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.05,
        random_state=42
    )
    model.fit(trainset)
    return model


def genre_scores(training_data, pairs, shrinkage=10):
    """Calculate user genre preferences using training data only."""

    global_mean = training_data["rating"].mean()

    user_means = (
        training_data.groupby("user_id")["rating"].mean().to_dict()
    )

    joined = training_data.merge(
        movies[["movie_id", "genres"]],
        on="movie_id",
        how="left"
    )

    joined["genre"] = joined["genres"].fillna("").str.split("|")
    expanded = joined.explode("genre")

    genre_stats = (
        expanded[expanded["genre"] != ""]
        .groupby(["user_id", "genre"])["rating"]
        .agg(["sum", "count"])
    )

    preferences = {}

    for (user, genre), row in genre_stats.iterrows():
        user_mean = user_means.get(user, global_mean)

        # Smoothed genre preference
        score = (
            row["sum"] + shrinkage * user_mean
        ) / (row["count"] + shrinkage)

        preferences[(user, genre)] = score

    movie_genres = (
        movies.set_index("movie_id")["genres"]
        .fillna("")
        .to_dict()
    )

    scores = []

    for user, movie in pairs:
        user_mean = user_means.get(user, global_mean)
        genres = movie_genres.get(movie, "").split("|")

        values = [
            preferences[(user, genre)]
            for genre in genres
            if (user, genre) in preferences
        ]

        scores.append(
            float(np.mean(values)) if values else user_mean
        )

    return np.array(scores)


def evaluate(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    return mse, rmse, mae


# PHASE 1: Tune hybrid weight using validation data
print("\nTraining SVD for validation...")
validation_model = train_svd(train_df)

val_pairs = list(
    val_df[["user_id", "movie_id"]].itertuples(
        index=False, name=None
    )
)

val_testset = list(
    val_df.itertuples(index=False, name=None)
)

val_predictions = validation_model.test(val_testset)

val_svd = np.array([p.est for p in val_predictions])
val_genre = genre_scores(train_df, val_pairs)
val_actual = val_df["rating"].to_numpy()

best_alpha = 1.0
best_rmse = float("inf")

print("\nTesting hybrid weights...")

for alpha in np.arange(0.50, 1.001, 0.05):
    hybrid = alpha * val_svd + (1 - alpha) * val_genre

    rmse = np.sqrt(mean_squared_error(val_actual, hybrid))

    print(f"Alpha: {alpha:.2f} | Validation RMSE: {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha:.2f}")


# PHASE 2: Final training using 80% training data
print("\nTraining final SVD model...")
final_model = train_svd(train_val)

test_pairs = list(
    test_df[["user_id", "movie_id"]].itertuples(
        index=False, name=None
    )
)

testset = list(
    test_df.itertuples(index=False, name=None)
)

predictions = final_model.test(testset)

svd_predictions = np.array([p.est for p in predictions])
test_genre = genre_scores(train_val, test_pairs)
actual = test_df["rating"].to_numpy()

hybrid_predictions = (
    best_alpha * svd_predictions
    + (1 - best_alpha) * test_genre
)

# Evaluate both models
svd_metrics = evaluate(actual, svd_predictions)
hybrid_metrics = evaluate(actual, hybrid_predictions)

results = pd.DataFrame([
    {
        "Model": "Original SVD",
        "MSE": svd_metrics[0],
        "RMSE": svd_metrics[1],
        "MAE": svd_metrics[2]
    },
    {
        "Model": "Hybrid SVD",
        "MSE": hybrid_metrics[0],
        "RMSE": hybrid_metrics[1],
        "MAE": hybrid_metrics[2]
    }
])

print("\nFINAL TEST RESULTS")
print(results.to_string(index=False))
print(f"\nSelected alpha: {best_alpha:.2f}")

results.to_csv(
    BASE_DIR / "svd_hybrid_results.csv",
    index=False
)

# Generate comparison graphs
for metric in ["MSE", "RMSE", "MAE"]:
    plt.figure(figsize=(8, 5))

    plt.bar(results["Model"], results[metric])

    plt.title(f"Original SVD vs Hybrid SVD - {metric}")
    plt.ylabel(metric + " (Lower is Better)")
    plt.tight_layout()

    plt.savefig(
        BASE_DIR / f"svd_hybrid_{metric.lower()}.png",
        dpi=300
    )
    plt.close()

print("\nSaved:")
print("svd_hybrid_results.csv")
print("svd_hybrid_mse.png")
print("svd_hybrid_rmse.png")
print("svd_hybrid_mae.png")

if hybrid_metrics[1] < svd_metrics[1]:
    print("\nHybrid SVD improved RMSE!")
elif np.isclose(hybrid_metrics[1], svd_metrics[1]):
    print("\nHybrid SVD and Original SVD have equal RMSE.")
    print("No accuracy improvement was achieved.")
else:
    print("\nHybrid SVD did not improve RMSE.")
    print("Do not claim an accuracy improvement.")