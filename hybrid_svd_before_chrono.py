
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/ml-1m"

# Load MovieLens 1M
ratings = pd.read_csv(
    DATA / "ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    DATA / "movies.dat",
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

# Split: 70% training, 10% validation, 20% testing
train_val, test = train_test_split(
    ratings, test_size=0.20, random_state=123
)

train, validation = train_test_split(
    train_val, test_size=0.125, random_state=42
)

reader = Reader(rating_scale=(1, 5))

ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
MAX_USERS = 500
K = 10

movie_ids = movies["movie_id"].unique()
movie_genres = movies.set_index("movie_id")["genres"].fillna("").to_dict()


def train_svd(df):
    data = Dataset.load_from_df(
        df[["user_id", "movie_id", "rating"]], reader
    )
    model = SVD(
        n_factors=50,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.05,
        random_state=42
    )
    model.fit(data.build_full_trainset())
    return model



def build_genre_preferences(df, half_life_days=180):

    df = df.copy()

    # Calculate each user's average rating
    user_means = (
        df.groupby("user_id")["rating"]
        .mean()
        .to_dict()
    )

    global_mean = df["rating"].mean()

    # Convert timestamps into numeric values
    df["timestamp"] = pd.to_numeric(
        df["timestamp"]
    )

    # Find the latest rating timestamp for each user
    latest_timestamp = (
        df.groupby("user_id")["timestamp"]
        .max()
    )

    df["latest_timestamp"] = df["user_id"].map(
        latest_timestamp
    )

    # Calculate age of each rating in days
    df["age_days"] = (
        df["latest_timestamp"] - df["timestamp"]
    ) / 86400

    # Calculate time-decay weights
    df["time_weight"] = (
        0.5 ** (df["age_days"] / half_life_days)
    )

    # Attach movie genres
    expanded = df.merge(
        movies[["movie_id", "genres"]],
        on="movie_id",
        how="left"
    )

    # Expand movies into individual genres
    expanded["genre"] = (
        expanded["genres"]
        .fillna("")
        .str.split("|")
    )

    expanded = expanded.explode("genre")

    expanded = expanded[
        (expanded["genre"] != "") &
        (expanded["genre"] != "(no genres listed)")
    ]

    # Calculate weighted rating values
    expanded["weighted_rating"] = (
        expanded["rating"] * expanded["time_weight"]
    )

    grouped = expanded.groupby(
        ["user_id", "genre"]
    ).agg(
        weighted_sum=("weighted_rating", "sum"),
        total_weight=("time_weight", "sum")
    )

    preferences = {}

    # Smooth genre preferences using user's average rating
    for (user, genre), row in grouped.iterrows():

        user_mean = user_means.get(
            user, global_mean
        )

        preferences[(user, genre)] = (
            row["weighted_sum"] + 5 * user_mean
        ) / (
            row["total_weight"] + 5
        )

    return preferences, user_means, global_mean


    user_means = df.groupby("user_id")["rating"].mean().to_dict()
    global_mean = df["rating"].mean()

    expanded = df.merge(
        movies[["movie_id", "genres"]],
        on="movie_id",
        how="left"
    )

    expanded["genre"] = (
        expanded["genres"].fillna("").str.split("|")
    )
    expanded = expanded.explode("genre")
    expanded = expanded[
        (expanded["genre"] != "") &
        (expanded["genre"] != "(no genres listed)")
    ]

    grouped = expanded.groupby(
        ["user_id", "genre"]
    )["rating"].agg(["sum", "count"])

    # Smooth genre averages using the user's average rating
    preferences = {}

    for (user, genre), row in grouped.iterrows():
        user_mean = user_means.get(user, global_mean)
        preferences[(user, genre)] = (
            row["sum"] + 5 * user_mean
        ) / (row["count"] + 5)

    return preferences, user_means, global_mean


def metrics_at_10(recommended, relevant):
    hits = [
        1 if movie in relevant else 0
        for movie in recommended[:K]
    ]

    precision = sum(hits) / K
    recall = sum(hits) / len(relevant)

    dcg = sum(
        hit / np.log2(rank + 2)
        for rank, hit in enumerate(hits)
    )

    ideal_count = min(len(relevant), K)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_count)
    )

    ndcg = dcg / idcg if idcg else 0

    return precision, recall, ndcg


def evaluate(model, fit_df, eval_df):
    preferences, user_means, global_mean = (
        build_genre_preferences(fit_df)
    )

    seen = (
        fit_df.groupby("user_id")["movie_id"]
        .apply(set).to_dict()
    )

    relevant = (
        eval_df[eval_df["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set).to_dict()
    )

    # Keep users who have training history and relevant held-out movies
    users = [
        u for u in relevant
        if u in seen and len(relevant[u]) > 0
    ]

    # Sample users to keep runtime manageable
    rng = np.random.RandomState(42)
    if len(users) > MAX_USERS:
        users = sorted(
            rng.choice(users, MAX_USERS, replace=False)
        )

    scores = {alpha: [] for alpha in ALPHAS}

    for index, user in enumerate(users):
        # Exclude training-rated movies only.
        # Held-out relevant movies remain eligible candidates.
        candidates = [
            int(m) for m in movie_ids
            if m not in seen.get(user, set())
        ]

        if not candidates:
            continue

        user_mean = user_means.get(user, global_mean)

        svd_scores = np.array([
            model.predict(int(user), movie).est
            for movie in candidates
        ])

        genre_scores = []

        for movie in candidates:
            genres = movie_genres.get(movie, "").split("|")

            values = [
                preferences[(user, genre)]
                for genre in genres
                if (user, genre) in preferences
            ]

            genre_scores.append(
                np.mean(values) if values else user_mean
            )

        genre_scores = np.array(genre_scores)

        for alpha in ALPHAS:
            hybrid_scores = (
                alpha * svd_scores
                + (1 - alpha) * genre_scores
            )

            top_indices = np.argsort(-hybrid_scores)[:K]
            recommendations = [
                candidates[i] for i in top_indices
            ]

            scores[alpha].append(
                metrics_at_10(
                    recommendations, relevant[user]
                )
            )

        if (index + 1) % 100 == 0:
            print(f"Evaluated {index + 1} users")

    results = {}

    for alpha, values in scores.items():
        if values:
            results[alpha] = np.mean(values, axis=0)
        else:
            results[alpha] = np.zeros(3)

    print("Users evaluated:", len(users))
    return results


# 1. Tune alpha using validation data
print("\nTraining validation SVD...")
validation_model = train_svd(train)

print("\nTesting hybrid weights using validation NDCG@10...")
validation_results = evaluate(
    validation_model, train, validation
)

for alpha in ALPHAS:
    p, r, n = validation_results[alpha]
    print(
        f"Alpha {alpha:.1f} | "
        f"Precision@10: {p:.4f} | "
        f"Recall@10: {r:.4f} | "
        f"NDCG@10: {n:.4f}"
    )

best_alpha = max(
    ALPHAS,
    key=lambda a: validation_results[a][2]
)

print("\nBest alpha:", best_alpha)

# 2. Train final model using training + validation data
print("\nTraining final SVD...")
final_model = train_svd(train_val)

# 3. Evaluate on untouched test data
print("\nEvaluating final test results...")
test_results = evaluate(
    final_model, train_val, test
)
print("\nTEST RESULTS FOR ALL ALPHA VALUES")

for alpha in ALPHAS:
    p, r, n = test_results[alpha]

    print(
        f"Alpha {alpha:.1f} | "
        f"Precision@10: {p:.4f} | "
        f"Recall@10: {r:.4f} | "
        f"NDCG@10: {n:.4f}"
    )

original = test_results[1.0]
hybrid = test_results[best_alpha]

output = pd.DataFrame([
    {
        "Model": "Original SVD",
        "Precision@10": original[0],
        "Recall@10": original[1],
        "NDCG@10": original[2]
    },
    {
        "Model": "Hybrid SVD",
        "Precision@10": hybrid[0],
        "Recall@10": hybrid[1],
        "NDCG@10": hybrid[2]
    }
])

print("\nFINAL TEST RESULTS")
print(output.to_string(index=False))
print("\nSelected alpha:", best_alpha)

output.to_csv(
    BASE / "hybrid_svd_ranking_results.csv",
    index=False
)

# Plot comparison
for metric in [
    "Precision@10", "Recall@10", "NDCG@10"
]:
    plt.figure(figsize=(8, 5))
    plt.bar(output["Model"], output[metric])
    plt.ylabel(metric)
    plt.title("Original SVD vs Hybrid SVD")
    plt.tight_layout()

    filename = metric.lower().replace("@", "_")
    plt.savefig(
        BASE / f"hybrid_{filename}.png",
        dpi=300
    )
    plt.close()

print("\nSaved results and comparison graphs.")
