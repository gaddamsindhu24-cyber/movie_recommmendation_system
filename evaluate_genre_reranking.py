
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

# Same split for both ranking methods
train, test = train_test_split(
    ratings,
    test_size=0.20,
    random_state=123
)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Train original SVD
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    train[["user_id", "movie_id", "rating"]],
    reader
)

trainset = data.build_full_trainset()

model = SVD(
    n_factors=100,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.05,
    random_state=42
)

print("Training SVD...")
model.fit(trainset)

# Learn genre preferences from training data only
genre_data = train.merge(
    movies[["movie_id", "genres"]],
    on="movie_id",
    how="left"
)

genre_data["genre"] = (
    genre_data["genres"].fillna("").str.split("|")
)

genre_data = genre_data.explode("genre")
genre_data = genre_data[genre_data["genre"] != ""]

user_genres = (
    genre_data.groupby(["user_id", "genre"])["rating"]
    .mean()
    .to_dict()
)

user_means = train.groupby("user_id")["rating"].mean().to_dict()
global_mean = train["rating"].mean()

# Build test relevance sets
test_relevant = (
    test[test["rating"] >= 4]
    .groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

# Exclude all movies rated in either training or test
# This avoids evaluating known test items as recommendations.
all_seen = (
    ratings.groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

movie_genres = (
    movies.set_index("movie_id")["genres"]
    .fillna("")
    .to_dict()
)

movie_ids = movies["movie_id"].unique()

# Use users with at least one relevant test movie
users = sorted(test_relevant.keys())

# Alpha values for ranking
alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

def ranking_metrics(recommended, relevant, k=10):
    hits = [
        1 if movie in relevant else 0
        for movie in recommended[:k]
    ]

    precision = sum(hits) / k
    recall = sum(hits) / len(relevant)

    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(np.array(hits) * discounts)

    ideal_hits = min(len(relevant), k)
    idcg = np.sum(discounts[:ideal_hits])

    ndcg = dcg / idcg if idcg > 0 else 0

    return precision, recall, ndcg


def recommend(user, alpha):
    seen = all_seen.get(user, set())
    candidates = [
        movie for movie in movie_ids
        if movie not in seen
    ]

    if not candidates:
        return []

    user_mean = user_means.get(user, global_mean)

    rows = []

    for movie in candidates:
        svd_score = model.predict(int(user), int(movie)).est

        genres = movie_genres.get(movie, "").split("|")

        genre_values = [
            user_genres[(user, genre)]
            for genre in genres
            if (user, genre) in user_genres
        ]

        genre_score = (
            np.mean(genre_values)
            if genre_values else user_mean
        )

        # Re-rank using both SVD and genre preference
        score = (
            alpha * svd_score
            + (1 - alpha) * genre_score
        )

        rows.append((movie, score))

    rows.sort(key=lambda x: x[1], reverse=True)

    return [movie for movie, score in rows[:10]]


# Tune alpha using a validation portion of training data
train_main, validation = train_test_split(
    train,
    test_size=0.10,
    random_state=42
)

# Use validation only to select alpha.
# Refit SVD on train_main for this validation stage.
val_data = Dataset.load_from_df(
    train_main[["user_id", "movie_id", "rating"]],
    reader
)

val_model = SVD(
    n_factors=100,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.05,
    random_state=42
)

val_model.fit(val_data.build_full_trainset())

# For a simple, consistent evaluation, select alpha
# using the final training model and validation relevance.
# The final test results remain untouched.
validation_relevant = (
    validation[validation["rating"] >= 4]
    .groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

# Validation candidate pool excludes train and validation items
validation_seen = (
    pd.concat([train_main, validation])
    .groupby("user_id")["movie_id"]
    .apply(set)
    .to_dict()
)

# Build validation genre preferences
val_genres = train_main.merge(
    movies[["movie_id", "genres"]],
    on="movie_id",
    how="left"
)
val_genres["genre"] = (
    val_genres["genres"].fillna("").str.split("|")
)
val_genres = val_genres.explode("genre")
val_genres = val_genres[val_genres["genre"] != ""]

val_user_genres = (
    val_genres.groupby(["user_id", "genre"])["rating"]
    .mean().to_dict()
)
val_user_means = train_main.groupby("user_id")["rating"].mean().to_dict()
val_global = train_main["rating"].mean()

def validation_recommend(user, alpha):
    candidates = [
        m for m in movie_ids
        if m not in validation_seen.get(user, set())
    ]

    scored = []

    for movie in candidates:
        svd_score = val_model.predict(int(user), int(movie)).est
        genres = movie_genres.get(movie, "").split("|")

        vals = [
            val_user_genres[(user, g)]
            for g in genres
            if (user, g) in val_user_genres
        ]

        genre_score = (
            np.mean(vals)
            if vals else val_user_means.get(user, val_global)
        )

        score = alpha * svd_score + (1 - alpha) * genre_score
        scored.append((movie, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, score in scored[:10]]


validation_users = sorted(validation_relevant.keys())
best_alpha = 1.0
best_ndcg = -1

print("\nSelecting alpha using validation data...")

for alpha in alphas:
    values = []

    for user in validation_users:
        recs = validation_recommend(user, alpha)
        values.append(
            ranking_metrics(
                recs, validation_relevant[user]
            )[2]
        )

    mean_ndcg = np.mean(values) if values else 0
    print(f"Alpha {alpha:.1f}: Validation NDCG@10 = {mean_ndcg:.4f}")

    if mean_ndcg > best_ndcg:
        best_ndcg = mean_ndcg
        best_alpha = alpha

print("\nSelected alpha:", best_alpha)

# Final SVD trained on the full 80% training set
print("\nTraining final SVD...")
model = SVD(
    n_factors=100,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.05,
    random_state=42
)
model.fit(trainset)

# Final test evaluation
results = []

print("\nEvaluating test users...")

for name, alpha in [
    ("Original SVD", 1.0),
    ("Genre-Aware SVD", best_alpha)
]:
    scores = []

    for user in users:
        recs = recommend(user, alpha)
        scores.append(
            ranking_metrics(recs, test_relevant[user])
        )

    scores = np.array(scores)

    results.append({
        "Model": name,
        "Precision@10": scores[:, 0].mean(),
        "Recall@10": scores[:, 1].mean(),
        "NDCG@10": scores[:, 2].mean()
    })

results_df = pd.DataFrame(results)

print("\nFINAL RANKING RESULTS")
print(results_df.to_string(index=False))
print("\nNumber of evaluated users:", len(users))

results_df.to_csv(
    BASE / "genre_reranking_results.csv",
    index=False
)

for metric in [
    "Precision@10", "Recall@10", "NDCG@10"
]:
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df[metric])
    plt.title(f"Original SVD vs Genre-Aware SVD: {metric}")
    plt.ylabel(metric + " (Higher is Better)")
    plt.tight_layout()
    filename = metric.lower().replace("@", "_")
    plt.savefig(
        BASE / f"genre_{filename}.png",
        dpi=300
    )
    plt.close()

print("\nGenerated ranking results and graphs.")
