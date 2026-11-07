from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Reshape, Multiply,
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "data/ml-1m/ratings.dat"
MOVIES_PATH = BASE_DIR / "data/ml-1m/movies.dat"

DEMO_USER_ID = 1
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load MovieLens dataset
# --------------------------------------------------

print("\nLoading MovieLens dataset...")

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    MOVIES_PATH,
    sep="::",
    engine="python",
    names=["movie_id", "title", "genres"],
    encoding="latin-1"
)

# Same training and testing records for all models
train_df, test_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Total ratings:", len(ratings))
print("Training ratings:", len(train_df))
print("Testing ratings:", len(test_df))

# --------------------------------------------------
# 3. KNN model: item-based collaborative filtering
# --------------------------------------------------

print("\nTraining KNN model...")

user_movie_matrix = train_df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

item_similarity = cosine_similarity(user_movie_matrix.T)

knn_movie_ids = list(user_movie_matrix.columns)
knn_movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(knn_movie_ids)
}

user_history = {
    user_id: group[["movie_id", "rating"]]
    for user_id, group in train_df.groupby("user_id")
}

global_mean = train_df["rating"].mean()

knn_predictions = []

print("Evaluating KNN on test ratings...")

for row in test_df.itertuples(index=False):
    history = user_history.get(row.user_id)

    if row.movie_id not in knn_movie_to_index or history is None:
        knn_predictions.append(global_mean)
        continue

    history_indices = history["movie_id"].map(
        knn_movie_to_index
    ).dropna().astype(int).to_numpy()

    history_ratings = history.loc[
        history_indices if len(history_indices) == 0 else
        history.index[:0], "rating"
    ].to_numpy() if False else None

    # Keep only movies that exist in the KNN training matrix
    valid_history = history[
        history["movie_id"].isin(knn_movie_to_index)
    ]

    history_indices = valid_history["movie_id"].map(
        knn_movie_to_index
    ).to_numpy(dtype=int)

    history_ratings = valid_history["rating"].to_numpy()

    if len(history_indices) == 0:
        knn_predictions.append(global_mean)
        continue

    target_index = knn_movie_to_index[row.movie_id]

    similarities = item_similarity[target_index, history_indices]
    similarities = np.maximum(similarities, 0)

    if similarities.sum() == 0:
        prediction = global_mean
    else:
        prediction = np.dot(
            similarities, history_ratings
        ) / similarities.sum()

    knn_predictions.append(np.clip(prediction, 1, 5))

# --------------------------------------------------
# 4. SVD model
# --------------------------------------------------

print("\nTraining SVD model...")

reader = Reader(rating_scale=(1, 5))

svd_data = Dataset.load_from_df(
    train_df[["user_id", "movie_id", "rating"]],
    reader
)

svd_trainset = svd_data.build_full_trainset()

svd_model = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=RANDOM_STATE
)

svd_model.fit(svd_trainset)

print("Evaluating SVD...")

svd_predictions = [
    svd_model.predict(
        int(row.user_id),
        int(row.movie_id)
    ).est
    for row in test_df.itertuples(index=False)
]

# --------------------------------------------------
# 5. CNN model
# --------------------------------------------------

print("\nPreparing CNN model...")

tf.keras.utils.set_random_seed(RANDOM_STATE)

# Create ID mappings using dataset IDs.
# The model learns only from training ratings.
all_user_ids = sorted(ratings["user_id"].unique())
all_movie_ids = sorted(ratings["movie_id"].unique())

user_to_index = {
    user_id: index
    for index, user_id in enumerate(all_user_ids)
}

movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(all_movie_ids)
}

X_user_train = train_df["user_id"].map(
    user_to_index
).to_numpy(dtype=np.int32)

X_movie_train = train_df["movie_id"].map(
    movie_to_index
).to_numpy(dtype=np.int32)

y_train = train_df["rating"].to_numpy(dtype=np.float32)

print("Building CNN architecture...")

embedding_size = 16

user_input = Input(shape=(1,), name="user_input")
movie_input = Input(shape=(1,), name="movie_input")

user_embedding = Embedding(
    input_dim=len(all_user_ids),
    output_dim=embedding_size
)(user_input)

movie_embedding = Embedding(
    input_dim=len(all_movie_ids),
    output_dim=embedding_size
)(movie_input)

user_embedding = Reshape((embedding_size,))(user_embedding)
movie_embedding = Reshape((embedding_size,))(movie_embedding)

user_embedding = Reshape((embedding_size, 1))(user_embedding)
movie_embedding = Reshape((1, embedding_size))(movie_embedding)

interaction = Multiply()([user_embedding, movie_embedding])
interaction = Reshape(
    (embedding_size, embedding_size, 1)
)(interaction)

x = Conv2D(
    32, kernel_size=(3, 3),
    activation="relu", padding="same"
)(interaction)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(
    64, kernel_size=(3, 3),
    activation="relu", padding="same"
)(x)

x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

output = Dense(1, activation="linear")(x)

cnn_model = Model(
    inputs=[user_input, movie_input],
    outputs=output
)

cnn_model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

print("Training CNN. This may take some time...")

cnn_model.fit(
    [X_user_train, X_movie_train],
    y_train,
    epochs=5,
    batch_size=512,
    validation_split=0.1,
    verbose=1
)

print("Evaluating CNN...")

X_user_test = test_df["user_id"].map(
    user_to_index
).to_numpy(dtype=np.int32)

X_movie_test = test_df["movie_id"].map(
    movie_to_index
).to_numpy(dtype=np.int32)

cnn_predictions = cnn_model.predict(
    [X_user_test, X_movie_test],
    batch_size=512,
    verbose=1
).flatten()

cnn_predictions = np.clip(cnn_predictions, 1, 5)

# --------------------------------------------------
# 6. Calculate RMSE and MAE
# --------------------------------------------------

actual = test_df["rating"].to_numpy()

predictions = {
    "KNN": np.array(knn_predictions),
    "SVD": np.array(svd_predictions),
    "CNN": np.array(cnn_predictions)
}

results = []

for model_name, predicted in predictions.items():
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae
    })

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    BASE_DIR / "model_results.csv",
    index=False
)

# --------------------------------------------------
# 7. Save RMSE comparison graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("RMSE Comparison of KNN, SVD and CNN")
plt.xlabel("Models")
plt.ylabel("RMSE (Lower is Better)")
plt.tight_layout()
plt.savefig(BASE_DIR / "model_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# 8. Save MAE comparison graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("MAE Comparison of KNN, SVD and CNN")
plt.xlabel("Models")
plt.ylabel("MAE (Lower is Better)")
plt.tight_layout()
plt.savefig(BASE_DIR / "mae_comparison.png", dpi=300)
plt.close()

# --------------------------------------------------
# 9. Compare top 10 recommendations for one user
# --------------------------------------------------

print("\nGenerating top 10 recommendations...")

demo_user = DEMO_USER_ID

if demo_user not in user_to_index:
    raise ValueError("Demo user ID is not present in the dataset.")

watched_movies = set(
    train_df.loc[
        train_df["user_id"] == demo_user, "movie_id"
    ]
)

candidate_movies = movies[
    ~movies["movie_id"].isin(watched_movies)
].copy()

# KNN recommendations
liked_movies = train_df[
    (train_df["user_id"] == demo_user) &
    (train_df["rating"] >= 4)
]

if liked_movies.empty:
    liked_movies = train_df[
        (train_df["user_id"] == demo_user) &
        (train_df["rating"] >= 3)
    ]

if liked_movies.empty:
    liked_movies = train_df[
        train_df["user_id"] == demo_user
    ]

knn_scores = {}

for row in liked_movies.itertuples(index=False):
    if row.movie_id not in knn_movie_to_index:
        continue

    movie_index = knn_movie_to_index[row.movie_id]
    similarities = item_similarity[movie_index]

    for index, similarity in enumerate(similarities):
        recommended_id = knn_movie_ids[index]

        if recommended_id in watched_movies or similarity <= 0:
            continue

        knn_scores[recommended_id] = (
            knn_scores.get(recommended_id, 0)
            + similarity * row.rating
        )

knn_top = pd.DataFrame(
    list(knn_scores.items()),
    columns=["movie_id", "Score"]
).sort_values("Score", ascending=False).head(10)

knn_top = knn_top.merge(movies, on="movie_id", how="left")
knn_top["Model"] = "KNN"

# SVD recommendations
svd_top = candidate_movies.copy()

svd_top["Score"] = svd_top["movie_id"].apply(
    lambda movie_id: svd_model.predict(
        demo_user, int(movie_id)
    ).est
)

svd_top = svd_top.nlargest(10, "Score")
svd_top["Model"] = "SVD"

# CNN recommendations
# Keep only movies that have a trained CNN embedding index
cnn_top = candidate_movies[
    candidate_movies["movie_id"].isin(movie_to_index)
].copy()

cnn_user_array = np.full(
    len(cnn_top),
    user_to_index[demo_user],
    dtype=np.int32
)

cnn_movie_array = cnn_top["movie_id"].map(
    movie_to_index
).to_numpy(dtype=np.int32)

cnn_top["Score"] = cnn_model.predict(
    [cnn_user_array, cnn_movie_array],
    batch_size=512,
    verbose=0
).flatten()

cnn_top = cnn_top.nlargest(10, "Score")
cnn_top["Model"] = "CNN"

recommendation_results = pd.concat(
    [knn_top, svd_top, cnn_top],
    ignore_index=True
)

recommendation_results = recommendation_results[
    ["Model", "movie_id", "title", "genres", "Score"]
]

recommendation_results.to_csv(
    BASE_DIR / "recommendation_comparison.csv",
    index=False
)

print("\nTop 10 recommendations saved for User ID:", demo_user)
print(recommendation_results.to_string(index=False))

print("\n========== FILES CREATED ==========")
print("model_results.csv")
print("model_comparison.png")
print("mae_comparison.png")
print("recommendation_comparison.csv")
print("\nComparison completed!")
