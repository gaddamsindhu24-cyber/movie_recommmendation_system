
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from surprise import (
    Dataset,
    Reader,
    KNNBasic,
    SVD,
    NMF,
    CoClustering
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "data/ml-1m/ratings.dat"

RESULTS_PATH = BASE_DIR / "four_model_results.csv"
COMPARISON_PATH = BASE_DIR / "four_model_comparison.png"
RMSE_PATH = BASE_DIR / "four_model_rmse.png"
MAE_PATH = BASE_DIR / "four_model_mae.png"

RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load MovieLens 1M dataset
# --------------------------------------------------

print("Loading MovieLens 1M dataset...")

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

ratings = ratings[["user_id", "movie_id", "rating"]]

print(f"Total ratings: {len(ratings)}")

# --------------------------------------------------
# 3. Create one common 80:20 split
# --------------------------------------------------

train_df, test_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Training ratings: {len(train_df)}")
print(f"Testing ratings: {len(test_df)}")

# --------------------------------------------------
# 4. Prepare Surprise training data
# --------------------------------------------------

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    train_df[["user_id", "movie_id", "rating"]],
    reader
)

trainset = data.build_full_trainset()

testset = list(
    test_df[["user_id", "movie_id", "rating"]]
    .itertuples(index=False, name=None)
)

# --------------------------------------------------
# 5. Define four models
# --------------------------------------------------

models = {
    "KNN": KNNBasic(
        k=40,
        min_k=1,
        sim_options={
            "name": "cosine",
            "user_based": False
        },
        verbose=False
    ),

    "SVD": SVD(
        n_factors=50,
        n_epochs=20,
        random_state=RANDOM_STATE
    ),

    "NMF": NMF(
        n_factors=20,
        n_epochs=30,
        random_state=RANDOM_STATE
    ),

    "CoClustering": CoClustering(
        n_cltr_u=3,
        n_cltr_i=3,
        n_epochs=20,
        random_state=RANDOM_STATE
    )
}

# --------------------------------------------------
# 6. Train and evaluate models
# --------------------------------------------------

results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {name} model...")
    print("=" * 50)

    model.fit(trainset)

    predictions = model.test(testset)

    actual = [pred.r_ui for pred in predictions]
    predicted = [pred.est for pred in predictions]

    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    mae = mean_absolute_error(actual, predicted)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    })

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(RESULTS_PATH, index=False)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 8. Find the best model by RMSE
# --------------------------------------------------

best_model = results_df.loc[
    results_df["RMSE"].idxmin()
]

print("\nBest model based on RMSE:")
print(best_model["Model"])

# --------------------------------------------------
# 9. Generate comparison graph
# --------------------------------------------------

plt.figure(figsize=(10, 6))

results_df.set_index("Model")[
    ["MSE", "RMSE", "MAE"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Four Model Performance Comparison")
plt.ylabel("Error Value")
plt.xlabel("Recommendation Models")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(COMPARISON_PATH, dpi=300)
plt.close()

# --------------------------------------------------
# 10. Generate RMSE graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(results_df["Model"], results_df["RMSE"])

plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.xlabel("Models")
plt.tight_layout()
plt.savefig(RMSE_PATH, dpi=300)
plt.close()

# --------------------------------------------------
# 11. Generate MAE graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(results_df["Model"], results_df["MAE"])

plt.title("MAE Comparison")
plt.ylabel("MAE")
plt.xlabel("Models")
plt.tight_layout()
plt.savefig(MAE_PATH, dpi=300)
plt.close()

print("\nAll comparison graphs generated successfully!")

print(f"Results: {RESULTS_PATH}")
print(f"Comparison graph: {COMPARISON_PATH}")
print(f"RMSE graph: {RMSE_PATH}")
print(f"MAE graph: {MAE_PATH}")
