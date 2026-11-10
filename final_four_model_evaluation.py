
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from surprise import (
    Dataset, Reader, KNNBasic, SVD, NMF, CoClustering
)

BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "data/ml-1m/ratings.dat"

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

ratings = ratings[["user_id", "movie_id", "rating"]]

# Same split used during SVD tuning
train_df, test_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=123
)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(train_df, reader)
trainset = data.build_full_trainset()

testset = list(
    test_df.itertuples(index=False, name=None)
)

# Four models with tuned SVD
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
        n_factors=100,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.05,
        random_state=42
    ),

    "NMF": NMF(
        n_factors=20,
        n_epochs=30,
        random_state=42
    ),

    "CoClustering": CoClustering(
        n_cltr_u=3,
        n_cltr_i=3,
        n_epochs=20,
        random_state=42
    )
}

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(trainset)

    predictions = model.test(testset)

    actual = [p.r_ui for p in predictions]
    predicted = [p.est for p in predictions]

    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    mae = mean_absolute_error(actual, predicted)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    })

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Save the final SVD model separately
    if name == "SVD":
        model_dir = BASE_DIR / "models"
        model_dir.mkdir(exist_ok=True)

        joblib.dump(
            model,
            model_dir / "svd_model_tuned.pkl"
        )

results_df = pd.DataFrame(results)

results_df.to_csv(
    BASE_DIR / "final_four_model_results.csv",
    index=False
)

print("\nFINAL RESULTS")
print(results_df.to_string(index=False))

best = results_df.loc[results_df["RMSE"].idxmin()]

print("\nBEST MODEL:", best["Model"])
print("BEST RMSE:", round(best["RMSE"], 4))

# Combined comparison graph
results_df.set_index("Model")[
    ["MSE", "RMSE", "MAE"]
].plot(kind="bar", figsize=(10, 6))

plt.title("Final Four Model Comparison")
plt.ylabel("Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("final_model_comparison.png", dpi=300)
plt.close()

# RMSE graph
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Final RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("final_rmse_comparison.png", dpi=300)
plt.close()

# MAE graph
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.title("Final MAE Comparison")
plt.ylabel("MAE")
plt.tight_layout()
plt.savefig("final_mae_comparison.png", dpi=300)
plt.close()

print("\nAll final evaluation files generated!")
