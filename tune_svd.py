import pandas as pd

from sklearn.model_selection import train_test_split

from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

# Load dataset
ratings = pd.read_csv(
    "data/ml-1m/ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

ratings = ratings[["user_id", "movie_id", "rating"]]

# Reserve a fresh final test set
train_df, test_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=123
)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    train_df,
    reader
)

# Hyperparameter combinations
param_grid = {
    "n_factors": [50, 100],
    "n_epochs": [20, 30],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.02, 0.05]
}

print("Starting SVD hyperparameter tuning...")

grid_search = GridSearchCV(
    SVD,
    param_grid,
    measures=["rmse", "mae"],
    cv=3,
    joblib_verbose=1
)

grid_search.fit(data)

print("\nBEST SVD PARAMETERS")
print(grid_search.best_params["rmse"])

print("\nBEST CROSS-VALIDATION RMSE")
print(grid_search.best_score["rmse"])

print("\nBEST CROSS-VALIDATION MAE")
print(grid_search.best_score["mae"])

print("\nSVD tuning completed successfully!")
