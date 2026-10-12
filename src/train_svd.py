
from pathlib import Path
import pandas as pd
import joblib

from surprise import Dataset, Reader, SVD

BASE_DIR = Path(__file__).resolve().parent.parent
RATINGS_PATH = BASE_DIR / "data/ml-1m/ratings.dat"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "svd_model.pkl"

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"]
)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    ratings[["UserID", "MovieID", "Rating"]],
    reader
)

trainset = data.build_full_trainset()

model = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42
)

print("Training SVD model. Please wait...")
model.fit(trainset)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("SVD training completed!")
print(f"Model saved at: {MODEL_PATH}")
