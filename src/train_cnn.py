
from src.cnn_recommender import (
    build_cnn_model,
    X_user,
    X_movie,
    y,
    MODEL_PATH
)

print("Building CNN model...")

model = build_cnn_model()

print("Training CNN model. Please wait...")

history = model.fit(
    [X_user, X_movie],
    y,
    epochs=5,
    batch_size=512,
    validation_split=0.1,
    verbose=1
)

model.save(MODEL_PATH)

print("\nCNN training completed!")
print(f"Model saved at: {MODEL_PATH}")
print(f"Final training MAE: {history.history['mae'][-1]:.4f}")
print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")
