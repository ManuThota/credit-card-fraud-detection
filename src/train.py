import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_FOREST_PARAMS
    )

def train_model():
    """
    Trains RandomForest model and saves it
    """

    # Load processed training data
    x_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    # Initialize Model
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

    # Train model
    model.fit(x_train, y_train)

    # Create model directory if not exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Traines model
    joblib.dump(model, MODEL_PATH)

    print("Model Training completed successfully.")
    print(f"Trained model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()