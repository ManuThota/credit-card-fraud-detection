import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import (
    RAW_DATA_FILE,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    SCALER_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    PROCESSED_DATA_DIR,
    MODELS_DIR
)

def preprocess_data():
    """
    Loads Raw data, preprocesses it, and saves train-test splits.
    """

    # Load dataset
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Separate features and target
    x = df.drop(columns=["Class"])
    y = df["Class"]

    # Scale 'Amount' column
    scaler = StandardScaler()
    x['Amount'] = scaler.fit_transform(x[['Amount']])

    # Train-Test split with statification
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    #Create directories if they don't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed data
    x_train.to_csv(X_TRAIN_PATH, index=False)
    x_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    # Save the Scaler
    joblib.dump(scaler, SCALER_PATH)

    print("Data preprocessing completed and files saved.")
    print(f"Train shape : {x_train.shape}")
    print(f"Test shape  : {x_test.shape}")

if __name__ == "__main__":
    preprocess_data()
