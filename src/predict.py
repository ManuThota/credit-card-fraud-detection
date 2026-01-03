import pandas as pd
import joblib

from src.config import (
    MODEL_PATH,
    SCALER_PATH
)


def predict(transaction_data: dict):
    """
    Predicts whether a transaction is fraudulent.
    
    Args:
        transaction_data (dict): Feature values of a transaction
    
    Returns:
        dict: prediction and probability
    """

    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Convert input to DataFrame
    df = pd.DataFrame([transaction_data])

    # Scale Amount column
    df["Amount"] = scaler.transform(df[["Amount"]])

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
    "prediction": int(prediction),
    "fraud_probability": float(round(probability, 4))
}



if __name__ == "__main__":
    # Example transaction (dummy values)
    sample_transaction = {
    "Time": 34,
    "V1": -0.291540245,
    "V2": 0.445575314,
    "V3": 1.249752116,
    "V4": -1.73573589,
    "V5": 0.085755559,
    "V6": -0.121924299,
    "V7": 0.407715857,
    "V8": 0.095309378,
    "V9": 0.815902287,
    "V10": -1.491188012,
    "V11": -0.84619138,
    "V12": 0.05653255,
    "V13": -0.05895353,
    "V14": 0.151922602,
    "V15": 1.982594882,
    "V16": -0.443295054,
    "V17": -0.318251412,
    "V18": 0.064787167,
    "V19": 0.613505272,
    "V20": -0.033522345,
    "V21": -0.064906147,
    "V22": -0.120448982,
    "V23": -0.156525634,
    "V24": -0.800212851,
    "V25": -0.000620392,
    "V26": -0.835203307,
    "V27": 0.131001421,
    "V28": 0.06289627,
    "Amount": 18.95
}


    result = predict(sample_transaction)
    print(result)
