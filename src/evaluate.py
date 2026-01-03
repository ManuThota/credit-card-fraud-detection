import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.config import (
    X_TEST_PATH,
    Y_TEST_PATH,
    MODEL_PATH
)


def evaluate_model():
    """
    Evaluates trained model on test data.
    """

    # Load test data
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n ROC-AUC Score: {roc_auc:.4f}")


if __name__ == "__main__":
    evaluate_model()
