# Credit Card Fraud Detection using Random Forest
Project Overview  :

Credit card fraud is a critical real-world problem where fraudulent transactions are extremely rare compared to legitimate ones.
This project builds a production-ready machine learning pipeline to detect fraudulent credit card transactions using a Random Forest Classifier, with a strong focus on clean code structure, reproducibility, and deployment readiness.

The project follows industry best practices by separating:

    - Experimentation (notebooks)
    - Data preprocessing
    - Model training
    - Evaluation
    - Prediction (inference)

---

## 📌 Problem Statement

Given transaction data, predict whether a transaction is fraudulent (1) or legitimate (0).

Key challenges:

    - Highly imbalanced dataset
    - False negatives (missed frauds) are costly
    - Model must generalize well to unseen data
---
## 💡Solution Approach

    - Used Random Forest Classifier with class_weight="balanced"
    - Focused on Recall, Precision, F1-Score, and ROC-AUC
    - Built a modular ML pipeline ready for deployment
    - Ensured reproducibility with config-driven design
---

## 📂 Project Structure
```text 
credit-card-fraud-detection/
│
├── data/
│   ├── raw/              # Original dataset 
│   └── processed/        # Train-test splits 
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
│
├── src/
│   ├── config.py         # Central configuration
│   ├── preprocess.py    # Data preprocessing & splitting
│   ├── train.py          # Model training
│   ├── evaluate.py       # Model evaluation
│   ├── predict.py        # Inference logic
│   └── __init__.py
│
├── models/               # Saved model & scaler 
│
├── .gitignore
├── README.md
└── requirements.txt
```
---
## 📊 Model Performance (Test Set)

    - Metric	Fraud Class (1)
    - Precision	96.05%
    - Recall	74.49%
    - F1-Score	83.91%
    - ROC-AUC	0.9529
---
## Interpretation:

- High precision → Very few false fraud alerts

- Strong ROC-A reminding model discrimination capability

- Suitable baseline for real-world fraud detection systems
---

## ⚙️ How to Run the Project

**1 Clone Repository**

     git clone <my-repo-url>
     cd credit-card-fraud-detection

**2 Install Dependencies**
    
     pip install -r requirements.txt

**3 Preprocess Data**

     python -m src.preprocess

**4 Train Model**

     python -m src.train

**5 Evaluate Model**

     python -m src.evaluate

**6 Make Predictions**

     python -m src.predict

**Sample Prediction Output**

    {
       "prediction": 0,
       "fraud_probability": 0.0123
    }

---
## 🚀 Key Highlights

✅ Modular & scalable ML pipeline

✅ No hard-coded paths (config-driven)

✅ Handles class imbalance properly

✅ Deployment-ready inference logic

✅ Clean Git & project hygiene

---

## 🔮 Future Improvements
    - Hyperparameter tuning (GridSearch / Bayesian Optimization)
    - Threshold tuning for higher recall
    - FastAPI / Flask REST API
    - MLflow for experiment tracking
    - Dockerization & CI/CD pipeline
---

## 👹Author

    Aspiring Machine Learning / Data Scientist
                                - Mad_titaN 


⭐ If you like this project, consider giving it a star!
