# ============================================================
# CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ------------------------------------------------------------
# 1. CREATE DATASET
# ------------------------------------------------------------

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)

np.random.seed(42)

# Number of transactions
n = 10000

# Generate transaction features
amount = np.random.exponential(scale=100, size=n)
time = np.random.randint(0, 172800, size=n)

# Customer-related features
customer_age = np.random.randint(18, 75, size=n)
distance_from_home = np.random.exponential(scale=20, size=n)
distance_from_last_transaction = np.random.exponential(scale=10, size=n)

# Merchant and transaction features
merchant_risk = np.random.uniform(0, 1, size=n)
transaction_frequency = np.random.poisson(lam=5, size=n)
online_transaction = np.random.randint(0, 2, size=n)

# Create fraud probability
fraud_probability = (
    0.02
    + 0.15 * (amount > 300)
    + 0.25 * (distance_from_home > 50)
    + 0.20 * (distance_from_last_transaction > 30)
    + 0.25 * (merchant_risk > 0.8)
    + 0.15 * online_transaction
)

# Add randomness
fraud_probability += np.random.normal(0, 0.05, n)

# Keep probability between 0 and 1
fraud_probability = np.clip(fraud_probability, 0, 1)

# Generate fraud labels
fraud = (np.random.random(n) < fraud_probability).astype(int)

# Create DataFrame
data = pd.DataFrame({
    "Time": time,
    "Amount": amount,
    "Customer_Age": customer_age,
    "Distance_From_Home": distance_from_home,
    "Distance_From_Last_Transaction": distance_from_last_transaction,
    "Merchant_Risk": merchant_risk,
    "Transaction_Frequency": transaction_frequency,
    "Online_Transaction": online_transaction,
    "Fraud": fraud
})

print("\nDataset created successfully!")
print("\nFirst 5 records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

# ------------------------------------------------------------
# 2. CHECK DATA
# ------------------------------------------------------------

print("\nMissing values:")
print(data.isnull().sum())

print("\nFraud distribution:")
print(data["Fraud"].value_counts())

# ------------------------------------------------------------
# 3. VISUALIZE FRAUD DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(x="Fraud", data=data)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Genuine", "Fraud"])

plt.show()

# ------------------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = data.drop("Fraud", axis=1)
y = data["Fraud"]

# ------------------------------------------------------------
# 5. TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ------------------------------------------------------------
# 6. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# 7. LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

logistic_prediction = logistic_model.predict(X_test_scaled)

# Metrics
logistic_accuracy = accuracy_score(y_test, logistic_prediction)
logistic_precision = precision_score(
    y_test, logistic_prediction, zero_division=0
)
logistic_recall = recall_score(
    y_test, logistic_prediction, zero_division=0
)
logistic_f1 = f1_score(
    y_test, logistic_prediction, zero_division=0
)

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1 Score :", round(logistic_f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    logistic_prediction,
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

# ------------------------------------------------------------
# 8. LOGISTIC REGRESSION CONFUSION MATRIX
# ------------------------------------------------------------

cm_logistic = confusion_matrix(y_test, logistic_prediction)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"]
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ------------------------------------------------------------
# 9. RANDOM FOREST
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

random_forest.fit(X_train, y_train)

rf_prediction = random_forest.predict(X_test)

# Metrics
rf_accuracy = accuracy_score(y_test, rf_prediction)
rf_precision = precision_score(
    y_test, rf_prediction, zero_division=0
)
rf_recall = recall_score(
    y_test, rf_prediction, zero_division=0
)
rf_f1 = f1_score(
    y_test, rf_prediction, zero_division=0
)

print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1 Score :", round(rf_f1, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_prediction,
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

# ------------------------------------------------------------
# 10. RANDOM FOREST CONFUSION MATRIX
# ------------------------------------------------------------

cm_rf = confusion_matrix(y_test, rf_prediction)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ------------------------------------------------------------
# 11. MODEL COMPARISON
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],
    "Precision": [
        logistic_precision,
        rf_precision
    ],
    "Recall": [
        logistic_recall,
        rf_recall
    ],
    "F1 Score": [
        logistic_f1,
        rf_f1
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.round(4))

# ------------------------------------------------------------
# 12. MODEL COMPARISON GRAPH
# ------------------------------------------------------------

comparison.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 13. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(9, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 14. ROC-AUC
# ------------------------------------------------------------

logistic_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

rf_probability = random_forest.predict_proba(
    X_test
)[:, 1]

logistic_auc = roc_auc_score(
    y_test,
    logistic_probability
)

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)

print("\nROC-AUC Scores:")
print("Logistic Regression:", round(logistic_auc, 4))
print("Random Forest      :", round(rf_auc, 4))

# ------------------------------------------------------------
# 15. SAVE RANDOM FOREST MODEL
# ------------------------------------------------------------

joblib.dump(
    random_forest,
    "credit_card_fraud_model.pkl"
)

joblib.dump(
    scaler,
    "credit_card_scaler.pkl"
)

print("\nModel saved successfully!")
print("File: credit_card_fraud_model.pkl")

# ------------------------------------------------------------
# 16. TEST A NEW TRANSACTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NEW TRANSACTION PREDICTION")
print("=" * 60)

new_transaction = pd.DataFrame({
    "Time": [50000],
    "Amount": [450],
    "Customer_Age": [35],
    "Distance_From_Home": [80],
    "Distance_From_Last_Transaction": [40],
    "Merchant_Risk": [0.90],
    "Transaction_Frequency": [2],
    "Online_Transaction": [1]
})

prediction = random_forest.predict(new_transaction)[0]

probability = random_forest.predict_proba(
    new_transaction
)[0][1]

print("\nTransaction Details:")
print(new_transaction)

print("\nFraud Probability:",
      round(probability * 100, 2), "%")

if prediction == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: GENUINE TRANSACTION")

print("\nProgram completed successfully!")