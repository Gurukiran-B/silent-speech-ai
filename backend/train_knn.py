import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ==========================================
# Load Dataset
# ==========================================

print("=" * 60)
print("TRAINING KNN V2")
print("=" * 60)

df = pd.read_csv("enhanced_data.csv")

print("\nDataset Loaded Successfully")
print(df.head())

# ==========================================
# Remove Duplicates
# ==========================================

df = df.drop_duplicates()

# ==========================================
# Features
# ==========================================

X = df.drop(columns=["label"])

y = df["label"]

# ==========================================
# Encode Labels
# ==========================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

print("\nClasses")

for i, c in enumerate(encoder.classes_):
    print(i, "->", c)

# ==========================================
# Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ==========================================
# Standardization
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Find Best K
# ==========================================

best_accuracy = 0
best_k = 1
best_model = None

accuracies = []

print("\nSearching Best K...\n")

for k in range(1, 26):

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    accuracies.append(acc)

    print(f"K = {k:2d}   Accuracy = {acc*100:.2f}%")

    if acc > best_accuracy:

        best_accuracy = acc
        best_k = k
        best_model = model

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best K :", best_k)
print("Accuracy :", round(best_accuracy * 100, 2), "%")

# ==========================================
# Final Evaluation
# ==========================================

pred = best_model.predict(X_test)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        pred,
        target_names=encoder.classes_,
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, pred))

# ==========================================
# Save Model
# ==========================================

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nSaved")

print("model.pkl")
print("scaler.pkl")
print("label_encoder.pkl")

# ==========================================
# Plot Accuracy
# ==========================================

plt.figure(figsize=(10,6))

plt.plot(
    range(1,26),
    accuracies,
    marker="o"
)

plt.title("Accuracy vs K")

plt.xlabel("K")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig("accuracy_vs_k.png")

plt.show()

print("\nAccuracy graph saved as accuracy_vs_k.png")