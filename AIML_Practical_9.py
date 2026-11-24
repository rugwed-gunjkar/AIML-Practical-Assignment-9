# AIML Practical Assignment No. 9
# Title: Bayesian Classifier on Iris Dataset

# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

# Step 2: Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Iris Dataset Loaded Successfully")
print("Dataset Shape:", X.shape)

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)

# Step 4: Create and train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

print("\nGaussian Naive Bayes Model Trained Successfully")

# Step 5: Make predictions
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

# Step 6: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")

# Step 7: Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 8: Calculate precision and recall
precision = precision_score(
    y_test, y_pred, average='macro'
)

recall = recall_score(
    y_test, y_pred, average='macro'
)

print("\nPrecision:", precision)
print("Recall:", recall)

# Step 9: Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# Step 10: Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Gaussian Naive Bayes")
plt.show()