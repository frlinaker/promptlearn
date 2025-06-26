#!/usr/bin/env python

"""
Example: Using PromptClassifier to learn XOR logic from binary input data.
"""

import logging
import numpy as np
from sklearn.metrics import accuracy_score
from promptlearn import PromptClassifier

logging.basicConfig(level=logging.INFO)

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])  # XOR outputs

# Create and train the prompt-based classifier
clf = PromptClassifier(verbose=True)
clf.fit(X, y)

# Predict on the full dataset
y_pred = clf.predict(X)

# Show predictions
print("Predictions:", y_pred)
print("True labels:", y.tolist())

# Evaluate accuracy
acc = accuracy_score(y, y_pred)
print(f"Accuracy: {acc:.2f}")

# Optional: Predict on a new input
x_new = np.array([[1, 1]])
y_new = clf.predict(x_new)
print(f"Predicted class for input {x_new.tolist()[0]}: {y_new[0]}")
print("(Expected output is 0 for XOR)")
