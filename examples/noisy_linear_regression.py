#!/usr/bin/env python

"""
Example: Using PromptRegressor to learn a noisy linear relationship (y = 2x + 3 + noise) from more data points.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor
import logging

logging.basicConfig(level=logging.INFO)

# Configuration
N = 20  # Number of data points
NOISE_STD = 0.2

# Set seed for reproducibility
np.random.seed(42)

# Generate training data
X = np.linspace(-3, 6, N).reshape(-1, 1)
true_y = 2 * X.flatten() + 3
noise = np.random.normal(0.0, NOISE_STD, size=N)
y = true_y + noise

# Fit the prompt-based regressor
model = PromptRegressor(verbose=True)
model.fit(X, y)

# Predict on test data
X_test = np.array([[4], [5], [6]])
y_pred = model.predict(X_test)

print("\nPredictions:")
for x_val, y_val in zip(X_test.flatten(), y_pred):
    print(f"x={x_val:.1f} → y={y_val:.2f}")

# Evaluate accuracy vs ground truth
y_true = 2 * X_test.flatten() + 3
mse = mean_squared_error(y_true, y_pred)
print(f"\nMean Squared Error (vs. true model): {mse:.4f}")
