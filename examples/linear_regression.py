#!/usr/bin/env python

"""
Example: Using PromptRegressor to learn a simple linear equation (y = 2x + 3).
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor
import logging

logging.basicConfig(level=logging.INFO)

# Create training data: y = 2x + 3
X = np.array([[-1], [0], [1], [2], [3]])
y = np.array([1, 3, 5, 7, 9])  # 2x + 3

# Fit the prompt-based regressor
model = PromptRegressor(verbose=True)
model.fit(X, y)

# Predict on test data
X_test = np.array([[4], [5]])
y_pred = model.predict(X_test)

print("Predictions:")
for x_val, y_val in zip(X_test.flatten(), y_pred):
    print(f"x={x_val} → y={y_val:.2f}")

# Optional: Evaluate against ground truth
y_true = 2 * X_test.flatten() + 3
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
