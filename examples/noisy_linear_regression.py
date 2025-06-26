#!/usr/bin/env python

"""
Example: Using PromptRegressor to learn a noisy linear relationship (y = 2x + 3 + noise).
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor
import logging

logging.basicConfig(level=logging.INFO)

# Set a seed for reproducibility
np.random.seed(42)

# Generate training data: y = 2x + 3 + noise
X = np.array([[-1], [0], [1], [2], [3]])
true_y = 2 * X.flatten() + 3
noise = np.random.normal(loc=0.0, scale=0.2, size=len(X))
y = true_y + noise

# Fit the prompt-based regressor
model = PromptRegressor(verbose=True)
model.fit(X, y)

# Predict on test data
X_test = np.array([[4], [5]])
y_pred = model.predict(X_test)

print("Predictions:")
for x_val, y_val in zip(X_test.flatten(), y_pred):
    print(f"x={x_val} → y={y_val:.2f}")

# Compare to expected true values
y_true = 2 * X_test.flatten() + 3
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (vs. true model): {mse:.4f}")
