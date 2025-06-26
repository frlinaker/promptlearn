#!/usr/bin/env python

"""
Example: Using PromptRegressor to learn a nonlinear function of two variables:
y = 3 * x1^2 + 2 * x2 + 5 + noise
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor
import logging

logging.basicConfig(level=logging.INFO)

# 🔧 Config
N = 50                # Number of training points (increase to improve LLM performance)
NOISE_STD = 0.2       # Standard deviation of added noise

# Seed for reproducibility
np.random.seed(42)

# Generate N evenly spaced x1 and x2 values (randomized pairs)
x1 = np.random.uniform(-2, 2, size=N)
x2 = np.random.uniform(0, 5, size=N)
X = np.column_stack((x1, x2))

# Nonlinear target with noise: y = 3x1² + 2x2 + 5 + noise
y = 3 * x1**2 + 2 * x2 + 5 + np.random.normal(0, NOISE_STD, size=N)

# Fit the prompt-based regressor
model = PromptRegressor(verbose=True)
model.fit(X, y)

# Test on known points
X_test = np.array([
    [1.0, 2.0],     # ~12
    [-1.5, 3.0],    # ~17.75
    [0.0, 4.0]      # ~13.0
])
y_pred = model.predict(X_test)
y_true = 3 * X_test[:, 0]**2 + 2 * X_test[:, 1] + 5

print("\nPredictions:")
for x, pred, true in zip(X_test, y_pred, y_true):
    print(f"x1={x[0]:.2f}, x2={x[1]:.2f} → predicted={pred:.2f}, expected={true:.2f}")

mse = mean_squared_error(y_true, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")

# gpt-4 as of June 2025 is giving a pretty terrible answer:
#   Based on the provided data, the final trained regression function is:
#   target = 14.5 + 2.3*x1 + 3.1*x2 + 1.2*x1^2 + 0.9*x2^2 + 1.5*x1*x2
#   ...
#   Predictions:
#   x1=1.00, x2=2.00 → predicted=26.40, expected=12.00
#   x1=-1.50, x2=3.00 → predicted=32.25, expected=17.75
#   x1=0.00, x2=4.00 → predicted=26.40, expected=13.00
#   Mean Squared Error: 199.0567
