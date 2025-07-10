#!/usr/bin/env python

"""
Example: Using PromptRegressor to learn a nonlinear function of two variables:
y = 3 * length^2 + 2 * volume + 5 + noise
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor
import logging

logging.basicConfig(level=logging.INFO)

# 🔧 Config
N = 50  # Number of training points
NOISE_STD = 0.2  # Standard deviation of added noise

# Seed for reproducibility
np.random.seed(42)

# Generate random features
length = np.random.uniform(-2, 2, size=N)
volume = np.random.uniform(0, 5, size=N)

# Construct DataFrame with custom column names
X = pd.DataFrame({"length": length, "volume": volume})

# Target: y = 3 * length^2 + 2 * volume + 5 + noise
y = 3 * length**2 + 2 * volume + 5 + np.random.normal(0, NOISE_STD, size=N)
y_series = pd.Series(y, name="output")

# Fit model
model = PromptRegressor(verbose=True)
model.fit(X, y_series)

# Test on known points
X_test = pd.DataFrame(
    [
        {"length": 1.0, "volume": 2.0},  # ~12
        {"length": -1.5, "volume": 3.0},  # ~17.75
        {"length": 0.0, "volume": 4.0},  # ~13
    ]
)
y_pred = model.predict(X_test)
y_true = 3 * X_test["length"] ** 2 + 2 * X_test["volume"] + 5

print("\nPredictions:")
for (_, row), pred, true in zip(X_test.iterrows(), y_pred, y_true):
    print(
        f"length={row['length']:.2f}, volume={row['volume']:.2f} → predicted={pred:.2f}, expected={true:.2f}"
    )

mse = mean_squared_error(y_true, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")
