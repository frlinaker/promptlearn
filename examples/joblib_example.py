#!/usr/bin/env python

"""
Example: Using Joblib on a trained PromptRegressor
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor
from pathlib import Path
import logging
import joblib

logging.basicConfig(level=logging.INFO)

# Create training data: y = 2x + 3
X = np.array([[-1], [0], [1], [2], [3]])
y = np.array([1, 3, 5, 7, 9])  # 2x + 3

# Fit the prompt-based regressor
model = PromptRegressor(verbose=True)
model.fit(X, y)

# Save to disk
model_path = Path(__file__).parent / "preg.joblib"
joblib.dump(model, model_path)

# Now load it back and see if it works (using a different variable to be sure
loaded_model = joblib.load(model_path)

# Predict on test data
X_test = np.array([[4], [5]])
y_pred = loaded_model.predict(X_test)

print("Predictions from loaded model:")
for x_val, y_val in zip(X_test.flatten(), y_pred):
    print(f"x={x_val} → y={y_val:.2f}")

# Optional: Evaluate against ground truth
y_true = 2 * X_test.flatten() + 3
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error for loaded model: {mse:.4f}")
