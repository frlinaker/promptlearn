#!/usr/bin/env python

"""
Example: Using PromptRegressor to generate samples
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

samples = model.sample(20)
print(samples)
