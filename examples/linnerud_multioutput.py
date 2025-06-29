#!/usr/bin/env python

"""
Example: Using scikit-learn's MultiOutputRegressor to wrap PromptRegressor
on the Linnerud dataset (multi-target regression).
"""

import pandas as pd
import logging
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor

# 🔧 Logging config
logging.basicConfig(level=logging.INFO)

# 📥 Load dataset
print("📥 Loading Linnerud dataset...")
data = load_linnerud()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=data.target_names)

print("\n📈 Input features:")
print(X.head())

print("\n🎯 Targets:")
print(y.head())

# 🎓 Create and train the multi-output model
print("\n🧠 Training MultiOutputRegressor with PromptRegressor...")
model = MultiOutputRegressor(PromptRegressor(verbose=True))
model.fit(X, y)

# 📤 Predict on the training set (since it's small)
print("\n🔮 Making predictions on training data...")
y_pred = model.predict(X)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)

# 📊 Display predictions
print("\n📋 Predictions:")
print(y_pred_df.round(2))

# 🧪 Evaluate each target separately
print("\n📊 Mean Squared Error per target:")
for col in y.columns:
    mse = mean_squared_error(y[col], y_pred_df[col])
    print(f"  {col}: {mse:.4f}")
