#!/usr/bin/env python

"""
Example: Using PromptRegressor to learn a simple riddle
https://www.quora.com/A-chicken-was-given-7-an-ant-was-given-21-a-spider-was-given-28-How-much-money-was-the-dog-given

Good luck to traditional ML models
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor
import logging

logging.basicConfig(level=logging.INFO)

# Create training data
data = pd.DataFrame(
    {
        "animal": ["chicken", "ant", "spider"],
        "money": [7, 21, 28],
    }
)
X = data[["animal"]]
y = data["money"]

# Fit the prompt-based regressor
model = PromptRegressor(model="o4-mini", verbose=True)
model.fit(X, y)

# Predict on test data
X_test = pd.DataFrame({"animal": ["dog"]})
y_pred = model.predict(X_test)

print("Predictions:")
for x_val, y_val in zip(X_test["animal"].values, y_pred):
    print(f"x={x_val} → y={y_val:.2f}")

# Evaluate against ground truth
y_true = [14]
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# output from running this with o4-mini:

# # First pass:
# def predict(animal=None):
#     # Mapping of known animals to money values
#     mapping = {
#         'chicken': 7.0,
#         'ant': 21.0,
#         'spider': 28.0
#     }
#     # Coerce animal to string for lookup
#     try:
#         key = str(animal)
#     except:
#         return 0.0
#     # Return mapped value or default 0.0
#     return float(mapping.get(key, 0.0))

# # Second pass:
# def predict(animal=None):
#     # Extended mapping of known animals to money values (price = number_of_legs * 3.5)
#     mapping = {
#         # Original examples
#         'chicken': 7.0,      # 2 legs * 3.5
#         'ant': 21.0,         # 6 legs * 3.5
#         'spider': 28.0,      # 8 legs * 3.5

#         # Insects (6 legs)
#         'bee': 21.0,
#         'butterfly': 21.0,
#         'fly': 21.0,
#         'cockroach': 21.0,
#         'grasshopper': 21.0,
#         'mosquito': 21.0,
#         'ladybug': 21.0,
#         'dragonfly': 21.0,

#         # Arachnids (8 legs)
#         'scorpion': 28.0,
#         'tarantula': 28.0,
#         'tick': 28.0,
#         'mite': 28.0,

#         # Birds (2 legs)
#         'duck': 7.0,
#         'turkey': 7.0,
#         'goose': 7.0,
#         'robin': 7.0,
#         'sparrow': 7.0,
#         'eagle': 7.0,
#         'hawk': 7.0,
#         'owl': 7.0,
#         'penguin': 7.0,
#         'flamingo': 7.0,
#         'peacock': 7.0,
#         'crow': 7.0,
#         'pigeon': 7.0,

#         # Mammals (4 legs unless noted)
#         'dog': 14.0,
#         'cat': 14.0,
#         'cow': 14.0,
#         'sheep': 14.0,
#         'goat': 14.0,
#         'horse': 14.0,
#         'pig': 14.0,
#         'deer': 14.0,
#         'rabbit': 14.0,
#         'mouse': 14.0,
#         'rat': 14.0,
#         'lion': 14.0,
#         'tiger': 14.0,
#         'bear': 14.0,
#         'elephant': 14.0,
#         'giraffe': 14.0,
#         'zebra': 14.0,
#         'wolf': 14.0,
#         'fox': 14.0,
#         'donkey': 14.0,
#         'buffalo': 14.0,
#         'moose': 14.0,

#         # Reptiles & Amphibians (typically 4 legs)
#         'lizard': 14.0,
#         'turtle': 14.0,
#         'crocodile': 14.0,
#         'alligator': 14.0,
#         'frog': 14.0,
#         'toad': 14.0,
#         'salamander': 14.0,

#         # Aquatic & other (0 legs, or special cases)
#         'fish': 0.0,
#         'shark': 0.0,
#         'whale': 0.0,
#         'dolphin': 0.0,
#         'octopus': 28.0,      # 8 arms * 3.5
#         'squid': 28.0,
#         'crab': 35.0,         # 10 legs * 3.5
#         'lobster': 35.0,
#         'shrimp': 35.0,
#         'snail': 0.0,
#         'worm': 0.0,
#         'jellyfish': 0.0,
#     }

#     # Coerce animal identifier to string for lookup
#     try:
#         key = str(animal)
#     except Exception:
#         return 0.0

#     # Return mapped value or default 0.0
#     return float(mapping.get(key, 0.0))

# Predictions:
# x=dog → y=14.00
# Mean Squared Error: 0.0000
