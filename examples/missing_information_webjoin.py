#!/usr/bin/env python

"""
Example: Using PromptClassifier to predict if a country's flag contains blue based on its name.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

from promptlearn import PromptClassifier

logging.basicConfig(level=logging.INFO)

# 1. Define the dataset
data = [
    {"country_name": "Sweden", "has_blue_in_flag": 1},
    {"country_name": "Australia", "has_blue_in_flag": 1},
    {"country_name": "United States", "has_blue_in_flag": 1},
    {"country_name": "New Zealand", "has_blue_in_flag": 1},
    {"country_name": "Norway", "has_blue_in_flag": 1},
    {"country_name": "Japan", "has_blue_in_flag": 0},
    {"country_name": "India", "has_blue_in_flag": 0},
    {"country_name": "Germany", "has_blue_in_flag": 0},
    {"country_name": "Italy", "has_blue_in_flag": 0},
    {"country_name": "Mexico", "has_blue_in_flag": 0},
]

df = pd.DataFrame(data)
X = df[["country_name"]]
y = df["has_blue_in_flag"]

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train the PromptClassifier
clf = PromptClassifier(verbose=True)
clf.fit(X_train, y_train)

# 4. Predict on test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# 5. Predict on a new example
new_example = pd.DataFrame([{"country_name": "France"}])  # 🇫🇷 has blue
prediction = clf.predict(new_example)[0]
print(f"\nDoes the flag of France contain blue? {'Yes 🟦' if prediction else 'No ❌'}")
