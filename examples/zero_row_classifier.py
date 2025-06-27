#!/usr/bin/env python

"""
Example: Using PromptClassifier on an empty dataset with headers only.
It is still able to learn the valid prediction function!
"""

import pandas as pd
import logging
from promptlearn import PromptClassifier

logging.basicConfig(level=logging.INFO)

# 1. Define an empty DataFrame with correct column names
X = pd.DataFrame(columns=["country_name"])
y = pd.Series(name="has_blue_in_flag", dtype=int)

# 2. Create and fit the PromptClassifier
clf = PromptClassifier(verbose=True)
clf.fit(X, y)  # This should trigger prompt construction with only headers

# 3. Predict on a new example
new_example = pd.DataFrame([{"country_name": "Japan"}])
prediction = clf.predict(new_example)[0]
print(f"\nDoes the flag of Japan contain blue? {'Yes (incorrect)' if prediction else 'No (correct)'}")
