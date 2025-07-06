#!/usr/bin/env python

"""
Example: Using PromptClassifier on the Adult Census dataset with chunked training.
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from promptlearn import PromptClassifier
import logging
import joblib

logging.basicConfig(level=logging.INFO)

# Load dataset from OpenML
adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data
y = adult.target

print(f"Dataset info: {X.shape}, {y.shape}")

# Clean and normalize
X = X.copy()  # ensure safe editing
for col in X.select_dtypes(include="category").columns:
    X[col] = X[col].astype(str).fillna("unknown")
y = (y == ">50K").astype(int)
X = X.astype(str)

# Shuffle and split before chunking
X_shuffled, X_val, y_shuffled, y_val = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,           # optional: maintain class balance
    random_state=42
)

# Truncate dataset to simulate chunked training
CHUNK_SIZE = 300
N_CHUNKS = 8
TRAIN_ROWS = CHUNK_SIZE * N_CHUNKS
VAL_ROWS = 20

# Just use some of the rows to save on LLM costs
X_train = X_shuffled.iloc[:TRAIN_ROWS]
y_train = y_shuffled.iloc[:TRAIN_ROWS]
X_val = X_val.iloc[:VAL_ROWS]
y_val = y_val.iloc[:VAL_ROWS]

# Persist the validation data so that we have the option to evaluate chunk by chunk
joblib.dump({"X_val": X_val, "y_val": y_val}, "examples/saves/val_data.joblib")

# Train the classifier with chunked logic
clf = PromptClassifier(
    verbose=True,
    chunk_threshold=100,
    force_chunking=True,
    max_chunks=N_CHUNKS,
    save_dir="examples/saves"
)

clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"✅ Validation Accuracy: {acc:.4f}")

clf.show_heuristic_evolution()
