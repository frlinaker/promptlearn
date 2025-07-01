#!/usr/bin/env python

"""
Example: Using PromptClassifier on the Titanic dataset with optional chunked training.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from promptlearn import PromptClassifier
import logging

logging.basicConfig(level=logging.INFO)

# Load Titanic dataset from seaborn (or other CSV source if needed)
try:
    import seaborn as sns
    df = sns.load_dataset("titanic")
except:
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Show dataset information
print(df.info())

# Drop columns with too many missing values or that leak the target
df = df.drop(columns=["deck", "embark_town", "alive", "who", "adult_male", "class", "alone"], errors="ignore")
df = df.dropna(subset=["Survived"])

# Select features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Fill missing values (simple preprocessing)
X = X.fillna("unknown")

# Encode categorical columns as strings (LLM-friendly)
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype(str)

# Split into train/validation
CHUNK_ROWS = 300
N_CHUNKS = 2
TRAIN_ROWS = CHUNK_ROWS * N_CHUNKS

X_train = X.iloc[:TRAIN_ROWS]
y_train = y.iloc[:TRAIN_ROWS]

X_val = X.iloc[TRAIN_ROWS:]
y_val = y.iloc[TRAIN_ROWS:]

# Initialize classifier with chunking enabled
clf = PromptClassifier(
    verbose=True,
    chunk_threshold=100,   # small threshold to always trigger chunking here
    max_chunks=N_CHUNKS,
    force_chunking=True
)

# Train on a limited number of chunks
clf.fit(X_train, y_train)

clf.show_heuristic_evolution()

# Evaluate on held-out data
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"✅ Validation Accuracy: {acc:.4f}")

clf.show_heuristic_evolution()
print(classification_report(y_val, y_pred))
