#!/usr/bin/env python

"""
Benchmark: Compare PromptClassifier (various LLMs) vs baseline classifiers on a tabular dataset.
"""

import pandas as pd
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
import logging
from promptlearn import PromptClassifier

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

print("📦 Loading Adult dataset...")
data = fetch_openml("adult", version=2, as_frame=True)

# For promptlearn (LLMs): use string format
X_llm = data.data.astype(str).fillna("unknown")
y_raw = (data.target == ">50K").astype(int)

# For baseline models: one-hot encode
X_encoded = pd.get_dummies(data.data)

print(f"✅ Dataset loaded: {X_llm.shape[0]} rows")

# Split both versions the same way
print("🔀 Shuffling and splitting...")
X_train_llm_full, X_val_llm, y_train_full, y_val = train_test_split(
    X_llm, y_raw, test_size=0.3, stratify=y_raw, random_state=42
)

X_train_encoded_full, X_val_encoded, _, _ = train_test_split(
    X_encoded, y_raw, test_size=0.3, stratify=y_raw, random_state=42
)

# Truncate training set for fairness
X_train_llm = X_train_llm_full.iloc[:2400]
X_train_encoded = X_train_encoded_full.iloc[:2400]
y_train = y_train_full.iloc[:2400]

# Truncate validation set to limit LLM costs
VAL_ROWS = 200
X_val_llm = X_val_llm.iloc[:VAL_ROWS]
X_val_encoded = X_val_encoded.iloc[:VAL_ROWS]
y_val = y_val.iloc[:VAL_ROWS]

print(f"✅ Training on {len(X_train_llm)} rows, validating on {len(X_val_llm)} rows")

# Define baseline models
baselines = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "decision_tree": DecisionTreeClassifier(max_depth=3),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=20, max_depth=4),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=30, max_depth=3),
}

# Define LLM models
llm_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "o3-mini", "o4-mini"]
promptlearners = {
    f"promptlearn_{model}": PromptClassifier(
        model=model, verbose=False, chunk_threshold=300, force_chunking=True, max_chunks=8
    ) for model in llm_models
}

# Combine models
all_models = {**baselines, **promptlearners}

results = []
print("🚀 Beginning benchmark...\n")

for name, clf in all_models.items():
    print(f"▶️ Training: {name}")
    start = time.time()

    if "promptlearn" in name:
        clf.fit(X_train_llm, y_train)
    else:
        clf.fit(X_train_encoded, y_train)

    fit_time = time.time() - start
    print(f"⏱️ Fit time: {fit_time:.2f}s")

    print(f"🔮 Predicting: {name}")
    start = time.time()
    if "promptlearn" in name:
        y_pred = clf.predict(X_val_llm)
    else:
        y_pred = clf.predict(X_val_encoded)
    pred_time = time.time() - start
    acc = accuracy_score(y_val, y_pred)
    print(f"✅ Accuracy ({name}): {acc:.4f} | ⏱️ Predict time: {pred_time:.2f}s\n")

    results.append({
        "model": name,
        "accuracy": acc,
        "fit_time_sec": fit_time,
        "predict_time_sec": pred_time
    })

# Show final results
df_results = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print("\n=== Final Results ===")
print(df_results.to_string(index=False))
