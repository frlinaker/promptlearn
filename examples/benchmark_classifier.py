#!/usr/bin/env python

# example:
# python examples/benchmark_classifier.py --train examples/data/mammal_train.csv --val examples/data/mammal_val.csv --target is_mammal --val_rows 100

import argparse
import pandas as pd
import numpy as np
import time
import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from promptlearn import PromptClassifier

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Args
parser = argparse.ArgumentParser(description="Run classifier benchmark with train/val files or OpenML dataset")
parser.add_argument("--dataset", type=str, help="OpenML dataset name or single CSV/TSV file path")
parser.add_argument("--target", type=str, required=True, help="Name of the target column")
parser.add_argument("--train", type=str, help="Path to train CSV/TSV")
parser.add_argument("--val", type=str, help="Path to val CSV/TSV")
parser.add_argument("--val_rows", type=int, default=None, help="Cap number of val rows")
args = parser.parse_args()

# Load data
def load_file(path):
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)

if args.train and args.val:
    print("📦 Loading train/val from separate files...")
    df_train = load_file(args.train)
    df_val = load_file(args.val)
    df_full = pd.concat([df_train, df_val], ignore_index=True)  # for consistent encoding
elif args.dataset:
    print(f"📦 Loading dataset: {args.dataset}")
    if args.dataset.endswith(".csv") or args.dataset.endswith(".tsv"):
        df_full = load_file(args.dataset)
        df_train, df_val = train_test_split(df_full, test_size=0.3, stratify=df_full[args.target], random_state=42)
    else:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(args.dataset, version=1, as_frame=True)
        df_full = pd.concat([data.data, data.target.rename(args.target)], axis=1)
        df_train, df_val = train_test_split(df_full, test_size=0.3, stratify=df_full[args.target], random_state=42)
else:
    raise ValueError("Must provide either --dataset or both --train and --val")

# Split X/y
X_train_raw = df_train.drop(columns=[args.target])
y_train = df_train[args.target]
X_val_raw = df_val.drop(columns=[args.target])
y_val = df_val[args.target]

# Normalize target if needed
if y_train.dtype == "object":
    y_train = pd.factorize(y_train)[0]
    y_val = pd.factorize(y_val)[0]

# Merge for encoding
X_combined = pd.concat([X_train_raw, X_val_raw], axis=0)
X_combined_encoded = pd.get_dummies(X_combined)
X_combined_llm = X_combined.astype(str).fillna("unknown")

# Split encoded
X_train_encoded = X_combined_encoded.iloc[:len(X_train_raw)]
X_val_encoded = X_combined_encoded.iloc[len(X_train_raw):]
X_train_llm = X_combined_llm.iloc[:len(X_train_raw)]
X_val_llm = X_combined_llm.iloc[len(X_train_raw):]

# Optional row cap
if args.val_rows:
    X_val_llm = X_val_llm.iloc[:args.val_rows]
    X_val_encoded = X_val_encoded.iloc[:args.val_rows]
    y_val = y_val.iloc[:args.val_rows]

print(f"✅ Training on {len(X_train_llm)} rows, validating on {len(X_val_llm)} rows")

# Models
baselines = {
    "dummy": DummyClassifier(strategy="most_frequent"),
    "decision_tree": DecisionTreeClassifier(max_depth=3),
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=20, max_depth=4),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=30, max_depth=3),
}
llm_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "o3-mini", "o4-mini"]
promptlearners = {
    f"promptlearn_{model}": PromptClassifier(
        model=model, verbose=True
    ) for model in llm_models
}
all_models = {**baselines, **promptlearners}

# Benchmark
results = []
print("🚀 Beginning benchmark...\n")

for name, clf in all_models.items():
    print(f"▶️ Training: {name}")
    start = time.time()
    clf.fit(X_train_llm if "promptlearn" in name else X_train_encoded, y_train)
    fit_time = time.time() - start
    print(f"⏱️ Fit time: {fit_time:.2f}s")

    print(f"🔮 Predicting: {name}")
    start = time.time()
    y_pred = clf.predict(X_val_llm if "promptlearn" in name else X_val_encoded)
    pred_time = time.time() - start
    acc = accuracy_score(y_val, y_pred)
    print(f"✅ Accuracy ({name}): {acc:.4f} | ⏱️ Predict time: {pred_time:.2f}s\n")

    results.append({
        "model": name,
        "accuracy": acc,
        "fit_time_sec": fit_time,
        "predict_time_sec": pred_time
    })

# Final report
df_results = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print("\n=== Final Results ===")
print(df_results.to_string(index=False))

# for data/mammal dataset:
# === Final Results ===
#                     model  accuracy  fit_time_sec  predict_time_sec
#       promptlearn_o4-mini      1.00     66.392957          0.003270
#       promptlearn_o3-mini      0.94     41.296453          0.002382
#         promptlearn_gpt-4      0.66     12.428878          0.002444
#        promptlearn_gpt-4o      0.66      3.613194          0.003657
# promptlearn_gpt-3.5-turbo      0.63      2.645796          0.002558
#       logistic_regression      0.60      0.028289          0.000958
#         gradient_boosting      0.53      0.016901          0.000801
#             decision_tree      0.53      0.002735          0.001191
#             random_forest      0.44      0.012090          0.001132
#                     dummy      0.34      0.000579          0.014026
