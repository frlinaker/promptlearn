#!/usr/bin/env python

# example:
# python examples/benchmark_regressor.py --train examples/data/fall_train.csv --val examples/data/fall_val.csv --target fall_time_s

import argparse
import pandas as pd
import numpy as np
import time
import logging
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from promptlearn import PromptRegressor

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(
    description="Run regressor benchmark with train/val files or OpenML dataset"
)
parser.add_argument(
    "--dataset", type=str, help="OpenML dataset name or single CSV/TSV file path"
)
parser.add_argument(
    "--target", type=str, required=True, help="Name of the target column"
)
parser.add_argument("--train", type=str, help="Path to train CSV/TSV")
parser.add_argument("--val", type=str, help="Path to val CSV/TSV")
parser.add_argument("--val_rows", type=int, default=None, help="Cap number of val rows")
args = parser.parse_args()


def load_file(path):
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


if args.train and args.val:
    print("📦 Loading train/val from separate files...")
    df_train = load_file(args.train)
    df_val = load_file(args.val)
    df_full = pd.concat([df_train, df_val], ignore_index=True)
elif args.dataset:
    print(f"📦 Loading dataset: {args.dataset}")
    if args.dataset.endswith(".csv") or args.dataset.endswith(".tsv"):
        df_full = load_file(args.dataset)
        df_train, df_val = train_test_split(df_full, test_size=0.3, random_state=42)
    else:
        from sklearn.datasets import fetch_openml

        data = fetch_openml(args.dataset, version=1, as_frame=True)
        df_full = pd.concat([data.data, data.target.rename(args.target)], axis=1)
        df_train, df_val = train_test_split(df_full, test_size=0.3, random_state=42)
else:
    raise ValueError("Must provide either --dataset or both --train and --val")

X_train_raw = df_train.drop(columns=[args.target])
y_train = df_train[args.target]
X_val_raw = df_val.drop(columns=[args.target])
y_val = df_val[args.target]

X_combined = pd.concat([X_train_raw, X_val_raw], axis=0)
X_combined_encoded = pd.get_dummies(X_combined)
X_combined_llm = X_combined.astype(str).fillna("unknown")

X_train_encoded = X_combined_encoded.iloc[: len(X_train_raw)]
X_val_encoded = X_combined_encoded.iloc[len(X_train_raw) :]
X_train_llm = X_combined_llm.iloc[: len(X_train_raw)]
X_val_llm = X_combined_llm.iloc[len(X_train_raw) :]

if args.val_rows:
    X_val_llm = X_val_llm.iloc[: args.val_rows]
    X_val_encoded = X_val_encoded.iloc[: args.val_rows]
    y_val = y_val.iloc[: args.val_rows]

print(f"✅ Training on {len(X_train_llm)} rows, validating on {len(X_val_llm)} rows")

baselines = {
    "dummy": DummyRegressor(strategy="mean"),
    "decision_tree": DecisionTreeRegressor(max_depth=4),
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=20, max_depth=4),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=30, max_depth=3),
}
llm_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "o3-mini", "o4-mini"]
promptlearners = {
    f"promptlearn_{model}": PromptRegressor(model=model, verbose=False)
    for model in llm_models
}
all_models = {**baselines, **promptlearners}

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
    mse = mean_squared_error(y_val, y_pred)
    print(f"✅ MSE ({name}): {mse:.4f} | ⏱️ Predict time: {pred_time:.2f}s\n")

    results.append(
        {
            "model": name,
            "mse": mse,
            "fit_time_sec": fit_time,
            "predict_time_sec": pred_time,
        }
    )

df_results = pd.DataFrame(results).sort_values("mse")
print("\n=== Final Results ===")
print(df_results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# python .\examples\benchmark_regressor.py --train examples/data/fall_train.csv --val examples/data/fall_val.csv --target fall_time_s
# === Final Results ===
#                     model          mse  fit_time_sec  predict_time_sec
#        promptlearn_gpt-4o 2.366583e-31      2.141143          0.000624
#       promptlearn_o3-mini 2.366583e-31      7.652526          0.000641
#       promptlearn_o4-mini 2.366583e-31     11.474670          0.000639
#             random_forest 2.347590e-02      0.010189          0.000904
#         gradient_boosting 3.537142e-02      0.009220          0.000680
#             decision_tree 6.674515e-02      0.000895          0.000411
#         linear_regression 4.978544e-01      0.015677          0.000996
#         promptlearn_gpt-4 1.560732e+00      7.180273          0.000686
#                     dummy 5.272532e+00      0.000854          0.000066
# promptlearn_gpt-3.5-turbo 1.819323e+01      1.182629          0.000602
