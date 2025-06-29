#!/usr/bin/env python

"""
Benchmark different LLM models on simple regression tasks using PromptRegressor.
Includes:
- Linear 1D regression (y = 2x + 3 + noise)
- Nonlinear 2D regression (baking time = 0.2 * diameter^1.5 + 0.05 * filling_volume + 10 + noise)
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
from promptlearn import PromptRegressor

def generate_linear_1d_data(n=20, noise_std=0.2, seed=42):
    np.random.seed(seed)
    X = np.linspace(-3, 6, n).reshape(-1, 1)
    y = 2 * X.flatten() + 3 + np.random.normal(0.0, noise_std, size=n)
    return pd.DataFrame(X, columns=["x"]), pd.Series(y, name="target")

def generate_baking_data(n=50, noise_std=0.5, seed=42):
    np.random.seed(seed)
    diameter = np.random.uniform(5, 25, size=n)  # cm
    filling_volume = np.random.uniform(0, 100, size=n)  # ml
    y = 0.2 * diameter**1.5 + 0.05 * filling_volume + 10 + np.random.normal(0, noise_std, size=n)
    X = pd.DataFrame({"diameter": diameter, "filling_volume": filling_volume})
    return X, pd.Series(y, name="baking_time")

def generate_test_points(task_name: str):
    if task_name == "linear_1d":
        X_test = pd.DataFrame([[4], [5], [6]], columns=["x"])
        y_true = 2 * X_test["x"] + 3
    elif task_name == "nonlinear_2d":
        X_test = pd.DataFrame([
            {"diameter": 10.0, "filling_volume": 20.0},
            {"diameter": 15.0, "filling_volume": 50.0},
            {"diameter": 20.0, "filling_volume": 80.0}
        ])
        y_true = 0.2 * X_test["diameter"]**1.5 + 0.05 * X_test["filling_volume"] + 10
    else:
        raise ValueError(f"Unknown task: {task_name}")
    return X_test, y_true

def evaluate_model(model_name: str, X_train, y_train, X_test, y_true) -> dict:
    print(f"\n▶️ Running model: {model_name}")
    reg = PromptRegressor(model=model_name, verbose=True)
    reg.fit(X_train, y_train)

    print("📜 Learned heuristic:")
    print(reg.heuristic_)

    y_pred = []
    for x in X_test.itertuples(index=False):
        try:
            x_dict = x._asdict() if hasattr(x, "_asdict") else dict(zip(X_test.columns, x))
            y_pred.append(reg._predict_one(x_dict))
        except Exception as e:
            warnings.warn(f"⚠️ Failed to predict for input {x}: {e}")
            y_pred.append(np.nan)

    y_pred = np.array(y_pred)
    valid_mask = ~np.isnan(y_pred)
    num_dropped = (~valid_mask).sum()

    if num_dropped > 0:
        print(f"⚠️ Dropped {num_dropped} non-numeric predictions for model: {model_name}")

    y_pred_valid = y_pred[valid_mask]
    y_true_valid = np.array(y_true)[valid_mask]

    if len(y_pred_valid) == 0:
        mse = np.nan
        print("⚠️ No valid predictions — MSE set to NaN")
    else:
        mse = mean_squared_error(y_true_valid, y_pred_valid)

    print("🔍 Predictions vs ground truth:")
    for x, yp, yt in zip(X_test.values, y_pred, y_true):
        x_str = ", ".join(f"{name}={val:.2f}" for name, val in zip(X_test.columns, x))
        val_str = f"{yp:.2f}" if not np.isnan(yp) else "NaN"
        print(f"  {x_str} → predicted={val_str}, true={yt:.2f}")

    print(f"📊 MSE: {mse:.4f}" if not np.isnan(mse) else "📊 MSE: NaN")

    return {
        "model": model_name,
        "mse": mse,
        "heuristic": reg.heuristic_
    }

def run_benchmark(models, task: str, seeds=[42], noise_std=0.2):
    results = []
    for seed in seeds:
        print(f"\n=== Task: {task}, Seed: {seed} ===")
        if task == "linear_1d":
            X_train, y_train = generate_linear_1d_data(n=20, noise_std=noise_std, seed=seed)
        elif task == "nonlinear_2d":
            X_train, y_train = generate_baking_data(n=50, noise_std=noise_std, seed=seed)
        else:
            raise ValueError(f"Unknown task: {task}")

        X_test, y_true = generate_test_points(task)

        for model in models:
            result = evaluate_model(model, X_train, y_train, X_test, y_true)
            result.update({"task": task, "seed": seed})
            results.append(result)

    return pd.DataFrame(results)

if __name__ == "__main__":
    MODELS = ["o3-mini", "o4-mini", "gpt-4", "gpt-4o"]
    TASKS = ["linear_1d", 
             "nonlinear_2d"]

    all_results = []
    for task in TASKS:
        df = run_benchmark(MODELS, task)
        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)
    print("\n=== Final Results ===")
    print(final_df[["task", "model", "seed", "mse"]])
