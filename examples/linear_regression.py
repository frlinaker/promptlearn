#!/usr/bin/env python
"""Example: PromptRegressor learning a linear relationship y = 2x + 3.

By default the data is clean; pass --noise to add Gaussian noise and use more
points, to show the regressor still recovers the underlying line.

Usage:
    python linear_regression.py
    python linear_regression.py --noise 0.2 --points 20
"""

import argparse
import logging

import numpy as np
from sklearn.metrics import mean_squared_error

from promptlearn import PromptRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise", type=float, default=0.0, help="Std-dev of Gaussian noise on y."
    )
    parser.add_argument(
        "--points", type=int, default=5, help="Number of training points."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    # Training data: y = 2x + 3 (+ optional noise)
    X = np.linspace(-1, 3, args.points).reshape(-1, 1)
    y = 2 * X.flatten() + 3 + np.random.normal(0.0, args.noise, size=args.points)

    model = PromptRegressor(verbose=True)
    model.fit(X, y)

    X_test = np.array([[4], [5], [6]])
    y_pred = model.predict(X_test)

    print("\nPredictions:")
    for x_val, y_val in zip(X_test.flatten(), y_pred):
        print(f"x={x_val} → y={y_val:.2f}")

    y_true = 2 * X_test.flatten() + 3
    print(
        f"\nMean Squared Error (vs. true 2x+3): {mean_squared_error(y_true, y_pred):.4f}"
    )


if __name__ == "__main__":
    main()
