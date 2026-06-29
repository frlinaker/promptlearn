"""Command-line interface for promptlearn.

Usage
-----
    promptlearn fit mydata.csv --target label
    promptlearn fit mydata.csv --target label --model gpt-5.5 --task regression
    promptlearn fit mydata.csv --target label --test-size 0.2 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from promptlearn import PromptClassifier, PromptRegressor


def _load_csv(path: str):
    import pandas as pd

    return pd.read_csv(path)


def _split(X, y, test_size: float, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def cmd_fit(args: argparse.Namespace) -> int:
    import numpy as np
    from sklearn.metrics import accuracy_score, r2_score

    df = _load_csv(args.file)

    if args.target not in df.columns:
        print(
            f"error: column '{args.target}' not found in {args.file}", file=sys.stderr
        )
        print(f"  available columns: {', '.join(df.columns)}", file=sys.stderr)
        return 1

    X = df.drop(columns=[args.target])
    y = df[args.target]

    if args.test_size > 0:
        X_train, X_test, y_train, y_test = _split(X, y, args.test_size)
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    task = args.task
    if task == "auto":
        task = (
            "regression"
            if y.dtype in (float, "float64", "float32")
            else "classification"
        )
    print(
        f"task: {task}  |  model: {args.model}  |  rows: {len(X_train)} train", end=""
    )
    if X_test is not None:
        print(f" / {len(X_test)} test", end="")
    print()

    if task == "classification":
        clf = PromptClassifier(model=args.model, verbose=args.verbose)
        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"train accuracy: {train_acc:.4f}")

        if X_test is not None:
            test_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            print(f"test  accuracy: {test_acc:.4f}")

    elif task == "regression":
        reg = PromptRegressor(model=args.model, verbose=args.verbose)
        reg.fit(X_train, y_train)

        train_pred = reg.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        print(f"train R²: {train_r2:.4f}")

        if X_test is not None:
            test_pred = reg.predict(X_test)
            test_r2 = r2_score(y_test, test_pred)
            print(f"test  R²: {test_r2:.4f}")

    else:
        print(
            f"error: unknown task '{task}' — use 'classification' or 'regression'",
            file=sys.stderr,
        )
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="promptlearn",
        description="LLM-powered zero-shot classification and regression",
    )
    sub = parser.add_subparsers(dest="command")

    fit_p = sub.add_parser("fit", help="Fit a model on a CSV and report accuracy")
    fit_p.add_argument("file", help="Path to CSV file")
    fit_p.add_argument("--target", required=True, help="Name of the target column")
    fit_p.add_argument(
        "--model",
        default=None,
        help="LLM model ID (default: uses PROMPTLEARN_MODEL env or gpt-4o)",
    )
    fit_p.add_argument(
        "--task",
        choices=["auto", "classification", "regression"],
        default="auto",
        help="Task type (default: auto-detect from target dtype)",
    )
    fit_p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Fraction of data held out for test evaluation (default: 0.2, 0 to skip)",
    )
    fit_p.add_argument("--verbose", action="store_true", help="Show generated code")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "fit":
        return cmd_fit(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
