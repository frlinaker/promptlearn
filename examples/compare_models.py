#!/usr/bin/env python
"""Compare promptlearn against classical models on the same dataset.

Fits a configurable set of models — promptlearn's PromptClassifier/PromptRegressor
alongside scikit-learn baselines and (if installed) XGBoost — on one dataset and
prints:

  * a side-by-side METRICS table (one row per model), and
  * a row-by-row PREDICTIONS table (one column per model).

All the heavy lifting is in the reusable ``promptlearn.compare_models`` helper;
this script just assembles a model zoo and a dataset.

Usage:
    python compare_models.py                       # iris (classification)
    python compare_models.py --dataset titanic
    python compare_models.py --dataset diabetes    # regression
    python compare_models.py --dataset mammal      # world-knowledge classification
    python compare_models.py --dataset fall        # physics regression (recover f=ma)
    python compare_models.py --no-llm              # skip promptlearn (fast, free)
    python compare_models.py --rows 15 --model claude-sonnet-4-6
"""

import argparse
import os

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from promptlearn import PromptClassifier, PromptRegressor, compare_models


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def _load_csv_pair(stem, target, task):
    """Load examples/data/<stem>_{train,val}.csv and return (X, y, task)."""
    here = os.path.dirname(os.path.abspath(__file__))
    frames = [
        pd.read_csv(os.path.join(here, "data", f"{stem}_{split}.csv"))
        for split in ("train", "val")
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.drop(columns=[target]), df[target], task


def load_dataset(name):
    """Return (X, y, task) for one of: iris, titanic, diabetes, mammal, fall."""
    if name == "iris":
        from sklearn.datasets import load_iris

        data = load_iris(as_frame=True)
        return data.data, data.target, "classification"

    if name == "diabetes":
        from sklearn.datasets import load_diabetes

        data = load_diabetes(as_frame=True)
        return data.data, data.target, "regression"

    # World-knowledge classification: only promptlearn can use the `animal` name.
    if name == "mammal":
        return _load_csv_pair("mammal", "is_mammal", "classification")

    # Physics regression: promptlearn can recover fall_time = sqrt(2*h/g).
    if name == "fall":
        return _load_csv_pair("fall", "fall_time_s", "regression")

    if name == "titanic":
        try:
            import seaborn as sns

            df, target = sns.load_dataset("titanic"), "survived"
        except Exception:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df, target = pd.read_csv(url), "Survived"
        # Keep readable, low-cardinality features (drop names/tickets/cabins).
        keep = {"Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        target = "Survived"
        df = df.dropna(subset=[target])
        X = df[[c for c in df.columns if c in keep]]
        return X, df[target].astype(int), "classification"

    raise ValueError(f"unknown dataset {name!r}")


def build_models(task, use_llm, llm_model):
    """Assemble {name: estimator} for the task, plus XGBoost if available."""
    if task == "classification":
        models = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "logreg": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=0),
        }
        if use_llm:
            models[f"promptlearn[{llm_model}]"] = PromptClassifier(
                model=llm_model, verbose=False
            )
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(eval_metric="logloss")
        except ImportError:
            pass
    else:
        models = {
            "dummy": DummyRegressor(),
            "linreg": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=50, random_state=0),
        }
        if use_llm:
            models[f"promptlearn[{llm_model}]"] = PromptRegressor(
                model=llm_model, verbose=False
            )
        try:
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor()
        except ImportError:
            pass
    return models


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="iris",
        choices=["iris", "titanic", "diabetes", "mammal", "fall"],
        help="Which built-in dataset to use.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the promptlearn model (fast, no API key / cost).",
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model string for promptlearn (e.g. gpt-4o, claude-sonnet-4-6).",
    )
    p.add_argument(
        "--rows", type=int, default=10, help="Rows of per-instance predictions to show."
    )
    return p.parse_args()


def main():
    args = parse_args()
    X, y, task = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = build_models(task, use_llm=not args.no_llm, llm_model=args.model)
    print(
        f"Dataset: {args.dataset} ({task}, {len(X)} rows, {X.shape[1]} features)\n"
        f"Models:  {', '.join(models)}"
    )

    metrics, predictions = compare_models(
        models, X_train, y_train, X_test, y_test, task
    )

    primary, ascending = ("rmse", True) if task == "regression" else ("accuracy", False)
    metrics = metrics.sort_values(primary, ascending=ascending)

    banner(f"METRICS (side by side, sorted by {primary})")
    print(metrics.to_string(float_format=lambda v: f"{v:.4f}"))

    banner(f"PREDICTIONS (row by row, first {args.rows} test rows)")
    print(predictions.head(args.rows).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
