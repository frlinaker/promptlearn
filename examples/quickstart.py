#!/usr/bin/env python
"""A quick tour of promptlearn's headline capabilities.

Each demo is a small, self-contained illustration of one feature. They make
live LLM calls, so run them one at a time with --demo to keep cost down:

    python quickstart.py --list
    python quickstart.py --demo zero_row
    python quickstart.py --demo world_knowledge --model claude-sonnet-4-6
    python quickstart.py                       # run them all

For the deeper, guided walkthrough (generated code, explain(), joblib) see
titanic_classifier.py; for model-vs-model benchmarking see compare_models.py.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from promptlearn import PromptClassifier, PromptRegressor


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def demo_zero_row(model):
    """Fit with column names only — no rows. The LLM infers the rule from the
    schema and its world knowledge."""
    X = pd.DataFrame(columns=["country_name"])
    y = pd.Series(name="has_blue_in_flag", dtype=int)

    clf = PromptClassifier(model=model, verbose=False)
    clf.fit(X, y)  # only headers — nothing to learn from but the names

    for country, expected in [("Japan", "no"), ("France", "yes")]:
        pred = int(clf.predict(pd.DataFrame([{"country_name": country}]))[0])
        print(f"  {country}: has_blue_in_flag={pred}  (expected ~{expected})")


def demo_sample(model):
    """Generate synthetic rows from a fitted model with .sample(n)."""
    X = np.array([[-1], [0], [1], [2], [3]])
    y = np.array([1, 3, 5, 7, 9])  # y = 2x + 3

    model_ = PromptRegressor(model=model, verbose=False)
    model_.fit(X, y)
    print(model_.sample(10).to_string(index=False))


def demo_joblib(model):
    """The fitted model is just code, so it serializes tiny and reloads without
    an LLM client."""
    import joblib

    X = np.array([[-1], [0], [1], [2], [3]])
    y = np.array([1, 3, 5, 7, 9])  # y = 2x + 3

    reg = PromptRegressor(model=model, verbose=False)
    reg.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(reg, path)
        size = path.stat().st_size
        reloaded = joblib.load(path)
        preds = reloaded.predict(np.array([[4], [5]]))
    print(f"  serialized to {size} bytes, reloaded and predicted: {np.round(preds, 2)}")
    print("  (no LLM client is stored; the heuristic is recompiled on load)")


def demo_world_knowledge(model):
    """promptlearn can fold in real-world knowledge that the raw features alone
    don't contain — both for classification and regression."""
    banner("world knowledge — classification (does the flag contain blue?)")
    data = pd.DataFrame(
        {
            "country_name": ["Sweden", "Japan", "Italy", "United States", "Germany"],
            "has_blue_in_flag": [1, 0, 0, 1, 0],
        }
    )
    clf = PromptClassifier(model=model, verbose=False)
    clf.fit(data[["country_name"]], data["has_blue_in_flag"])
    for country in ["France", "Brazil", "Spain"]:
        pred = int(clf.predict(pd.DataFrame([{"country_name": country}]))[0])
        print(f"  {country}: {'blue' if pred else 'no blue'}")

    # The classic "how much money was the dog given?" riddle: payout scales with
    # the number of legs, which the model knows per animal.
    banner("world knowledge — regression (money-per-animal riddle)")
    train = pd.DataFrame({"animal": ["chicken", "ant", "spider"], "money": [7, 21, 28]})
    reg = PromptRegressor(model=model, verbose=False)
    reg.fit(train[["animal"]], train["money"])
    for animal in ["dog", "bee", "crab"]:
        pred = float(reg.predict(pd.DataFrame([{"animal": animal}]))[0])
        print(f"  {animal}: {pred:.1f}")


def demo_nonlinear(model):
    """Recover a nonlinear, multi-variable relationship:
    y = 3 * length^2 + 2 * volume + 5."""
    rng = np.random.default_rng(42)
    length = rng.uniform(-2, 2, size=40)
    volume = rng.uniform(0, 5, size=40)
    X = pd.DataFrame({"length": length, "volume": volume})
    y = pd.Series(3 * length**2 + 2 * volume + 5, name="output")

    reg = PromptRegressor(model=model, verbose=False)
    reg.fit(X, y)

    tests = pd.DataFrame(
        [
            {"length": 1.0, "volume": 2.0},
            {"length": -1.5, "volume": 3.0},
            {"length": 0.0, "volume": 4.0},
        ]
    )
    preds = reg.predict(tests)
    truth = 3 * tests["length"] ** 2 + 2 * tests["volume"] + 5
    for (_, row), pred, true in zip(tests.iterrows(), preds, truth):
        print(
            f"  length={row['length']:+.1f} volume={row['volume']:.1f}"
            f" → predicted={pred:6.2f}  expected={true:6.2f}"
        )


def demo_multioutput(model):
    """promptlearn estimators are sklearn-compatible, so meta-estimators like
    MultiOutputRegressor wrap them directly (Linnerud: 3 targets)."""
    from sklearn.datasets import load_linnerud
    from sklearn.multioutput import MultiOutputRegressor

    data = load_linnerud()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=data.target_names)

    reg = MultiOutputRegressor(PromptRegressor(model=model, verbose=False))
    reg.fit(X, y)
    preds = pd.DataFrame(reg.predict(X.head()), columns=y.columns)
    print(preds.round(1).to_string(index=False))


def demo_gridsearch(model):
    """Because the estimators follow the sklearn API, GridSearchCV tunes their
    hyper-parameters (here: how many training rows to send the LLM)."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV

    data = load_iris(as_frame=True)
    X, y = data.data.head(60), data.target.head(60)  # keep it small/cheap

    search = GridSearchCV(
        PromptClassifier(model=model, verbose=False),
        param_grid={"max_train_rows": [20, 40]},
        cv=2,
    )
    search.fit(X, y)
    print(
        f"  best params: {search.best_params_}  best CV score: {search.best_score_:.3f}"
    )


DEMOS = {
    "zero_row": demo_zero_row,
    "sample": demo_sample,
    "joblib": demo_joblib,
    "world_knowledge": demo_world_knowledge,
    "nonlinear": demo_nonlinear,
    "multioutput": demo_multioutput,
    "gridsearch": demo_gridsearch,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--demo",
        choices=sorted(DEMOS),
        help="Run a single demo (default: run all).",
    )
    p.add_argument("--list", action="store_true", help="List demos and exit.")
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model string (e.g. gpt-4o, claude-sonnet-4-6, ollama:llama3.1).",
    )
    p.add_argument(
        "--verbose", action="store_true", help="Show LLM prompts during fit."
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        for name, fn in DEMOS.items():
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:16} {summary}")
        return

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    selected = [args.demo] if args.demo else list(DEMOS)
    for name in selected:
        banner(f"DEMO: {name}  (model={args.model})")
        DEMOS[name](args.model)


if __name__ == "__main__":
    main()
