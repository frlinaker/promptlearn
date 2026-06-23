#!/usr/bin/env python
"""Titanic example for promptlearn — a teaching script.

promptlearn fits an LLM-backed classifier by having the model *write a Python
function* that encodes the decision rule. This example is built to show what
makes that approach interesting:

  1. ``fit()`` asks an LLM to generate a ``predict()`` function from your data,
     then validates that it actually runs (retrying if not). Inspect the result
     via ``clf.python_code_`` — that function *is* the model.
  2. ``predict()`` runs that generated code directly. There is **no LLM call per
     row**, so inference is fast and fully reproducible.
  3. ``explain()`` returns a plain-English description of the learned rule
     (global), or of a single prediction (local).
  4. Because the fitted model is just code, it serializes cleanly with joblib.

Usage:
    python titanic_classifier.py                      # fit, score, show generated code
    python titanic_classifier.py --explain            # also print explanations
    python titanic_classifier.py --dump artifacts/    # also save code/explanation/model
    python titanic_classifier.py --model claude-sonnet-4-6   # use a different provider
    python titanic_classifier.py --verbose            # show the LLM prompts during fit
"""

import argparse
import logging
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from promptlearn import PromptClassifier


def parse_args():
    p = argparse.ArgumentParser(
        description="Train and inspect a promptlearn classifier on the Titanic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="Print a plain-English explanation of the learned rule, plus a few explained predictions.",
    )
    p.add_argument(
        "--dump",
        nargs="?",
        const="titanic_artifacts",
        default=None,
        metavar="DIR",
        help="Save the generated code, explanation, and fitted model to DIR.",
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model string (e.g. gpt-4o, claude-sonnet-4-6, ollama:llama3.1).",
    )
    p.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of validation rows to explain (with --explain).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show the LLM prompts and responses during fit.",
    )
    return p.parse_args()


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def load_titanic():
    """Return (dataframe, target_column). Uses seaborn if available, else a CSV.

    The two sources name the target differently ('survived' vs 'Survived'), so
    we return the right name alongside the data.
    """
    try:
        import seaborn as sns

        return sns.load_dataset("titanic"), "survived"
    except Exception:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        return pd.read_csv(url), "Survived"


def preprocess(df, target):
    """Light, LLM-friendly preprocessing: drop leaky/sparse columns, fill gaps,
    and keep categoricals as readable strings (the LLM reasons over them by name)."""
    df = df.drop(
        columns=["deck", "embark_town", "alive", "who", "adult_male", "class", "alone"],
        errors="ignore",
    )
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    X = X.fillna("unknown")
    for col in [c for c in X.columns if X[c].dtype == object]:
        X[col] = X[col].astype(str)
    return X, y


def show_generated_code(clf):
    banner("GENERATED PYTHON HEURISTIC  (clf.python_code_)")
    print(
        "This function *is* the model — predict() runs it directly, with no LLM\n"
        "call per row. It was written by the LLM from the training data and then\n"
        "validated to make sure it executes.\n"
    )
    print(clf.python_code_)


def show_explanations(clf, X_val, y_val, n):
    banner("GLOBAL EXPLANATION  (clf.explain())")
    explanation = clf.explain()
    print(explanation.summary)
    # features_used is derived from the code, so it never invents a feature the
    # rule doesn't actually look at.
    print("\nFeatures the rule actually uses:", explanation.features_used)

    n = min(n, len(X_val))
    banner(f"LOCAL EXPLANATIONS  (clf.explain(row))  — first {n} validation rows")
    for i in range(n):
        row = X_val.iloc[[i]]
        pred = int(clf.predict(row)[0])
        actual = int(y_val.iloc[i])
        mark = "correct" if pred == actual else "WRONG"
        print(f"\n--- row {i}: predicted={pred}, actual={actual} ({mark}) ---")
        print(clf.explain(row).summary)


def dump_artifacts(clf, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "titanic_raw.py": clf.raw_python_code_ or "",
        "titanic_extended.py": clf.python_code_ or "",
        "titanic_explanation.txt": clf.explain().summary,
    }
    written = []
    for name, content in paths.items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(path)

    # The model is just code + metadata, so joblib serialization is tiny and
    # needs no LLM client to reload.
    model_path = os.path.join(out_dir, "titanic_model.joblib")
    joblib.dump(clf, model_path)
    written.append(model_path)

    banner("DUMPED ARTIFACTS")
    for path in written:
        print(" •", os.path.abspath(path))


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    df, target = load_titanic()
    X, y = preprocess(df, target)
    print(f"Loaded {len(X)} rows, {X.shape[1]} features: {list(X.columns)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf = PromptClassifier(model=args.model, verbose=args.verbose)
    print(
        f"\nFitting PromptClassifier(model={args.model!r}) on {len(X_train)} rows ..."
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    banner(f"VALIDATION ACCURACY: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    # The headline feature: always show the code the model learned.
    show_generated_code(clf)

    if args.explain:
        show_explanations(clf, X_val, y_val, args.examples)

    if args.dump is not None:
        dump_artifacts(clf, args.dump)


if __name__ == "__main__":
    main()
