#!/usr/bin/env python
"""Benchmark promptlearn against classical baselines on OpenML datasets.

Compares, on a common train/test split per dataset:

  * ``promptlearn``       — PromptClassifier on the raw DataFrame (LLM writes the
                            classifier code; no downstream model)
  * ``promptFE->logreg``  — PromptFeatureEngineer -> one-hot -> LogisticRegression
                            (the LLM engineers features; logreg does the classifying)
  * ``logreg``            — LogisticRegression (one-hot + scaled)
  * ``xgboost``           — XGBClassifier / gradient-boosted trees (one-hot + scaled)

Built on ``promptlearn.compare_models``. Datasets are fetched with
``sklearn.datasets.fetch_openml`` (no extra dependency). Per-dataset results are
cached as JSON so reruns don't re-pay for LLM calls; delete the cache dir (or
pass ``--no-cache``) to force a fresh run.

Examples
--------
    # quick run on a couple of datasets with the cheap default model
    python benchmarks/run_openml_benchmark.py --datasets adult credit-g

    # full curated suite on the flagship model, write a markdown table
    python benchmarks/run_openml_benchmark.py --model gpt-5.5 --output results.md
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from promptlearn import PromptClassifier, PromptFeatureEngineer, compare_models

logger = logging.getLogger("promptlearn.benchmark")

# Curated classification datasets with semantically meaningful categorical
# columns (where world knowledge can plausibly help). (openml_name, version).
DEFAULT_DATASETS = {
    "adult": ("adult", 2),
    "credit-g": ("credit-g", 1),
    "bank-marketing": ("bank-marketing", 1),
    "mushroom": ("mushroom", 1),
    "car": ("car", 3),
    "nursery": ("nursery", 3),
    "vote": ("vote", 1),
    "tic-tac-toe": ("tic-tac-toe", 1),
    "kr-vs-kp": ("kr-vs-kp", 1),
    "monks-2": ("monks-problems-2", 1),
}


def _generic_encoder() -> ColumnTransformer:
    """One-hot categoricals + scale numerics, selected by dtype at runtime so it
    adapts to whatever columns PromptFeatureEngineer produces."""
    num = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num, make_column_selector(dtype_include=np.number)),
            # everything non-numeric (object, category, bool, string) is categorical
            ("cat", cat, make_column_selector(dtype_exclude=np.number)),
        ]
    )


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0
    )


def build_models(model_name: str) -> dict:
    """The contenders for one dataset, given the LLM model to use."""
    models = {
        "promptlearn": PromptClassifier(model=model_name, verbose=False),
        "promptFE->logreg": Pipeline(
            [
                ("fe", PromptFeatureEngineer(model=model_name, verbose=False)),
                ("enc", _generic_encoder()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "logreg": LogisticRegression(max_iter=1000),
    }
    xgb = _xgb_classifier()
    if xgb is not None:
        models["xgboost"] = xgb
    return models


def load_dataset(openml_name: str, version: int, max_rows: int):
    bunch = fetch_openml(
        name=openml_name, version=version, as_frame=True, parser="auto"
    )
    X = bunch.data.copy()
    y = pd.Series(np.asarray(bunch.target)).astype(str)
    # Encode the target to integer class ids (promptlearn predicts ints).
    classes = {c: i for i, c in enumerate(sorted(y.unique()))}
    y = y.map(classes).astype(int)
    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]
    return X.reset_index(drop=True), y.reset_index(drop=True), len(classes)


def _cache_key(dataset: str, model_name: str, max_rows: int) -> str:
    raw = f"{dataset}|{model_name}|{max_rows}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def run_one(dataset: str, spec, model_name: str, max_rows: int, cache_dir: Path | None):
    cache_file = (
        cache_dir / f"{dataset}-{_cache_key(dataset, model_name, max_rows)}.json"
        if cache_dir
        else None
    )
    if cache_file and cache_file.exists():
        logger.info("[%s] cached", dataset)
        return pd.read_json(cache_file)

    openml_name, version = spec
    logger.info("[%s] fetching openml(%s, v%s)…", dataset, openml_name, version)
    X, y, n_classes = load_dataset(openml_name, version, max_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logger.info(
        "[%s] %d rows, %d cols, %d classes", dataset, len(X), X.shape[1], n_classes
    )
    metrics, _ = compare_models(
        build_models(model_name),
        X_train,
        y_train,
        X_test,
        y_test,
        task="classification",
    )
    metrics = metrics.assign(dataset=dataset, n_rows=len(X), n_classes=n_classes)
    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_json(cache_file)
    return metrics


def to_markdown(accuracy_table: pd.DataFrame) -> str:
    """Render the accuracy table as a GitHub markdown table (no tabulate dep)."""
    df = accuracy_table.copy()
    df.loc["mean"] = df.mean(numeric_only=True)
    df = df.round(3)
    header = "| dataset | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, sep]
    for idx, row in df.iterrows():
        cells = " | ".join("" if pd.isna(v) else f"{v:.3f}" for v in row)
        lines.append(f"| {idx} | {cells} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Dataset keys to run (default: the full curated suite).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="LLM model for promptlearn contenders (default: gpt-5.4-mini).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Subsample datasets larger than this many rows (cost/time control).",
    )
    parser.add_argument("--output", help="Write the markdown accuracy table here.")
    parser.add_argument("--cache-dir", default="benchmarks/.cache")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("promptlearn").setLevel(logging.WARNING)  # quiet per-prompt logs

    cache_dir = None if args.no_cache else Path(args.cache_dir)

    acc_rows = {}
    for dataset in args.datasets:
        spec = DEFAULT_DATASETS.get(dataset)
        if spec is None:
            logger.warning("unknown dataset %r, skipping", dataset)
            continue
        try:
            start = time.time()
            metrics = run_one(dataset, spec, args.model, args.max_rows, cache_dir)
            acc_rows[dataset] = metrics["accuracy"]
            print(f"\n=== {dataset}  ({time.time() - start:.0f}s) ===")
            print(metrics.round(3).to_string())
        except Exception as e:
            logger.warning("[%s] failed: %s", dataset, e)

    if not acc_rows:
        print("No results.")
        return 1

    accuracy_table = pd.DataFrame(acc_rows).T  # rows=datasets, cols=models
    md = to_markdown(accuracy_table)
    print(f"\n## Accuracy (model={args.model})\n")
    print(md)
    if args.output:
        Path(args.output).write_text(
            f"## promptlearn OpenML benchmark — accuracy (model={args.model})\n\n{md}\n"
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
