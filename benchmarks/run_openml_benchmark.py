#!/usr/bin/env python
"""Benchmark the impact of LLM feature engineering across OpenML datasets.

This is a 2-factor study on each dataset's common train/test split:

  * learner  ∈ {promptlearn (PromptClassifier), logreg, xgboost}
  * features ∈ {FE off (raw columns), FE on (PromptFeatureEngineer applied first)}

For each dataset a single `PromptFeatureEngineer` is fit once and its transform
is reused for every "FE on" learner, so all three see identical engineered
features. Results are reported as accuracy with FE off, FE on, and the delta —
isolating what the LLM-engineered features add to each learner.

Built on `promptlearn.compare_models` (which one-hot-wraps the plain sklearn
learners). Datasets come from `sklearn.datasets.fetch_openml` (no extra
dependency). Per-dataset results are cached as JSON so reruns don't re-pay for
LLM calls; delete the cache dir (or pass `--no-cache`) to force a fresh run.

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
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from promptlearn import PromptClassifier, PromptFeatureEngineer, compare_models

logger = logging.getLogger("promptlearn.benchmark")

# A bump here invalidates older cache files when the result schema changes.
CACHE_SCHEMA = "v2-factorial"

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

# Display order for the learners.
LEARNERS = ["promptlearn", "logreg", "xgboost"]


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0
    )


def build_learners(model_name: str) -> dict:
    """Fresh learner instances. `compare_models` one-hot-wraps the sklearn ones;
    PromptClassifier consumes the (possibly FE-augmented) DataFrame directly."""
    learners = {
        "promptlearn": PromptClassifier(model=model_name, verbose=False),
        "logreg": LogisticRegression(max_iter=1000),
    }
    xgb = _xgb_classifier()
    if xgb is not None:
        learners["xgboost"] = xgb
    return learners


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
    raw = f"{CACHE_SCHEMA}|{dataset}|{model_name}|{max_rows}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def run_one(dataset, spec, model_name, max_rows, cache_dir):
    """Return a DataFrame indexed by learner with columns fe_off / fe_on / delta."""
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

    # FE off: learners on the raw columns.
    off, _ = compare_models(
        build_learners(model_name),
        X_train,
        y_train,
        X_test,
        y_test,
        task="classification",
    )

    # FE on: fit one feature engineer, reuse its transform for every learner.
    try:
        fe = PromptFeatureEngineer(model=model_name, verbose=False).fit(
            X_train, y_train
        )
        Xtr, Xte = fe.transform(X_train), fe.transform(X_test)
        on, _ = compare_models(
            build_learners(model_name), Xtr, y_train, Xte, y_test, task="classification"
        )
        on_acc = on["accuracy"]
        logger.info("[%s] engineered features: %s", dataset, fe.new_feature_names_)
    except Exception as e:
        logger.warning("[%s] feature engineering failed: %s", dataset, e)
        on_acc = pd.Series(np.nan, index=off.index)

    result = pd.DataFrame({"fe_off": off["accuracy"], "fe_on": on_acc})
    result["delta"] = result["fe_on"] - result["fe_off"]
    result = result.assign(dataset=dataset, n_rows=len(X), n_classes=n_classes)
    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        result.to_json(cache_file)
    return result


def _fmt(v, signed=False):
    if pd.isna(v):
        return ""
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def impact_markdown(impact: pd.DataFrame) -> str:
    """Learner × {FE off, FE on, Δ} summary table."""
    lines = [
        "| learner | FE off | FE on | Δ (FE on − off) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for learner, row in impact.iterrows():
        lines.append(
            f"| {learner} | {_fmt(row['fe_off'])} | {_fmt(row['fe_on'])} | "
            f"{_fmt(row['delta'], signed=True)} |"
        )
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
        help="LLM model for promptlearn / feature engineering (default: gpt-5.4-mini).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Subsample datasets larger than this many rows (cost/time control).",
    )
    parser.add_argument("--output", help="Write the markdown impact table here.")
    parser.add_argument("--cache-dir", default="benchmarks/.cache")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("promptlearn").setLevel(logging.WARNING)  # quiet per-prompt logs
    logger.setLevel(logging.INFO)

    cache_dir = None if args.no_cache else Path(args.cache_dir)

    results = {}
    for dataset in args.datasets:
        spec = DEFAULT_DATASETS.get(dataset)
        if spec is None:
            logger.warning("unknown dataset %r, skipping", dataset)
            continue
        try:
            start = time.time()
            res = run_one(dataset, spec, args.model, args.max_rows, cache_dir)
            results[dataset] = res
            print(f"\n=== {dataset}  ({time.time() - start:.0f}s) ===", flush=True)
            print(res[["fe_off", "fe_on", "delta"]].round(3).to_string(), flush=True)
        except Exception as e:
            logger.warning("[%s] failed: %s", dataset, e)

    if not results:
        print("No results.")
        return 1

    # Mean accuracy per learner across datasets, FE off vs on.
    off = pd.DataFrame({d: r["fe_off"] for d, r in results.items()}).T
    on = pd.DataFrame({d: r["fe_on"] for d, r in results.items()}).T
    impact = pd.DataFrame({"fe_off": off.mean(), "fe_on": on.mean()})
    impact["delta"] = impact["fe_on"] - impact["fe_off"]
    impact = impact.reindex([l for l in LEARNERS if l in impact.index])

    md = impact_markdown(impact)
    print(f"\n## Feature-engineering impact (model={args.model}, mean accuracy)\n")
    print(md)
    if args.output:
        Path(args.output).write_text(
            f"## promptlearn OpenML benchmark — feature-engineering impact "
            f"(model={args.model}, mean accuracy)\n\n{md}\n"
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
