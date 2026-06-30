#!/usr/bin/env python
"""Run logreg / XGBoost / TabPFN baselines on the curated OpenML suite.

Results are written to --output-dir/cache/ as JSON files named
``baselines-{dataset}-{hash}.json``.  These files are later read by
collate.py to build the summary table and charts.

This script does NOT import or call anything from promptlearn.

Examples
--------
    # run all baselines for the full dataset suite
    python benchmarks/run_baselines.py

    # only logistic regression on a subset of datasets
    python benchmarks/run_baselines.py --datasets adult credit-g --learners logreg

    # force re-run even if cached results exist
    python benchmarks/run_baselines.py --no-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

load_dotenv()

# Allow running from repo root: python benchmarks/run_baselines.py
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_utils import (
    DEFAULT_DATASETS,
    _baseline_cache_key,
    _rich_metrics,
    _tabpfn_classifier,
    _xgb_classifier,
    load_dataset,
)

logger = logging.getLogger("promptlearn.progression")

ALL_LEARNERS = ["logreg", "xgboost", "tabpfn"]


def run_dataset_baselines(
    dataset: str,
    spec: tuple,
    max_rows: int,
    cache_dir: Path | None,
    learners: list[str] | None = None,
    skip_cache_read: bool = False,
) -> dict:
    """Run the requested baseline learners on one dataset.

    Parameters
    ----------
    dataset:
        Short name used in cache keys and log messages.
    spec:
        ``(openml_name, version)`` tuple from DEFAULT_DATASETS.
    max_rows:
        Cap on training+test rows combined (sampled deterministically).
    cache_dir:
        Directory for JSON cache files.  Pass ``None`` to disable all caching.
    learners:
        Subset of ``["logreg", "xgboost", "tabpfn"]`` to run.  Defaults to
        all three when ``None``.
    skip_cache_read:
        When True, ignore any existing cache file and re-run, but still write
        the new result to cache.  Used by ``--no-cache``.

    Returns
    -------
    dict
        Keys are learner names; values are metric dicts (or ``{"error": ...}``).
        An existing cache file is returned as-is if all requested learners are
        present in it.
    """
    if learners is None:
        learners = list(ALL_LEARNERS)

    cache_file = (
        cache_dir / f"baselines-{dataset}-{_baseline_cache_key(dataset, max_rows)}.json"
        if cache_dir
        else None
    )

    # Return cached file only if every requested learner is present and successful.
    if cache_file and cache_file.exists() and not skip_cache_read:
        with open(cache_file) as f:
            cached = json.load(f)
        # Treat errored entries as missing so they are automatically retried.
        missing = [
            name for name in learners
            if name not in cached or (isinstance(cached[name], dict) and cached[name].get("error"))
        ]
        if not missing:
            logger.info("[%s] baselines cached (all requested learners present)", dataset)
            return cached
        logger.info(
            "[%s] cache hit but missing/errored %s — recomputing those learners", dataset, missing
        )
        result = {k: v for k, v in cached.items() if k not in missing}
        learners = missing  # type: ignore[assignment]
    else:
        result = {}

    openml_name, version = spec
    logger.info("[%s] loading dataset for baselines…", dataset)
    X, y, class_map, _ = load_dataset(openml_name, version, max_rows)
    n_classes = len(class_map)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── logreg ───────────────────────────────────────────────────────────────
    if "logreg" in learners:
        t0 = time.time()
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            cat_cols = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            num_cols = [c for c in X_train.columns if c not in cat_cols]
            transformers = []
            if cat_cols:
                transformers.append(
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="most_frequent")),
                                (
                                    "enc",
                                    OneHotEncoder(
                                        handle_unknown="ignore", sparse_output=False
                                    ),
                                ),
                            ]
                        ),
                        cat_cols,
                    )
                )
            if num_cols:
                transformers.append(
                    (
                        "num",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="mean")),
                                ("scl", StandardScaler()),
                            ]
                        ),
                        num_cols,
                    )
                )
            preproc = ColumnTransformer(transformers, remainder="passthrough")
            lr = Pipeline([("pre", preproc), ("clf", LogisticRegression(max_iter=1000))])
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            y_proba_lr = lr.predict_proba(X_test) if hasattr(lr, "predict_proba") else None
            result["logreg"] = _rich_metrics(
                np.array(y_test), y_pred_lr, y_proba_lr, n_classes
            )
            result["logreg"]["fit_time_s"] = round(time.time() - t0, 2)
            print(
                f"  logreg        {dataset}  accuracy={result['logreg']['accuracy']:.3f}",
                flush=True,
            )
        except Exception as e:
            logger.warning("[%s] logreg failed: %s", dataset, e)
            result["logreg"] = {"error": str(e)}

    # ── xgboost ──────────────────────────────────────────────────────────────
    if "xgboost" in learners:
        xgb = _xgb_classifier()
        if xgb is None:
            logger.warning("[%s] xgboost not installed, skipping", dataset)
        else:
            t0 = time.time()
            try:
                from sklearn.compose import ColumnTransformer
                from sklearn.impute import SimpleImputer
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OrdinalEncoder

                cat_cols = X_train.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()
                num_cols = [c for c in X_train.columns if c not in cat_cols]
                transformers = []
                if cat_cols:
                    transformers.append(
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "enc",
                                        OrdinalEncoder(
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1,
                                        ),
                                    ),
                                ]
                            ),
                            cat_cols,
                        )
                    )
                if num_cols:
                    transformers.append(
                        ("num", SimpleImputer(strategy="mean"), num_cols)
                    )
                preproc = ColumnTransformer(transformers, remainder="passthrough")
                xgb_pipe = Pipeline([("pre", preproc), ("clf", xgb)])
                xgb_pipe.fit(X_train, y_train)
                y_pred_xgb = xgb_pipe.predict(X_test)
                y_proba_xgb = (
                    xgb_pipe.predict_proba(X_test)
                    if hasattr(xgb_pipe, "predict_proba")
                    else None
                )
                result["xgboost"] = _rich_metrics(
                    np.array(y_test), y_pred_xgb, y_proba_xgb, n_classes
                )
                result["xgboost"]["fit_time_s"] = round(time.time() - t0, 2)
                print(
                    f"  xgboost       {dataset}  accuracy={result['xgboost']['accuracy']:.3f}",
                    flush=True,
                )
            except Exception as e:
                logger.warning("[%s] xgboost failed: %s", dataset, e)
                result["xgboost"] = {"error": str(e)}

    # ── tabpfn ───────────────────────────────────────────────────────────────
    if "tabpfn" in learners:
        tabpfn = _tabpfn_classifier()
        if tabpfn is None:
            logger.warning("[%s] tabpfn not installed, skipping", dataset)
        else:
            t0 = time.time()
            try:
                from sklearn.compose import ColumnTransformer
                from sklearn.impute import SimpleImputer
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OrdinalEncoder

                cat_cols = X_train.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()
                num_cols = [c for c in X_train.columns if c not in cat_cols]
                transformers = []
                if cat_cols:
                    transformers.append(
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imp", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "enc",
                                        OrdinalEncoder(
                                            handle_unknown="use_encoded_value",
                                            unknown_value=-1,
                                        ),
                                    ),
                                ]
                            ),
                            cat_cols,
                        )
                    )
                if num_cols:
                    transformers.append(
                        ("num", SimpleImputer(strategy="mean"), num_cols)
                    )
                preproc = ColumnTransformer(transformers, remainder="passthrough")
                tabpfn_pipe = Pipeline([("pre", preproc), ("clf", tabpfn)])
                tabpfn_pipe.fit(X_train, y_train)
                y_pred_tabpfn = tabpfn_pipe.predict(X_test)
                y_proba_tabpfn = (
                    tabpfn_pipe.predict_proba(X_test)
                    if hasattr(tabpfn_pipe, "predict_proba")
                    else None
                )
                result["tabpfn"] = _rich_metrics(
                    np.array(y_test), y_pred_tabpfn, y_proba_tabpfn, n_classes
                )
                result["tabpfn"]["fit_time_s"] = round(time.time() - t0, 2)
                print(
                    f"  tabpfn        {dataset}  accuracy={result['tabpfn']['accuracy']:.3f}",
                    flush=True,
                )
            except Exception as e:
                logger.warning("[%s] tabpfn failed: %s", dataset, e)
                result["tabpfn"] = {"error": str(e)}

    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("[%s] baselines written to %s", dataset, cache_file)

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Dataset keys to run (default: full suite).",
    )
    parser.add_argument(
        "--learners",
        nargs="*",
        default=list(ALL_LEARNERS),
        choices=ALL_LEARNERS,
        metavar="LEARNER",
        help=f"Baseline learners to run. Choices: {ALL_LEARNERS} (default: all).",
    )
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/progression_results",
        help="Directory for cached results (default: benchmarks/progression_results).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cache files and re-run, but still write results to cache.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    unknown_datasets = [d for d in args.datasets if d not in DEFAULT_DATASETS]
    if unknown_datasets:
        logger.warning("Unknown datasets (not in DEFAULT_DATASETS): %s", unknown_datasets)

    unknown_learners = [name for name in args.learners if name not in ALL_LEARNERS]
    if unknown_learners:
        parser.error(f"Unknown learners: {unknown_learners}. Choose from {ALL_LEARNERS}")

    datasets_to_run = [d for d in args.datasets if d in DEFAULT_DATASETS]
    if not datasets_to_run:
        print("No valid datasets to run.")
        return 1

    print(
        f"Running baselines: learners={args.learners}  datasets={datasets_to_run}  "
        f"max_rows={args.max_rows}  cache={cache_dir}{'  (skip-read)' if args.no_cache else ''}",
        flush=True,
    )

    for dataset in datasets_to_run:
        spec = DEFAULT_DATASETS[dataset]
        print(f"\n[{dataset}]", flush=True)
        try:
            run_dataset_baselines(
                dataset,
                spec,
                args.max_rows,
                cache_dir,
                learners=args.learners,
                skip_cache_read=args.no_cache,
            )
        except Exception as e:
            logger.warning("[%s] baselines failed: %s", dataset, e)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
