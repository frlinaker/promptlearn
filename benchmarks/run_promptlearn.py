#!/usr/bin/env python
"""Run promptlearn on the curated OpenML suite for one LLM model.

Results are written to --output-dir/cache/ as JSON files named
``{dataset}-{model_id}-{hash}.json``.  These files are later read by
collate.py to build the summary table and charts.

This script does NOT run any baselines.  Use run_baselines.py for those.

Examples
--------
    # run GPT-5.5 on the full suite
    python benchmarks/run_promptlearn.py --llm gpt-5.5

    # run GPT-5.5 with web search enabled on two datasets
    python benchmarks/run_promptlearn.py --llm gpt-5.5+web --datasets adult credit-g

    # use AdaptiveFeatureEngineer with a different model
    python benchmarks/run_promptlearn.py --llm gpt-5.5 --fe-model gpt-5.4-mini

    # see all available model IDs
    python benchmarks/run_promptlearn.py --list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# benchmark_utils must be importable; add benchmarks/ to sys.path when running
# from repo root so `import benchmark_utils` works without installing the package.
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_utils import (
    DEFAULT_DATASETS,
    MODEL_PROGRESSION,
    _cache_key,
    _rich_metrics,
    load_dataset,
)

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger("promptlearn.progression")

_MODEL_LOOKUP = {m["model_id"]: m for m in MODEL_PROGRESSION}


def run_dataset_model(
    dataset: str,
    spec: tuple,
    model_id: str,
    max_rows: int,
    cache_dir: Path | None,
    vertex_region: str | None = None,
    fe_model: str | None = None,
    web_search: bool = False,
    base_model_id: str | None = None,
    skip_cache_read: bool = False,
) -> dict:
    """Run promptlearn on one (dataset, model) cell.

    Parameters
    ----------
    dataset:
        Short name used in cache keys and log messages.
    spec:
        ``(openml_name, version)`` tuple from DEFAULT_DATASETS.
    model_id:
        The canonical model key from MODEL_PROGRESSION (e.g. ``"gpt-5.5+web"``).
    max_rows:
        Cap on training+test rows combined (sampled deterministically).
    cache_dir:
        Directory for JSON cache files.  Pass ``None`` to disable all caching.
    vertex_region:
        Overrides ``VERTEXAI_LOCATION`` env var for this call (restored afterwards).
    fe_model:
        LLM model id for AdaptiveFeatureEngineer.  ``None`` disables FE.
    web_search:
        Pass ``web_search=True`` to PromptClassifier.fit().
    base_model_id:
        Actual LLM model id when ``model_id`` is a synthetic key like
        ``"gpt-5.5+web"``.  Stripped from model_id when ``None`` and
        ``web_search`` is True.
    skip_cache_read:
        When True, ignore any existing cache file and re-run, but still write
        the new result to cache.  Used by ``--no-cache``.

    Returns
    -------
    dict
        Metrics dict with keys ``dataset``, ``model_id``, ``promptlearn``, etc.
    """
    # For web-search variants the model_id is a synthetic key (e.g. "gpt-5.5+web");
    # the actual LLM call uses base_model_id when provided, else strips the +web suffix.
    actual_model_id = base_model_id or (
        model_id.removesuffix("+web") if web_search else model_id
    )

    # Per-model region override for Vertex AI.
    _region_override = None
    if vertex_region:
        current = os.environ.get("VERTEXAI_LOCATION", "")
        if current != vertex_region:
            os.environ["VERTEXAI_LOCATION"] = vertex_region
            _region_override = current

    safe_model_id = model_id.replace("/", "-")
    cache_file = (
        cache_dir
        / f"{dataset}-{safe_model_id}-{_cache_key(dataset, model_id, max_rows, fe_model=fe_model, web_search=web_search)}.json"
        if cache_dir
        else None
    )
    if cache_file and cache_file.exists() and not skip_cache_read:
        logger.info("[%s × %s] cached", dataset, model_id)
        with open(cache_file) as f:
            return json.load(f)

    openml_name, version = spec
    logger.info("[%s × %s] loading dataset…", dataset, model_id)
    X, y, class_map, description = load_dataset(openml_name, version, max_rows)
    n_classes = len(class_map)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logger.info(
        "[%s] %d rows, %d cols, %d classes", dataset, len(X), X.shape[1], n_classes
    )

    result = {
        "dataset": dataset,
        "model_id": model_id,
        "n_rows": len(X),
        "n_cols": X.shape[1],
        "n_classes": n_classes,
        "class_map": class_map,
    }

    # Lazy import — only run_promptlearn.py touches promptlearn.
    from promptlearn import AdaptiveFeatureEngineer, PromptClassifier

    # Optional feature engineering pass before the classifier.
    if fe_model:
        try:
            fe_step = AdaptiveFeatureEngineer(model=fe_model, verbose=False)
            X_train = fe_step.fit_transform(X_train, y_train)
            X_test = fe_step.transform(X_test)
            skip = getattr(fe_step, "skip_reason_", None)
            if skip:
                logger.info("[%s] AdaptiveFE skipped (%s)", dataset, skip)
            else:
                logger.info(
                    "[%s] AdaptiveFE (%s) produced %d cols (delta=%.3f)",
                    dataset,
                    fe_model,
                    X_train.shape[1],
                    getattr(fe_step, "probe_delta_", float("nan")),
                )
        except Exception as e:
            logger.warning(
                "[%s] AdaptiveFE (%s) failed, using original features: %s",
                dataset,
                fe_model,
                e,
            )

    # ── promptlearn ───────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        clf = PromptClassifier(
            model=actual_model_id, verbose=False, web_search=web_search
        )
        clf.fit(X_train, y_train, dataset_description=description or None)
        y_pred = clf.predict(X_test)
        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_test)
            except Exception:
                pass
        result["promptlearn"] = _rich_metrics(
            np.array(y_test), y_pred, y_proba, n_classes
        )
        result["promptlearn"]["fit_time_s"] = round(time.time() - t0, 2)
        result["promptlearn"]["generated_code"] = clf.raw_python_code_
        result["promptlearn"]["fit_prompt"] = getattr(clf, "fit_prompt_", None)
        logger.info(
            "[%s × %s] promptlearn accuracy=%.3f",
            dataset,
            model_id,
            result["promptlearn"]["accuracy"],
        )
    except Exception as e:
        logger.warning("[%s × %s] promptlearn failed: %s", dataset, model_id, e)
        result["promptlearn"] = {"error": str(e)}

    # Only cache successful results — errors are not cached so re-running the
    # script automatically retries failed datasets without manual cache cleanup.
    pl = result.get("promptlearn", {})
    if cache_file and not (isinstance(pl, dict) and pl.get("error")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    if _region_override is not None:
        os.environ["VERTEXAI_LOCATION"] = _region_override

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        metavar="MODEL_ID",
        help=(
            "LLM model_id from MODEL_PROGRESSION to run "
            "(e.g. gpt-5.5, gpt-5.5+web, vertex_ai/gemini-2.5-pro).  "
            "Use --list-models to see all valid values."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all valid --llm values and exit.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Dataset keys to run (default: full suite).",
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
    parser.add_argument(
        "--fe-model",
        default=None,
        metavar="MODEL_ID",
        help=(
            "LLM to use for AdaptiveFeatureEngineer (e.g. gpt-5.5). "
            "Applied before PromptClassifier.  Omit to disable FE."
        ),
    )
    args = parser.parse_args(argv)

    if args.list_models:
        print("Valid --llm values (MODEL_PROGRESSION):")
        for m in MODEL_PROGRESSION:
            ws = " [+web]" if m.get("web_search") else ""
            print(f"  {m['model_id']:<45}  {m['label']}{ws}")
        return 0

    if not args.llm:
        parser.error("--llm is required (use --list-models to see valid values)")

    if args.llm not in _MODEL_LOOKUP:
        valid = [m["model_id"] for m in MODEL_PROGRESSION]
        parser.error(
            f"Unknown --llm value {args.llm!r}.\n"
            f"Valid model IDs: {valid}\n"
            f"Use --list-models to see all options."
        )

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("promptlearn").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    meta = _MODEL_LOOKUP[args.llm]
    vertex_region = meta.get("vertex_region")
    web_search = meta.get("web_search", False)
    base_model_id = meta.get("base_model_id")
    label = meta.get("label", args.llm)

    unknown_datasets = [d for d in args.datasets if d not in DEFAULT_DATASETS]
    if unknown_datasets:
        logger.warning("Unknown datasets (not in DEFAULT_DATASETS): %s", unknown_datasets)
    datasets_to_run = [d for d in args.datasets if d in DEFAULT_DATASETS]

    if not datasets_to_run:
        print("No valid datasets to run.")
        return 1

    print(
        f"Running promptlearn  model={label!r}  datasets={datasets_to_run}  "
        f"max_rows={args.max_rows}  fe_model={args.fe_model or 'none'}  "
        f"cache={cache_dir}{'  (skip-read)' if args.no_cache else ''}",
        flush=True,
    )

    datasets_done = 0
    for dataset in datasets_to_run:
        spec = DEFAULT_DATASETS[dataset]
        print(f"\n[{dataset} × {label}]", flush=True)
        try:
            r = run_dataset_model(
                dataset,
                spec,
                args.llm,
                args.max_rows,
                cache_dir,
                vertex_region=vertex_region,
                fe_model=args.fe_model,
                web_search=web_search,
                base_model_id=base_model_id,
                skip_cache_read=args.no_cache,
            )
            datasets_done += 1
            pl = r.get("promptlearn", {})
            pl_acc = pl.get("accuracy", float("nan"))
            pl_err = pl.get("error")
            status = f"accuracy={pl_acc:.3f}" if not pl_err else f"FAILED: {pl_err}"
            print(
                f"  promptlearn[{label}]  {dataset}  {status}  "
                f"({datasets_done}/{len(datasets_to_run)} done)",
                flush=True,
            )
        except Exception as e:
            logger.warning("[%s × %s] failed: %s", dataset, args.llm, e)

    print(
        f"\nFinished: {datasets_done}/{len(datasets_to_run)} datasets completed for {label!r}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
