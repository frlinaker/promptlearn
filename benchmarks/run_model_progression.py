#!/usr/bin/env python
"""Model-progression benchmark: accuracy of promptlearn across OpenAI model generations.

Runs PromptClassifier (FE off) on the curated OpenML suite for each model in
MODEL_PROGRESSION and records per-dataset accuracy alongside logistic regression
and XGBoost baselines. Results are cached per model so reruns are cheap.

The intended output is a timeline chart where:
  - x-axis  = model release date (oldest → newest)
  - y-axis  = mean accuracy across datasets
  - lines   = promptlearn (rising), logreg + xgboost (flat baselines)

Examples
--------
    # full run across all progression models
    python benchmarks/run_model_progression.py

    # subset of models or datasets for a quick smoke-test
    python benchmarks/run_model_progression.py --models gpt-4o gpt-5.5 --datasets adult credit-g

    # force fresh LLM calls (ignore cache)
    python benchmarks/run_model_progression.py --no-cache

    # write metrics JSON + plots
    python benchmarks/run_model_progression.py --output-dir benchmarks/progression_results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from promptlearn import PromptClassifier, PromptFeatureEngineer, compare_models

logger = logging.getLogger("promptlearn.progression")

CACHE_SCHEMA = "progression-v1"

# Ordered oldest → newest. release_date is approximate; used as the x-axis value.
MODEL_PROGRESSION = [
    # OpenAI
    {
        "model_id": "gpt-4o",
        "label": "GPT-4o",
        "release_date": date(2024, 5, 13),
        "family": "GPT-4",
        "provider": "openai",
    },
    {
        "model_id": "gpt-4o-mini",
        "label": "GPT-4o mini",
        "release_date": date(2024, 7, 18),
        "family": "GPT-4",
        "provider": "openai",
    },
    {
        "model_id": "gpt-4.1",
        "label": "GPT-4.1",
        "release_date": date(2025, 4, 14),
        "family": "GPT-4.1",
        "provider": "openai",
    },
    {
        "model_id": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
        "release_date": date(2026, 3, 1),
        "family": "GPT-5",
        "provider": "openai",
    },
    {
        "model_id": "gpt-5.5",
        "label": "GPT-5.5",
        "release_date": date(2026, 4, 1),
        "family": "GPT-5",
        "provider": "openai",
    },
    {
        "model_id": "gpt-5.5+web",
        "base_model_id": "gpt-5.5",
        "label": "GPT-5.5 +web",
        "release_date": date(2026, 4, 1),
        "family": "GPT-5",
        "provider": "openai",
        "web_search": True,
    },
    # Google Gemini (via Vertex AI)
    {
        "model_id": "vertex_ai/gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash Lite",
        "release_date": date(2025, 7, 22),
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
    },
    {
        "model_id": "vertex_ai/gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "release_date": date(2025, 6, 17),
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
    },
    {
        "model_id": "vertex_ai/gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "release_date": date(2025, 6, 17),
        "family": "Gemini 2.5",
        "provider": "google",
        "vertex_region": "us-central1",
    },
    {
        "model_id": "vertex_ai/gemini-3.5-flash",
        "label": "Gemini 3.5 Flash",
        "release_date": date(2026, 5, 19),
        "family": "Gemini 3",
        "provider": "google",
        "vertex_region": "asia-southeast1",
    },
    {
        "model_id": "vertex_ai/gemini-3.5-flash+web",
        "base_model_id": "vertex_ai/gemini-3.5-flash",
        "label": "Gemini 3.5 Flash +web",
        "release_date": date(2026, 5, 19),
        "family": "Gemini 3",
        "provider": "google",
        "vertex_region": "asia-southeast1",
        "web_search": True,
    },
]

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
    "soybean": ("soybean", 1),
    "hepatitis": ("hepatitis", 1),
    "lymph": ("lymph", 1),
}


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=4, verbosity=0
    )


def _tabpfn_classifier():
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return None
    return TabPFNClassifier()


def load_dataset(openml_name: str, version: int, max_rows: int):
    bunch = fetch_openml(
        name=openml_name, version=version, as_frame=True, parser="auto"
    )
    X = bunch.data.copy()
    y = pd.Series(np.asarray(bunch.target)).astype(str)
    classes = {c: i for i, c in enumerate(sorted(y.unique()))}
    y = y.map(classes).astype(int)
    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]
    description = getattr(bunch, "DESCR", None) or ""
    return X.reset_index(drop=True), y.reset_index(drop=True), classes, description


def _rich_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, n_classes: int
) -> dict:
    """Compute a broad set of classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "error_rate": float(1 - accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
                lb = LabelBinarizer().fit(y_true)
                y_bin = lb.transform(y_true)
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return metrics


def _cache_key(
    dataset: str,
    model_id: str,
    max_rows: int,
    fe_model: str | None = None,
    web_search: bool = False,
) -> str:
    raw = f"{CACHE_SCHEMA}|{dataset}|{model_id}|{max_rows}|fe={fe_model or ''}|ws={web_search}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _baseline_cache_key(dataset: str, max_rows: int) -> str:
    raw = f"{CACHE_SCHEMA}|baselines|{dataset}|{max_rows}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def run_dataset_baselines(
    dataset: str,
    spec: tuple,
    max_rows: int,
    cache_dir: Path | None,
) -> dict:
    """Run logreg, xgboost, and TabPFN once per dataset (model-independent)."""
    cache_file = (
        cache_dir / f"baselines-{dataset}-{_baseline_cache_key(dataset, max_rows)}.json"
        if cache_dir
        else None
    )
    if cache_file and cache_file.exists():
        logger.info("[%s] baselines cached", dataset)
        with open(cache_file) as f:
            return json.load(f)

    openml_name, version = spec
    logger.info("[%s] running baselines…", dataset)
    X, y, class_map, _ = load_dataset(openml_name, version, max_rows)
    n_classes = len(class_map)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    result: dict = {}

    # logreg
    t0 = time.time()
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

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
    except Exception as e:
        logger.warning("[%s] logreg failed: %s", dataset, e)
        result["logreg"] = {"error": str(e)}

    # xgboost
    xgb = _xgb_classifier()
    if xgb is not None:
        t0 = time.time()
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OrdinalEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer

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
                transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
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
        except Exception as e:
            logger.warning("[%s] xgboost failed: %s", dataset, e)
            result["xgboost"] = {"error": str(e)}

    # tabpfn
    tabpfn = _tabpfn_classifier()
    if tabpfn is not None:
        t0 = time.time()
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OrdinalEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer

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
                transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
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
        except Exception as e:
            logger.warning("[%s] tabpfn failed: %s", dataset, e)
            result["tabpfn"] = {"error": str(e)}

    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


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
) -> dict:
    """Run one (dataset, model) cell. Returns a metrics dict."""
    # For web-search variants the model_id is a synthetic key (e.g. "gpt-5.5+web");
    # the actual LLM call uses base_model_id when provided, else strips the +web suffix.
    actual_model_id = base_model_id or (
        model_id.removesuffix("+web") if web_search else model_id
    )

    # Per-model region override for Vertex AI (some models only exist in specific regions).
    _region_override = None
    if vertex_region:
        current = os.environ.get("VERTEXAI_LOCATION", "")
        if current != vertex_region:
            os.environ["VERTEXAI_LOCATION"] = vertex_region
            _region_override = current  # remember to restore after

    cache_file = (
        cache_dir
        / f"{dataset}-{model_id.replace('/', '-')}-{_cache_key(dataset, model_id, max_rows, fe_model=fe_model, web_search=web_search)}.json"
        if cache_dir
        else None
    )
    if cache_file and cache_file.exists():
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

    # When fe_model is set, use that LLM to engineer features before all learners run.
    if fe_model:
        try:
            fe_step = PromptFeatureEngineer(model=fe_model, verbose=False)
            X_train = fe_step.fit_transform(X_train, y_train)
            X_test = fe_step.transform(X_test)
            logger.info(
                "[%s] FE (%s) produced %d cols", dataset, fe_model, X_train.shape[1]
            )
        except Exception as e:
            logger.warning(
                "[%s] FE (%s) failed, using original features: %s", dataset, fe_model, e
            )

    # --- promptlearn ---
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

    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    if _region_override is not None:
        os.environ["VERTEXAI_LOCATION"] = _region_override

    return result


def load_all_results(output_dir: Path) -> list[dict]:
    results = []
    for f in sorted(output_dir.glob("*.json")):
        if f.name == "metrics_all.json":
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def build_summary_df(results: list[dict]) -> pd.DataFrame:
    """Long-form DataFrame: one row per (dataset, learner) with all metrics.

    promptlearn learner names are qualified as "promptlearn[<llm-label>]" so they
    are never confused with the LLM model dimension.  Baseline learners
    (logreg, xgboost, tabpfn) appear once per dataset with no LLM association.
    """
    rows = []
    model_meta = {m["model_id"]: m for m in MODEL_PROGRESSION}
    seen_baselines: set[tuple] = set()  # (dataset, learner) — emit baselines once

    for r in results:
        if "dataset" not in r or "model_id" not in r:
            continue  # skip baseline-only cache files
        dataset = r["dataset"]
        model_id = r["model_id"]
        meta = model_meta.get(model_id, {})
        llm_label = meta.get("label", model_id)

        # promptlearn — qualified name, carries LLM metadata
        if "promptlearn" in r and "error" not in r["promptlearn"]:
            m = r["promptlearn"]
            web_search = meta.get("web_search", False)
            row = {
                "dataset": dataset,
                "model_id": model_id,
                "llm_label": llm_label,
                "release_date": str(meta.get("release_date", "")),
                "family": meta.get("family", ""),
                "provider": meta.get("provider", "openai"),
                "web_search": web_search,
                "learner": f"promptlearn[{llm_label}]",
                "n_rows": r.get("n_rows"),
                "n_cols": r.get("n_cols"),
                "n_classes": r.get("n_classes"),
            }
            row.update({k: v for k, v in m.items() if k not in ("fit_time_s",)})
            row["fit_time_s"] = m.get("fit_time_s")
            rows.append(row)

        # baselines — emit once per (dataset, learner)
        for learner in ("logreg", "xgboost", "tabpfn"):
            key = (dataset, learner)
            if key in seen_baselines:
                continue
            if learner not in r or "error" in r[learner]:
                continue
            seen_baselines.add(key)
            m = r[learner]
            row = {
                "dataset": dataset,
                "model_id": None,
                "llm_label": None,
                "release_date": None,
                "family": None,
                "provider": None,
                "learner": learner,
                "n_rows": r.get("n_rows"),
                "n_cols": r.get("n_cols"),
                "n_classes": r.get("n_classes"),
            }
            row.update({k: v for k, v in m.items() if k not in ("fit_time_s",)})
            row["fit_time_s"] = m.get("fit_time_s")
            rows.append(row)

    return pd.DataFrame(rows)


def plot_progression(df: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── shared prep ──────────────────────────────────────────────────────────
    # promptlearn rows have a release_date; baseline rows do not.
    pl_df = df[df["learner"].str.startswith("promptlearn[")].copy()
    pl_df["release_date"] = pd.to_datetime(pl_df["release_date"])

    if "web_search" not in pl_df.columns:
        pl_df["web_search"] = False
    pl_df["web_search"] = pl_df["web_search"].fillna(False)

    pl_summary = (
        pl_df.groupby(
            [
                "model_id",
                "llm_label",
                "release_date",
                "learner",
                "provider",
                "web_search",
            ]
        )["accuracy"]
        .mean()
        .reset_index()
        .sort_values("release_date")
    )

    pl_data = pl_summary.copy()

    lr_data = df[df["learner"] == "logreg"]
    xgb_data = df[df["learner"] == "xgboost"]
    tabpfn_data = df[df["learner"] == "tabpfn"]

    n_datasets = df["dataset"].nunique()

    # ── 1. Timeline: mean accuracy vs model release date ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    if not lr_data.empty:
        lr_mean = lr_data["accuracy"].mean()
        ax.axhline(
            lr_mean,
            color="#4878CF",
            linewidth=1.8,
            linestyle="--",
            label=f"Logistic Regression  ({lr_mean:.3f})",
        )

    if not xgb_data.empty:
        xgb_mean = xgb_data["accuracy"].mean()
        ax.axhline(
            xgb_mean,
            color="#6ACC65",
            linewidth=1.8,
            linestyle="--",
            label=f"XGBoost  ({xgb_mean:.3f})",
        )

    if not tabpfn_data.empty:
        tabpfn_mean = tabpfn_data["accuracy"].mean()
        ax.axhline(
            tabpfn_mean,
            color="#FF7F0E",
            linewidth=1.8,
            linestyle="--",
            label=f"TabPFN  ({tabpfn_mean:.3f})",
        )

    # promptlearn — one line per provider, colour-coded.
    # Web-search variants are plotted as star markers offset above their base model.
    # Use cumulative-max envelope so lite/weaker models don't cause visual dips.
    provider_styles = {
        "openai": {"color": "#D65F5F", "marker": "o", "label": "promptlearn / OpenAI"},
        "google": {
            "color": "#4285F4",
            "marker": "s",
            "label": "promptlearn / Google Gemini",
        },
    }
    if "web_search" not in pl_data.columns:
        pl_data["web_search"] = False

    if not pl_data.empty:
        # Split standard vs web-search rows
        pl_standard = pl_data[~pl_data["web_search"].fillna(False)].copy()
        pl_web = pl_data[pl_data["web_search"].fillna(False)].copy()

        for provider, grp in pl_standard.groupby("provider"):
            grp = grp.sort_values("release_date").reset_index(drop=True)
            style = provider_styles.get(
                provider,
                {"color": "#999", "marker": "o", "label": f"promptlearn / {provider}"},
            )
            color = style["color"]

            # Cumulative-max envelope line (the "best so far" trajectory).
            grp["best_so_far"] = grp["accuracy"].cummax()
            ax.plot(
                grp["release_date"],
                grp["best_so_far"],
                color=color,
                linewidth=2.5,
                linestyle="-",
                label=style["label"],
                zorder=3,
            )

            # Individual model dots — weaker models shown slightly faded.
            for _, row in grp.iterrows():
                is_best = abs(row["accuracy"] - row["best_so_far"]) < 1e-9
                alpha = 1.0 if is_best else 0.55
                ax.scatter(
                    row["release_date"],
                    row["accuracy"],
                    marker=style["marker"],
                    s=80,
                    color=color,
                    alpha=alpha,
                    zorder=4,
                )
                ax.annotate(
                    f"{row['accuracy']:.3f}\n{row['llm_label']}",
                    xy=(row["release_date"], row["accuracy"]),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=color,
                    alpha=alpha,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=alpha),
                )

        # Web-search variants: star markers, lighter shade, dotted vertical connector
        # to their same-date base model point.
        _web_legend_added: set[str] = set()
        for _, row in pl_web.iterrows():
            provider = row["provider"]
            style = provider_styles.get(
                provider,
                {"color": "#999", "marker": "o", "label": f"promptlearn / {provider}"},
            )
            color = style["color"]
            legend_key = f"{provider}_web"
            label = (
                f"promptlearn / {provider.title()} +web search"
                if legend_key not in _web_legend_added
                else "_nolegend_"
            )
            _web_legend_added.add(legend_key)
            ax.scatter(
                row["release_date"],
                row["accuracy"],
                marker="*",
                s=220,
                color=color,
                alpha=0.9,
                zorder=5,
                label=label,
            )
            # Dotted vertical line connecting to base model point (same date)
            base_rows = pl_standard[
                (pl_standard["provider"] == provider)
                & (pl_standard["release_date"] == row["release_date"])
            ]
            if not base_rows.empty:
                base_acc = base_rows["accuracy"].iloc[0]
                ax.plot(
                    [row["release_date"], row["release_date"]],
                    [base_acc, row["accuracy"]],
                    color=color,
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.7,
                    zorder=3,
                )
            ax.annotate(
                f"{row['accuracy']:.3f}\n{row['llm_label']}",
                xy=(row["release_date"], row["accuracy"]),
                xytext=(30, 0),
                textcoords="offset points",
                ha="left",
                fontsize=8.5,
                color=color,
                alpha=0.9,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
            )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    fig.autofmt_xdate(rotation=30)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    all_acc = pl_data["accuracy"]
    ax.set_ylim(max(0.0, all_acc.min() - 0.08), min(1.02, all_acc.max() + 0.12))
    ax.set_xlabel("Model release date", fontsize=12)
    ax.set_ylabel(f"Mean accuracy ({n_datasets} OpenML datasets)", fontsize=12)
    ax.set_title(
        "promptlearn accuracy grows with LLM evolution\n"
        "Classical ML baselines shown as dashed horizontals",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    out = output_dir / "model_progression.png"
    fig.savefig(out, dpi=150)
    logger.info("Saved timeline chart → %s", out)
    plt.close(fig)

    # ── 2. Per-dataset heatmap: datasets × LLM models, promptlearn accuracy ──
    # Order columns by release date.
    col_order = (
        pl_data.sort_values("release_date")["llm_label"].tolist()
        if not pl_data.empty
        else None
    )
    pl_pivot = pl_df.pivot_table(
        index="dataset", columns="llm_label", values="accuracy"
    )
    if col_order:
        pl_pivot = pl_pivot.reindex(
            columns=[c for c in col_order if c in pl_pivot.columns]
        )
    # Sort rows by mean accuracy ascending so weakest datasets sit at the top.
    pl_pivot = pl_pivot.loc[pl_pivot.mean(axis=1).sort_values().index]

    if not pl_pivot.empty:
        fig2, ax2 = plt.subplots(
            figsize=(
                max(8, len(pl_pivot.columns) * 1.8),
                max(5, len(pl_pivot) * 0.7 + 1.5),
            )
        )
        sns.heatmap(
            pl_pivot,
            ax=ax2,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Accuracy", "shrink": 0.8},
        )
        ax2.set_title("promptlearn accuracy per dataset × model", fontsize=12, pad=12)
        ax2.set_xlabel("")
        ax2.set_ylabel("")
        ax2.tick_params(axis="x", rotation=30)
        ax2.tick_params(axis="y", rotation=0)
        fig2.tight_layout()
        out2 = output_dir / "per_dataset_heatmap.png"
        fig2.savefig(out2, dpi=150)
        logger.info("Saved heatmap → %s", out2)
        plt.close(fig2)

    # ── 3. All-learner bar chart: promptlearn bars per LLM + baseline lines ──
    if not pl_data.empty:
        pl_bar = (
            pl_data.groupby(["llm_label", "release_date"])["accuracy"]
            .mean()
            .reset_index()
            .sort_values("release_date")
        )
        llm_labels = pl_bar["llm_label"].tolist()
        n_models = len(llm_labels)
        x = np.arange(n_models)

        fig3, ax3 = plt.subplots(figsize=(max(9, n_models * 1.4), 6))
        bars = ax3.bar(
            x, pl_bar["accuracy"], color="#D65F5F", label="promptlearn", zorder=3
        )
        for bar, val in zip(bars, pl_bar["accuracy"]):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Baselines as horizontal lines across all bars.
        baseline_styles = [
            ("logreg", lr_data, "#4878CF", "Logistic Regression"),
            ("xgboost", xgb_data, "#6ACC65", "XGBoost"),
            ("tabpfn", tabpfn_data, "#FF7F0E", "TabPFN"),
        ]
        for _, bdata, color, label in baseline_styles:
            if not bdata.empty:
                val = bdata["accuracy"].mean()
                ax3.axhline(
                    val,
                    color=color,
                    linewidth=1.8,
                    linestyle="--",
                    label=f"{label}  ({val:.3f})",
                    zorder=4,
                )

        ax3.set_xticks(x)
        ax3.set_xticklabels(llm_labels, rotation=25, ha="right", fontsize=10)
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax3.set_ylim(0, 1.12)
        ax3.set_xlabel("LLM model (oldest → newest)", fontsize=12)
        ax3.set_ylabel(f"Mean accuracy ({n_datasets} datasets)", fontsize=12)
        ax3.set_title(
            "promptlearn vs baselines: mean accuracy by LLM generation", fontsize=13
        )
        ax3.legend(fontsize=10)
        ax3.grid(axis="y", alpha=0.4)
        fig3.tight_layout()
        out3 = output_dir / "all_learners_bar.png"
        fig3.savefig(out3, dpi=150)
        logger.info("Saved grouped bar chart → %s", out3)
        plt.close(fig3)

    # ── 4. Gap-to-baseline chart: promptlearn vs best baseline per LLM ───────
    if not pl_data.empty:
        best_baseline_acc = max(
            (
                bdata["accuracy"].mean()
                for bdata in [lr_data, xgb_data, tabpfn_data]
                if not bdata.empty
            ),
            default=None,
        )
        if best_baseline_acc is not None:
            pl_bar2 = (
                pl_data.groupby(["llm_label", "release_date"])["accuracy"]
                .mean()
                .reset_index()
                .sort_values("release_date")
            )
            pl_bar2["gap"] = pl_bar2["accuracy"] - best_baseline_acc

            fig4, ax4 = plt.subplots(figsize=(12, 5))
            colors = ["#D65F5F" if g >= 0 else "#4878CF" for g in pl_bar2["gap"]]
            ax4.bar(pl_bar2["llm_label"], pl_bar2["gap"], color=colors, zorder=3)
            ax4.axhline(0, color="black", linewidth=1.0)
            for _, row in pl_bar2.iterrows():
                ax4.text(
                    row["llm_label"],
                    row["gap"] + (0.004 if row["gap"] >= 0 else -0.008),
                    f"{row['gap']:+.3f}",
                    ha="center",
                    va="bottom" if row["gap"] >= 0 else "top",
                    fontsize=9,
                )
            ax4.set_xlabel("LLM model (oldest → newest)", fontsize=12)
            ax4.set_ylabel("Accuracy gap vs best baseline", fontsize=12)
            ax4.set_title(
                "promptlearn gap to best baseline (logreg / XGBoost / TabPFN)\n"
                "Red = above baseline  ·  Blue = below baseline",
                fontsize=12,
            )
            ax4.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1.0, decimals=1)
            )
            ax4.tick_params(axis="x", rotation=25)
            ax4.grid(axis="y", alpha=0.4)
            fig4.tight_layout()
            out4 = output_dir / "gap_to_baseline.png"
            fig4.savefig(out4, dpi=150)
            logger.info("Saved gap chart → %s", out4)
            plt.close(fig4)

    # ── 5. Per-dataset timelines ──────────────────────────────────────────────
    datasets = sorted(df["dataset"].unique())
    if datasets and not pl_data.empty:
        ncols = 2
        nrows = (len(datasets) + ncols - 1) // ncols
        fig5, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 6, nrows * 3.8), squeeze=False
        )

        # Baseline means are dataset-specific here (not cross-dataset).
        for idx, dataset in enumerate(datasets):
            ax = axes[idx // ncols][idx % ncols]
            ds_df = df[df["dataset"] == dataset].copy()
            ds_df["release_date"] = pd.to_datetime(ds_df["release_date"])
            ds_df = ds_df.sort_values("release_date")

            for learner, color, ls, lw in [
                ("logreg", "#4878CF", "--", 1.5),
                ("xgboost", "#6ACC65", "--", 1.5),
                ("tabpfn", "#FF7F0E", "--", 1.5),
            ]:
                ld = ds_df[ds_df["learner"] == learner]
                if ld.empty:
                    continue
                val = ld["accuracy"].mean()
                pl_dates = ds_df[ds_df["learner"].str.startswith("promptlearn[")][
                    "release_date"
                ].dropna()
                if pl_dates.empty:
                    continue
                x_min = pl_dates.min()
                x_max = pl_dates.max()
                ax.plot(
                    [x_min, x_max],
                    [val, val],
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
                    label=f"{learner} ({val:.3f})",
                )

            # promptlearn — one envelope line per provider, same logic as chart 1.
            pl_ds = ds_df[ds_df["learner"].str.startswith("promptlearn[")].copy()
            pl_ds["release_date"] = pd.to_datetime(pl_ds["release_date"])
            ds_provider_styles = {
                "openai": {"color": "#D65F5F", "marker": "o", "label": "OpenAI"},
                "google": {"color": "#4285F4", "marker": "s", "label": "Gemini"},
            }
            for provider, grp in pl_ds.groupby("provider"):
                grp = grp.sort_values("release_date").reset_index(drop=True)
                pstyle = ds_provider_styles.get(
                    provider, {"color": "#999", "marker": "o", "label": provider}
                )
                grp["best_so_far"] = grp["accuracy"].cummax()
                ax.plot(
                    grp["release_date"],
                    grp["best_so_far"],
                    color=pstyle["color"],
                    linewidth=2.2,
                    linestyle="-",
                    label=f"promptlearn/{pstyle['label']}",
                    zorder=3,
                )
                for _, row in grp.iterrows():
                    is_best = abs(row["accuracy"] - row["best_so_far"]) < 1e-9
                    alpha = 1.0 if is_best else 0.45
                    ax.scatter(
                        row["release_date"],
                        row["accuracy"],
                        marker=pstyle["marker"],
                        s=40,
                        color=pstyle["color"],
                        alpha=alpha,
                        zorder=4,
                    )
                    ax.annotate(
                        f"{row['accuracy']:.3f}",
                        xy=(row["release_date"], row["accuracy"]),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=pstyle["color"],
                        alpha=alpha,
                    )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
            ds_acc = ds_df["accuracy"]
            ax.set_ylim(max(0.0, ds_acc.min() - 0.1), min(1.05, ds_acc.max() + 0.15))
            ax.set_title(dataset, fontsize=11, fontweight="bold")
            ax.legend(fontsize=7.5, loc="lower right")
            ax.tick_params(axis="x", rotation=25, labelsize=7.5)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots.
        for idx in range(len(datasets), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig5.suptitle(
            "promptlearn accuracy per dataset across model generations\n"
            "Dashed = classical ML baselines",
            fontsize=13,
            y=1.01,
        )
        fig5.tight_layout()
        out5 = output_dir / "per_dataset_timelines.png"
        fig5.savefig(out5, dpi=150, bbox_inches="tight")
        logger.info("Saved per-dataset timelines → %s", out5)
        plt.close(fig5)


def print_summary_table(df: pd.DataFrame):
    print("\n## Model progression — mean metrics across datasets\n")

    # promptlearn — one row per LLM, sorted by release date
    pl_rows = df[df["learner"].str.startswith("promptlearn[")].copy()
    if not pl_rows.empty:
        pl_rows["release_date"] = pd.to_datetime(pl_rows["release_date"])
        pl_summary = (
            pl_rows.groupby(["llm_label", "release_date"])
            .agg(
                accuracy=("accuracy", "mean"),
                balanced_accuracy=("balanced_accuracy", "mean"),
                f1_macro=("f1_macro", "mean"),
                n_datasets=("dataset", "count"),
            )
            .reset_index()
            .sort_values("release_date")
        )
        print("### promptlearn")
        print(
            pl_summary[
                ["llm_label", "accuracy", "balanced_accuracy", "f1_macro", "n_datasets"]
            ].to_string(index=False, float_format="%.3f")
        )
        print()

    # baselines — one row each
    for learner in ("logreg", "xgboost", "tabpfn"):
        rows = df[df["learner"] == learner]
        if rows.empty:
            continue
        acc = rows["accuracy"].mean()
        bal = rows["balanced_accuracy"].mean()
        f1 = rows["f1_macro"].mean()
        n = rows["dataset"].nunique()
        print(
            f"### {learner}  (accuracy={acc:.3f}  balanced={bal:.3f}  f1={f1:.3f}  n_datasets={n})"
        )
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=[m["model_id"] for m in MODEL_PROGRESSION],
        help="Model IDs to run (default: full progression list).",
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
        help="Directory for cached results, metrics JSON, and plots.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--fe-model",
        default=None,
        metavar="MODEL_ID",
        help="LLM to use for PromptFeatureEngineer (e.g. gpt-5.5). Applied before all learners. Omit to disable FE.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.getLogger("promptlearn").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    output_dir = Path(args.output_dir)
    cache_dir = None if args.no_cache else output_dir / "cache"

    model_lookup = {m["model_id"]: m for m in MODEL_PROGRESSION}
    models_to_run = [m for m in args.models if m in model_lookup]
    unknown = [m for m in args.models if m not in model_lookup]
    if unknown:
        logger.warning("Unknown model IDs (not in MODEL_PROGRESSION): %s", unknown)

    # Compute baselines once per dataset (model-independent).
    baselines: dict[str, dict] = {}
    for dataset in args.datasets:
        spec = DEFAULT_DATASETS.get(dataset)
        if spec is None:
            continue
        try:
            baselines[dataset] = run_dataset_baselines(
                dataset, spec, args.max_rows, cache_dir
            )
        except Exception as e:
            logger.warning("[%s] baselines failed: %s", dataset, e)
            baselines[dataset] = {}

    all_results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_all.json"

    def _flush_results():
        """Write metrics JSON + CSV and regenerate chart from current results."""
        if not all_results:
            return
        with open(metrics_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        df = build_summary_df(all_results)
        csv_path = output_dir / "metrics_all.csv"
        df.to_csv(csv_path, index=False)
        try:
            plot_progression(df, output_dir)
        except Exception as e:
            logger.warning("Chart update failed: %s", e)

    for model_id in models_to_run:
        meta = model_lookup[model_id]
        vertex_region = meta.get("vertex_region")
        web_search = meta.get("web_search", False)
        base_model_id = meta.get("base_model_id")
        label = meta.get("label", model_id)
        datasets_done = 0
        for dataset in args.datasets:
            spec = DEFAULT_DATASETS.get(dataset)
            if spec is None:
                logger.warning("Unknown dataset %r, skipping", dataset)
                continue
            try:
                r = run_dataset_model(
                    dataset,
                    spec,
                    model_id,
                    args.max_rows,
                    cache_dir,
                    vertex_region=vertex_region,
                    fe_model=args.fe_model,
                    web_search=web_search,
                    base_model_id=base_model_id,
                )
                # Merge baseline results into this row for summary/plotting.
                r.update(baselines.get(dataset, {}))
                all_results.append(r)
                datasets_done += 1
                pl = r.get("promptlearn", {})
                pl_acc = pl.get("accuracy", float("nan"))
                pl_err = pl.get("error")
                status = f"accuracy={pl_acc:.3f}" if not pl_err else f"FAILED: {pl_err}"
                print(
                    f"  promptlearn[{label}]  {dataset}  {status}",
                    flush=True,
                )
                # Incremental flush: update outputs after every dataset so the
                # user can inspect partial results and abort if numbers look bad.
                _flush_results()
                print(
                    f"  [chart + metrics updated — {datasets_done}/{len(args.datasets)} datasets done for {label}]",
                    flush=True,
                )
            except Exception as e:
                logger.warning("[%s × %s] failed: %s", dataset, model_id, e)

    if not all_results:
        print("No results.")
        return 1

    # Final flush (no-op if last dataset already triggered it, but ensures
    # the files are always in sync at exit).
    _flush_results()
    logger.info("Saved all metrics → %s", metrics_path)

    df = build_summary_df(all_results)
    print_summary_table(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
