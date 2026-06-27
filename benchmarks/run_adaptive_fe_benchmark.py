"""Benchmark AdaptiveFeatureEngineer across the 13 OpenML datasets.

For each dataset:
  1. Load + split (same split as the progression benchmark)
  2. Fit AdaptiveFeatureEngineer(model=FE_MODEL) on X_train
  3. Evaluate logreg and xgboost on X_train/X_test — both with and without AFE
  4. Record skip_reason_ and stats_ so we can verify the skip logic fired correctly

Writes results to benchmarks/progression_results_afe/metrics_afe.json
and generates fe_per_dataset_lift_afe.png next to it.

Usage
-----
    GOOGLE_APPLICATION_CREDENTIALS=... VERTEXAI_PROJECT=... VERTEXAI_LOCATION=us-central1 \\
    .venv/bin/python3 benchmarks/run_adaptive_fe_benchmark.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from promptlearn import AdaptiveFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("afe_benchmark")

FE_MODEL = "gpt-5.5"
MAX_ROWS = 2000
OUT_DIR = Path("benchmarks/progression_results_afe")

DATASETS = {
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

ABSTRACT_DATASETS = {"tic-tac-toe", "kr-vs-kp", "monks-2"}


def _make_classical_pipeline(X: pd.DataFrame, clf):
    cat_cols = [
        c
        for c in X.columns
        if X[c].dtype == object
        or str(X[c].dtype) in ("category", "string", "str")
        or pd.api.types.is_string_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
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
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    if not transformers:
        return Pipeline([("clf", clf)])
    return Pipeline(
        [("pre", ColumnTransformer(transformers, remainder="drop")), ("clf", clf)]
    )


def _xgb():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=4,
            verbosity=0,
            eval_metric="logloss",
        )
    except ImportError:
        return None


def evaluate_classicals(X_train, X_test, y_train, y_test) -> dict:
    results = {}
    lr = _make_classical_pipeline(
        X_train, LogisticRegression(max_iter=1000, solver="lbfgs")
    )
    lr.fit(X_train, y_train)
    results["logreg"] = float(accuracy_score(y_test, lr.predict(X_test)))

    xgb_cls = _xgb()
    if xgb_cls is not None:
        xgb = _make_classical_pipeline(X_train, xgb_cls)
        xgb.fit(X_train, y_train)
        results["xgboost"] = float(accuracy_score(y_test, xgb.predict(X_test)))
    return results


def run_dataset(name: str, openml_name: str, version: int) -> dict:
    logger.info("=== %s ===", name)
    bunch = fetch_openml(
        name=openml_name, version=version, as_frame=True, parser="auto"
    )
    X = bunch.data.copy()
    y = pd.Series(np.asarray(bunch.target)).astype(str)
    classes = {c: i for i, c in enumerate(sorted(y.unique()))}
    y = y.map(classes).astype(int)
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=42)
        y = y.loc[X.index]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Baseline — no FE
    base = evaluate_classicals(X_train, X_test, y_train, y_test)
    logger.info(
        "[%s] baseline  logreg=%.3f xgboost=%.3f",
        name,
        base.get("logreg", float("nan")),
        base.get("xgboost", float("nan")),
    )

    # Adaptive FE
    afe = AdaptiveFeatureEngineer(model=FE_MODEL, cv=3, verbose=True)
    afe.fit(X_train, y_train)

    record = {
        "dataset": name,
        "n_rows": len(X),
        "skip_reason": afe.skip_reason_,
        "probe_score_base": afe.probe_score_base_,
        "probe_score_fe": afe.probe_score_fe_,
        "probe_delta": afe.probe_delta_,
        "baseline": base,
    }

    if afe.skip_reason_:
        logger.info("[%s] AFE SKIPPED: %s", name, afe.skip_reason_)
        record["afe"] = None
        record["afe_delta"] = {k: 0.0 for k in base}
    else:
        X_train_fe = afe.transform(X_train)
        X_test_fe = afe.transform(X_test)
        n_new = X_train_fe.shape[1] - X_train.shape[1]
        logger.info("[%s] AFE added %d new columns", name, n_new)
        afe_scores = evaluate_classicals(X_train_fe, X_test_fe, y_train, y_test)
        record["afe"] = afe_scores
        record["afe_delta"] = {
            k: afe_scores.get(k, float("nan")) - base.get(k, float("nan")) for k in base
        }
        logger.info(
            "[%s] AFE  logreg=%+.3f  xgboost=%+.3f",
            name,
            record["afe_delta"].get("logreg", 0),
            record["afe_delta"].get("xgboost", 0),
        )

    return record


def plot_lift(records: list[dict], out_path: Path):
    datasets = [r["dataset"] for r in records]
    lr_delta = [
        r["afe_delta"].get("logreg", 0.0) if r["afe_delta"] else 0.0 for r in records
    ]
    xgb_delta = [
        r["afe_delta"].get("xgboost", 0.0) if r["afe_delta"] else 0.0 for r in records
    ]
    skipped = [r["skip_reason"] is not None for r in records]

    order = sorted(range(len(datasets)), key=lambda i: lr_delta[i], reverse=True)
    datasets = [datasets[i] for i in order]
    lr_delta = [lr_delta[i] for i in order]
    xgb_delta = [xgb_delta[i] for i in order]
    skipped = [skipped[i] for i in order]

    def bar_color(ds, val, is_skipped):
        if is_skipped:
            return "#aaaaaa"
        if ds in ABSTRACT_DATASETS:
            return "#9467bd" if val >= 0 else "#c5b0d5"
        return "#2ca02c" if val >= 0 else "#d62728"

    x = np.arange(len(datasets))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (ds, ld, xd, sk) in enumerate(zip(datasets, lr_delta, xgb_delta, skipped)):
        ax.bar(i - w / 2, ld, w, color=bar_color(ds, ld, sk), alpha=0.85)
        ax.bar(i + w / 2, xd, w, color=bar_color(ds, xd, sk), alpha=0.55)

    ax.axhline(0, color="black", linewidth=0.8)

    for i, (ld, xd, sk) in enumerate(zip(lr_delta, xgb_delta, skipped)):
        for val, offset in [(ld, -w / 2), (xd, w / 2)]:
            if math.isnan(val) or (sk and val == 0.0):
                continue
            ax.text(
                i + offset,
                val + (0.005 if val >= 0 else -0.008),
                f"{val:+.2f}" if not sk else "skip",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
            )

    # Label skipped datasets
    for i, (ds, sk, r) in enumerate(
        zip(datasets, skipped, [records[order[j]] for j in range(len(order))])
    ):
        if sk:
            reason = r["skip_reason"] or ""
            short = (
                "ceiling"
                if "ceiling" in reason
                else ("n_rows" if "n_rows" in reason else "gap")
            )
            ax.text(
                i,
                0.005,
                f"skip\n({short})",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#666666",
            )

    ax.set_xticks(x)
    labels = [
        f"{ds}\n({'abstract' if ds in ABSTRACT_DATASETS else 'semantic'})"
        for ds in datasets
    ]
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Accuracy delta (AFE − no FE)")
    ax.set_title(
        f"Per-dataset AdaptiveFE lift ({FE_MODEL} PromptFeatureEngineer)\n"
        "Dark bars = logreg  ·  Light bars = xgboost  ·  Grey = skipped by pre-flight  ·  Purple = abstract",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.85, label="logreg +AFE (semantic)"),
        Patch(facecolor="#2ca02c", alpha=0.55, label="xgboost +AFE (semantic)"),
        Patch(facecolor="#9467bd", alpha=0.85, label="logreg +AFE (abstract)"),
        Patch(facecolor="#9467bd", alpha=0.55, label="xgboost +AFE (abstract)"),
        Patch(facecolor="#aaaaaa", alpha=0.85, label="skipped (pre-flight)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (openml_name, version) in DATASETS.items():
        try:
            rec = run_dataset(name, openml_name, version)
            records.append(rec)
        except Exception as e:
            logger.error("[%s] FAILED: %s", name, e)
            records.append(
                {"dataset": name, "error": str(e), "skip_reason": None, "afe_delta": {}}
            )

    out_json = OUT_DIR / "metrics_afe.json"
    out_json.write_text(json.dumps(records, indent=2, default=str))
    print(f"Saved → {out_json}")

    # Summary table
    print(f"\n{'dataset':<18} {'skip reason':<45} {'lr_Δ':>7} {'xgb_Δ':>7}")
    print("-" * 82)
    for r in sorted(
        records, key=lambda r: r.get("afe_delta", {}).get("logreg", 0), reverse=True
    ):
        lr_d = r.get("afe_delta", {}).get("logreg", float("nan"))
        xgb_d = r.get("afe_delta", {}).get("xgboost", float("nan"))
        skip = r.get("skip_reason") or "-"
        print(f"{r['dataset']:<18} {skip[:45]:<45} {lr_d:>+7.3f} {xgb_d:>+7.3f}")

    plot_lift(records, OUT_DIR / "fe_per_dataset_lift_afe.png")


if __name__ == "__main__":
    main()
