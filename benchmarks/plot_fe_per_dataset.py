"""Per-dataset FE lift chart: logreg and xgboost with vs without FE (gpt-5.5)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

NO_FE_DIR = Path("benchmarks/progression_results")
FE_DIR = Path("benchmarks/progression_results_fe2")
OUT_DIR = Path("benchmarks/progression_results_fe2")

SEMANTIC_DATASETS = {
    "adult",
    "credit-g",
    "bank-marketing",
    "hepatitis",
    "lymph",
    "soybean",
    "vote",
    "nursery",
    "car",
    "mushroom",
}
ABSTRACT_DATASETS = {"tic-tac-toe", "kr-vs-kp", "monks-2"}


def load_per_dataset_baselines(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    seen: set[str] = set()
    result: dict[str, dict[str, float]] = {}
    for r in data:
        ds = r["dataset"]
        if ds in seen:
            continue
        seen.add(ds)
        result[ds] = {
            "logreg": (r.get("logreg") or {}).get("accuracy", float("nan")),
            "xgboost": (r.get("xgboost") or {}).get("accuracy", float("nan")),
        }
    return result


def main():
    nofe = load_per_dataset_baselines(NO_FE_DIR / "metrics_all.json")
    fe = load_per_dataset_baselines(FE_DIR / "metrics_all.json")

    datasets = sorted(nofe.keys())
    lr_delta = [
        fe.get(ds, {}).get("logreg", float("nan")) - nofe[ds]["logreg"]
        for ds in datasets
    ]
    xgb_delta = [
        fe.get(ds, {}).get("xgboost", float("nan")) - nofe[ds]["xgboost"]
        for ds in datasets
    ]

    # Sort by logreg delta descending
    order = sorted(range(len(datasets)), key=lambda i: lr_delta[i], reverse=True)
    datasets = [datasets[i] for i in order]
    lr_delta = [lr_delta[i] for i in order]
    xgb_delta = [xgb_delta[i] for i in order]

    # Colour bars by semantic vs abstract
    def bar_color(ds, positive):
        if ds in ABSTRACT_DATASETS:
            return "#9467bd" if positive else "#c5b0d5"
        return "#2ca02c" if positive else "#d62728"

    x = np.arange(len(datasets))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5))

    for i, (ds, ld, xd) in enumerate(zip(datasets, lr_delta, xgb_delta)):
        ax.bar(i - w / 2, ld, w, color=bar_color(ds, ld >= 0), alpha=0.85)
        ax.bar(i + w / 2, xd, w, color=bar_color(ds, xd >= 0), alpha=0.55)

    ax.axhline(0, color="black", linewidth=0.8)

    # Annotate bars
    for i, (ld, xd) in enumerate(zip(lr_delta, xgb_delta)):
        for val, offset in [(ld, -w / 2), (xd, w / 2)]:
            if math.isnan(val):
                continue
            ax.text(
                i + offset,
                val + (0.005 if val >= 0 else -0.008),
                f"{val:+.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
            )

    ax.set_xticks(x)
    labels = [
        f"{ds}\n({'abstract' if ds in ABSTRACT_DATASETS else 'semantic'})"
        for ds in datasets
    ]
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Accuracy delta (FE − no FE)")
    ax.set_title(
        "Per-dataset FE lift (gpt-5.5 PromptFeatureEngineer)\n"
        "Dark bars = logreg  ·  Light bars = xgboost  ·  Purple = abstract datasets",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)

    # Legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.85, label="logreg +FE (semantic)"),
        Patch(facecolor="#2ca02c", alpha=0.55, label="xgboost +FE (semantic)"),
        Patch(facecolor="#9467bd", alpha=0.85, label="logreg +FE (abstract)"),
        Patch(facecolor="#9467bd", alpha=0.55, label="xgboost +FE (abstract)"),
        Patch(facecolor="#d62728", alpha=0.85, label="negative delta"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right", ncol=2)

    fig.tight_layout()
    out = OUT_DIR / "fe_per_dataset_lift.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")

    # Print table
    print(f"\n{'dataset':<18} {'semantic?':>10} {'lr_Δ':>7} {'xgb_Δ':>7}")
    print("-" * 46)
    for ds, ld, xd in zip(datasets, lr_delta, xgb_delta):
        tag = "abstract" if ds in ABSTRACT_DATASETS else "semantic"
        print(f"{ds:<18} {tag:>10} {ld:>+7.3f} {xd:>+7.3f}")


if __name__ == "__main__":
    main()
