"""Generate FE vs no-FE comparison charts.

Compares:
  - promptlearn (no FE)   vs  promptlearn + FE (gpt-5.5)
  - logreg (no FE)        vs  logreg + FE (gpt-5.5)
  - xgboost (no FE)       vs  xgboost + FE (gpt-5.5)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

NO_FE_DIR = Path("benchmarks/progression_results")
FE_DIR = Path("benchmarks/progression_results_fe2")
OUT_DIR = Path("benchmarks/progression_results_fe2")

MODEL_ORDER = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "vertex_ai/gemini-2.5-flash-lite",
    "vertex_ai/gemini-2.5-flash",
    "vertex_ai/gemini-2.5-pro",
    "gpt-5.4-mini",
    "gpt-5.5",
    "vertex_ai/gemini-3.5-flash",
]

MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4.1": "GPT-4.1",
    "vertex_ai/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "vertex_ai/gemini-2.5-flash": "Gemini 2.5 Flash",
    "vertex_ai/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.5": "GPT-5.5",
    "vertex_ai/gemini-3.5-flash": "Gemini 3.5 Flash",
}

RELEASE_DATES = {
    "gpt-4o": "2024-05-13",
    "gpt-4o-mini": "2024-07-18",
    "gpt-4.1": "2025-04-14",
    "vertex_ai/gemini-2.5-flash-lite": "2025-07-22",
    "vertex_ai/gemini-2.5-flash": "2025-06-17",
    "vertex_ai/gemini-2.5-pro": "2025-06-17",
    "gpt-5.4-mini": "2026-03-01",
    "gpt-5.5": "2026-04-01",
    "vertex_ai/gemini-3.5-flash": "2026-05-19",
}


def load_promptlearn_means(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    sums: dict[str, list[float]] = {}
    for r in data:
        mid = r["model_id"]
        acc = (r.get("promptlearn") or {}).get("accuracy", float("nan"))
        if not math.isnan(acc):
            sums.setdefault(mid, []).append(acc)
    return {mid: float(np.mean(vals)) for mid, vals in sums.items()}


def load_baseline_means(path: Path) -> dict[str, dict[str, float]]:
    """Returns {learner: {dataset: accuracy}} from first occurrence per dataset."""
    data = json.loads(path.read_text())
    seen: set[str] = set()
    per_ds: dict[str, dict[str, float]] = {"logreg": {}, "xgboost": {}}
    for r in data:
        ds = r["dataset"]
        if ds in seen:
            continue
        seen.add(ds)
        for learner in ("logreg", "xgboost"):
            acc = (r.get(learner) or {}).get("accuracy", float("nan"))
            if not math.isnan(acc):
                per_ds[learner][ds] = acc
    return {
        learner: float(np.mean(list(v.values()))) if v else float("nan")
        for learner, v in per_ds.items()
    }


def plot_timeline_comparison(
    means_nofe: dict,
    means_fe: dict,
    baselines_nofe: dict,
    baselines_fe: dict,
    out_path: Path,
):
    models = [m for m in MODEL_ORDER if m in means_nofe or m in means_fe]
    dates = pd.to_datetime([RELEASE_DATES[m] for m in models])
    acc_nofe = [means_nofe.get(m, float("nan")) for m in models]
    acc_fe = [means_fe.get(m, float("nan")) for m in models]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # baselines — no FE (dashed) and with FE (dotted)
    styles = [
        ("logreg", "#4878CF", baselines_nofe["logreg"], baselines_fe["logreg"]),
        ("xgboost", "#6ACC65", baselines_nofe["xgboost"], baselines_fe["xgboost"]),
    ]
    for name, color, nofe_val, fe_val in styles:
        ax.axhline(
            nofe_val,
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{name} ({nofe_val:.3f})",
            zorder=1,
        )
        ax.axhline(
            fe_val,
            color=color,
            linewidth=1.5,
            linestyle=":",
            label=f"{name}+FE ({fe_val:.3f})",
            zorder=1,
        )

    # promptlearn cummax envelopes
    valid_nofe = [(d, a) for d, a in zip(dates, acc_nofe) if not math.isnan(a)]
    valid_fe = [(d, a) for d, a in zip(dates, acc_fe) if not math.isnan(a)]

    if valid_nofe:
        d_nofe, a_nofe = zip(*sorted(valid_nofe))
        env_nofe = np.maximum.accumulate(a_nofe)
        ax.plot(
            d_nofe,
            env_nofe,
            color="#E24A33",
            linewidth=2.5,
            marker="o",
            markersize=6,
            label="promptlearn no FE",
            zorder=3,
        )
        ax.scatter(dates, acc_nofe, color="#E24A33", s=25, alpha=0.35, zorder=4)

    if valid_fe:
        d_fe, a_fe = zip(*sorted(valid_fe))
        env_fe = np.maximum.accumulate(a_fe)
        ax.plot(
            d_fe,
            env_fe,
            color="#8B0000",
            linewidth=2.5,
            marker="s",
            markersize=6,
            label="promptlearn+FE (gpt-5.5)",
            zorder=3,
            linestyle="-",
        )
        ax.scatter(dates, acc_fe, color="#8B0000", s=25, alpha=0.35, zorder=4)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Model release date")
    ax.set_ylabel("Mean accuracy (13 datasets)")
    ax.set_title(
        "FE impact: promptlearn, logreg, xgboost — with and without FE (gpt-5.5)\n"
        "Dashed = no FE  ·  Dotted = with FE  ·  Lines = promptlearn cummax envelope",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_bar_comparison(
    means_nofe: dict,
    means_fe: dict,
    baselines_nofe: dict,
    baselines_fe: dict,
    out_path: Path,
):
    models = [m for m in MODEL_ORDER if m in means_nofe or m in means_fe]
    labels = [MODEL_LABELS[m] for m in models]
    a_nofe = [means_nofe.get(m, float("nan")) for m in models]
    a_fe = [means_fe.get(m, float("nan")) for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars1 = ax.bar(
        x - w / 2, a_nofe, w, label="promptlearn no FE", color="#E24A33", alpha=0.85
    )
    bars2 = ax.bar(
        x + w / 2, a_fe, w, label="promptlearn+FE", color="#8B0000", alpha=0.85
    )

    for name, color, nofe_val, fe_val in [
        ("logreg", "#4878CF", baselines_nofe["logreg"], baselines_fe["logreg"]),
        ("xgboost", "#6ACC65", baselines_nofe["xgboost"], baselines_fe["xgboost"]),
    ]:
        ax.axhline(
            nofe_val,
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{name} ({nofe_val:.3f})",
        )
        ax.axhline(
            fe_val,
            color=color,
            linewidth=1.5,
            linestyle=":",
            label=f"{name}+FE ({fe_val:.3f})",
        )

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if not math.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.004,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Mean accuracy (13 datasets)")
    ax.set_title(
        "FE impact per model — promptlearn vs logreg vs xgboost, with/without FE",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_delta_bar(means_nofe: dict, means_fe: dict, out_path: Path):
    models = [m for m in MODEL_ORDER if m in means_nofe and m in means_fe]
    labels = [MODEL_LABELS[m] for m in models]
    deltas = [means_fe[m] - means_nofe[m] for m in models]
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, deltas, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, d in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + (0.002 if d >= 0 else -0.004),
            f"{d:+.3f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            fontsize=8,
        )
    ax.set_ylabel("Accuracy delta (FE − no FE)")
    ax.set_title("promptlearn FE impact per model (FE via gpt-5.5)", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    means_nofe = load_promptlearn_means(NO_FE_DIR / "metrics_all.json")
    means_fe = load_promptlearn_means(FE_DIR / "metrics_all.json")
    baselines_nofe = load_baseline_means(NO_FE_DIR / "metrics_all.json")
    baselines_fe = load_baseline_means(FE_DIR / "metrics_all.json")

    print(f"\n{'Learner':<20} {'no FE':>7} {'+ FE':>7} {'delta':>7}")
    print("-" * 42)
    print(
        f"{'logreg':<20} {baselines_nofe['logreg']:>7.3f} {baselines_fe['logreg']:>7.3f} {baselines_fe['logreg']-baselines_nofe['logreg']:>+7.3f}"
    )
    print(
        f"{'xgboost':<20} {baselines_nofe['xgboost']:>7.3f} {baselines_fe['xgboost']:>7.3f} {baselines_fe['xgboost']-baselines_nofe['xgboost']:>+7.3f}"
    )
    print()
    print(f"{'Model (promptlearn)':<25} {'no FE':>7} {'+ FE':>7} {'delta':>7}")
    print("-" * 48)
    for m in MODEL_ORDER:
        nf = means_nofe.get(m, float("nan"))
        fe = means_fe.get(m, float("nan"))
        d = fe - nf if not (math.isnan(nf) or math.isnan(fe)) else float("nan")
        label = MODEL_LABELS[m]
        print(f"{label:<25} {nf:>7.3f} {fe:>7.3f} {d:>+7.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_timeline_comparison(
        means_nofe,
        means_fe,
        baselines_nofe,
        baselines_fe,
        OUT_DIR / "fe_comparison_timeline.png",
    )
    plot_bar_comparison(
        means_nofe,
        means_fe,
        baselines_nofe,
        baselines_fe,
        OUT_DIR / "fe_comparison_bar.png",
    )
    plot_delta_bar(means_nofe, means_fe, OUT_DIR / "fe_delta.png")


if __name__ == "__main__":
    main()
