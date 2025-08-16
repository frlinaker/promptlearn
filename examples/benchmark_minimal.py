# examples/benchmark_minimal.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set

import yaml

from promptlearn.benchmark import (
    run_benchmark,
    mark_winners_for_display,
)

# -----------------------------------------------------------------------------
# Where is ccbench?
# Prefer CCBENCH_DIR (new). Fallback to CCBENCH_TASKS_DIR (legacy), or ../ccbench.
# -----------------------------------------------------------------------------
CCBENCH_DIR = Path(
    os.environ.get("CCBENCH_DIR") or os.environ.get("CCBENCH_TASKS_DIR", "../ccbench")
).resolve()
TASKS_ROOT = CCBENCH_DIR / "tasks"


def _read_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _task_passes_filters(
    task_dir: Path, ptc: Optional[Set[str]], temporal: Optional[Set[str]]
) -> bool:
    """Return True if task's meta.yaml matches semantic filters (if provided)."""
    meta = _read_yaml(task_dir / "meta.yaml")
    k = meta.get("knowledge") or {}
    pc = k.get("predict_connectivity")
    tm = k.get("temporal")
    if ptc and pc not in ptc:
        return False
    if temporal and tm not in temporal:
        return False
    return True


def discover_all_tasks() -> List[Path]:
    """Traverse tasks/**/meta.yaml and return task directories, optionally filtered by env."""
    if not TASKS_ROOT.exists():
        raise FileNotFoundError(f"Can't find tasks root: {TASKS_ROOT}")

    # Optional semantic filters via env (purely labels; analysis-first mindset)
    env_ptc = os.environ.get("PLBENCH_FILTER_PTC")  # e.g., "offline,web_non_ai"
    env_temporal = os.environ.get("PLBENCH_FILTER_TEMP")  # e.g., "static,slow"
    ptc_set = set(s.strip() for s in env_ptc.split(",")) if env_ptc else None
    temporal_set = (
        set(s.strip() for s in env_temporal.split(",")) if env_temporal else None
    )

    # Optional simple excludes via glob (comma-separated). Example: "**/draft_*"
    env_excl = os.environ.get("PLBENCH_EXCLUDE", "").strip()
    excl_globs = [g.strip() for g in env_excl.split(",") if g.strip()]

    all_task_dirs = sorted({m.parent.resolve() for m in TASKS_ROOT.rglob("meta.yaml")})
    if excl_globs:
        excluded = set()
        for g in excl_globs:
            excluded.update({p.parent.resolve() for p in TASKS_ROOT.rglob(g)})
        all_task_dirs = [p for p in all_task_dirs if p not in excluded]

    tasks = [p for p in all_task_dirs if _task_passes_filters(p, ptc_set, temporal_set)]
    if not tasks:
        print(
            f"[WARN] No tasks matched. Root={TASKS_ROOT}, filters ptc={ptc_set} temporal={temporal_set}, excludes={excl_globs}"
        )
    else:
        print(f"[INFO] Discovered {len(tasks)} tasks under {TASKS_ROOT}")
    return tasks


def main():
    tasks = discover_all_tasks()

    # Models:
    # Option A (recommended): let the runner pick sensible defaults per task type.
    #   Add LLM variants via env: PLBENCH_LLM_MODELS="gpt-4o,gpt-5"
    models = None

    # Option B: explicitly pin a few (example kept for convenience)
    # models = [
    #     make_sklearn("LogReg", "sklearn.linear_model.LogisticRegression", max_iter=1000),
    #     make_sklearn("RF", "sklearn.ensemble.RandomForestClassifier", n_estimators=300, random_state=0),
    #     make_promptlearn_variant("classifier", llm_name="gpt-4o", name="PromptLearnClf[gpt-4o]"),
    #     make_promptlearn_variant("classifier", llm_name="gpt-5",  name="PromptLearnClf[gpt-5]"),
    # ]

    cls_df, reg_df = run_benchmark(
        tasks, models=models, out_dir="runs", resume=True, return_kind="split"
    )

    print("\n=== Classification ===")
    print(
        mark_winners_for_display(
            cls_df, primary_metric="accuracy", higher_is_better=True
        )
    )

    print("\n=== Regression ===")
    print(
        mark_winners_for_display(reg_df, primary_metric="rmse", higher_is_better=False)
    )


if __name__ == "__main__":
    main()
