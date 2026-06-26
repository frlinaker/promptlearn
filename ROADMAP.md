# Roadmap

Current stable release: **v0.5.0**

Released versions are documented in [CHANGELOG.md](CHANGELOG.md).

---

## [v0.6.0] — more baselines + usability
- [ ] Add TabPFN as an optional benchmark baseline — currently our best models
      (Gemini 3.5 Flash 86.1%, GPT-5.5 83.6%) are within ~6pp of xgboost (92.3%);
      TabPFN is a stronger ceiling and a fairer comparison on small tabular datasets
- [ ] CLI entry point: `promptlearn fit mydata.csv --target col`
- [ ] `uncertainty` / `confidence` flag on generated code for zero-shot fits
- [ ] Per-dataset FE lift analysis — show which datasets benefit most from FE
      (hypothesis: semantic datasets like soybean/hepatitis/lymph should gain more
      than abstract ones like tic-tac-toe/monks-2)

---

## [v1.0.0] — stable API
Declare the `fit` / `predict` / `score` / `sample` / `explain` contract stable.
Requires benchmark suite and multi-provider support to be in good shape.

---

## Publicity (version-independent)
- [ ] Medium / Towards Data Science: *"How an LLM recovered F=ma from a CSV"*
      (built around the falling-object benchmark result)
- [ ] Medium: *"Zero-shot classification that outperforms XGBoost — when and why"*
      (model-progression chart is ready; Gemini 3.5 Flash at 86% closing fast on
      xgboost 92%; semantic dataset heatmap shows where and why LLMs win)
- [ ] Kaggle notebook: `PromptFeatureEngineer` + XGBoost vs vanilla XGBoost —
      benchmark shows FE lifts logreg by +3.7pp (0.877→0.914); XGBoost lift smaller
      (+0.6pp) but still positive; good story for competition leaderboard delta

---

## LLM evolution benchmark (model-progression chart) ✓ DONE

**Status:** Complete. `benchmarks/run_model_progression.py`

**Results (13 OpenML datasets, FE off):**

| Model | Mean accuracy |
|---|---|
| GPT-4o (May 2024) | 53.1% |
| GPT-4o mini | 44.8% |
| GPT-4.1 | 54.7% |
| Gemini 2.5 Flash | 67.8% |
| Gemini 2.5 Pro | 77.6% |
| GPT-5.5 | 83.6% |
| Gemini 3.5 Flash (May 2026) | **86.1%** |
| logreg baseline | 87.7% |
| xgboost baseline | 92.3% |

**FE results** (`--fe-model gpt-5.5`, `benchmarks/progression_results_fe2/`):

| Learner | no FE | + FE | delta |
|---|---|---|---|
| logreg | 87.7% | **91.4%** | +3.7pp |
| xgboost | 92.3% | 92.9% | +0.6pp |
| promptlearn (gpt-5.5) | 83.6% | 76.4% | -7.2pp |

FE helps classical ML (especially logreg) but hurts promptlearn, which already
reasons semantically in its classifier prompt.

**Datasets:** adult, credit-g, bank-marketing, mushroom, car, nursery, vote,
tic-tac-toe, kr-vs-kp, monks-2, soybean, hepatitis, lymph

**CLI:**
```bash
# no FE
python benchmarks/run_model_progression.py --output-dir benchmarks/progression_results

# with FE via gpt-5.5
python benchmarks/run_model_progression.py --fe-model gpt-5.5 --output-dir benchmarks/progression_results_fe2
```
