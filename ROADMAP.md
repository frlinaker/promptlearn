# Roadmap

Current stable release: **v0.5.0**

Released versions are documented in [CHANGELOG.md](CHANGELOG.md).

---

## [v0.6.0] — more baselines + usability ✓ DONE

- [x] Add TabPFN as an optional benchmark baseline — 93.7% mean accuracy across 13
      datasets, stronger ceiling than XGBoost (91.9%); runs once per dataset,
      model-independent
- [x] CLI entry point: `promptlearn fit mydata.csv --target col`
- [x] Per-dataset FE lift analysis — semantic datasets (car, adult, soybean) gained
      most; abstract datasets (tic-tac-toe, monks-2) also benefited via AFE;
      see `benchmarks/progression_results/fe_per_dataset_lift_afe.png`

---

## [v1.0.0] — stable API
Declare the `fit` / `predict` / `score` / `sample` / `explain` contract stable.
Requires benchmark suite and multi-provider support to be in good shape.

---

## Publicity (version-independent)
- [ ] Medium / Towards Data Science: *"How an LLM recovered F=ma from a CSV"*
      (issue #9) — falling-object regression demo, generated code, explain()
- [ ] Medium: *"Zero-shot classification closing in on XGBoost — when and why"*
      (issue #10) — model-progression chart ready; GPT-5.5/Gemini 3.5 Flash at
      84%, XGBoost 91.9%, TabPFN 93.7%; heatmap shows where LLMs win
- [ ] Kaggle notebook: `AdaptiveFeatureEngineer` + XGBoost vs vanilla XGBoost
      (issue #11) — AFE lifts logreg up to +29pp on semantic datasets

---

## LLM evolution benchmark (model-progression chart) ✓ DONE

**Status:** Complete. `benchmarks/run_model_progression.py`

**Results (13 OpenML datasets, FE off):**

| Model | Mean accuracy |
|---|---|
| GPT-4o (May 2024) | 54.0% |
| GPT-4o mini | 42.8% |
| GPT-4.1 | 60.0% |
| Gemini 2.5 Flash Lite | 57.6% |
| Gemini 2.5 Flash | 70.0% |
| GPT-5.4 mini | 50.4% |
| Gemini 2.5 Pro | 77.3% |
| GPT-5.5 | **84.4%** |
| Gemini 3.5 Flash | **84.1%** |
| logreg baseline | 87.6% |
| xgboost baseline | 91.9% |
| TabPFN baseline | **93.7%** |

**FE lift results** (`benchmarks/progression_results/metrics_afe.json`):

| Dataset | logreg Δ | xgboost Δ | notes |
|---|---|---|---|
| car | +28.7pp | +1.4pp | abstract-style but FE cracked the structure |
| tic-tac-toe | +32.9pp | +0.4pp | abstract — AFE found win-condition features |
| monks-2 | +2.0pp | +9.9pp | abstract |
| soybean | +8.2pp | +5.3pp | semantic |
| adult | +3.4pp | +1.0pp | semantic |
| nursery | +3.8pp | −2.0pp | semantic — xgb near ceiling |
| credit-g | +6.8pp | −2.8pp | semantic — xgb near ceiling |

AFE skipped: bank-marketing, mushroom, vote, kr-vs-kp, hepatitis, lymph (pre-flight vetoed)

**Datasets:** adult, credit-g, bank-marketing, mushroom, car, nursery, vote,
tic-tac-toe, kr-vs-kp, monks-2, soybean, hepatitis, lymph

**CLI:**
```bash
python benchmarks/run_model_progression.py
python benchmarks/run_adaptive_fe_benchmark.py
```
