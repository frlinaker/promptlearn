# Roadmap

Current stable release: **v0.3.0**

---

## [v0.3.1] — benchmark release *(unreleased, ready to tag)*
Benchmark improvements already on `main` since v0.3.0:
- [x] Benchmark output highlights winning models
- [x] Benchmark cleanup and black formatting
- [x] Remove unused functionality

---

## [v0.4.0] — multi-provider + reliability
The most important release. Removes the OpenAI lock-in and makes the
code-generation step production-safe.

- [ ] Add Anthropic Claude provider (`model="claude-sonnet-4-6"`)
- [ ] Add Ollama provider for local models (`model="ollama:llama3.1"`)
- [ ] Add code validation sandbox — run generated code on a held-out sample
      before `fit()` returns, retry up to N times on failure
- [ ] Add `explain()` method — plain English description of the generated heuristic

---

## [v0.5.0] — feature engineering + benchmarks
Widens the surface area and builds the credibility case.

- [ ] Add `PromptFeatureEngineer` transformer — sklearn preprocessing step that
      uses the LLM to generate new features from semantically meaningful columns
      before a downstream classical model
- [ ] Benchmark suite across 10+ OpenML datasets with semantically meaningful
      categoricals — compare promptlearn, scikit-LLM, TabPFN, XGBoost, logistic
      regression. Publish results in README.

---

## [v0.6.0] — usability
- [ ] CLI entry point: `promptlearn fit mydata.csv --target col`
- [ ] `uncertainty` / `confidence` flag on generated code for zero-shot fits

---

## [v1.0.0] — stable API
Declare the `fit` / `predict` / `score` / `sample` / `explain` contract stable.
Requires benchmark suite and multi-provider support to be in good shape.

---

## Publicity (version-independent)
- [ ] Medium / Towards Data Science: *"How an LLM recovered F=ma from a CSV"*
      (built around the falling-object benchmark result)
- [ ] Medium: *"Zero-shot classification that outperforms XGBoost — when and why"*
- [ ] Kaggle notebook on a Playground Series competition using `PromptFeatureEngineer`
      as a feature engineering step, showing leaderboard delta vs vanilla XGBoost
