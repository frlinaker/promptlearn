# Roadmap

Current stable release: **v0.4.1**

Released versions are documented in [CHANGELOG.md](CHANGELOG.md).

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
