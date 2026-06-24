# Roadmap

Current stable release: **v0.5.0**

Released versions are documented in [CHANGELOG.md](CHANGELOG.md).

---

## [v0.6.0] — more baselines + usability
- [ ] Add TabPFN and scikit-LLM as optional benchmark baselines (kept out of
      core deps); the v0.5.0 suite ships XGBoost + logistic regression
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
