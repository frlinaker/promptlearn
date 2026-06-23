# Changelog

All notable changes to promptlearn are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — benchmark polish
### Changed
- Benchmark output now highlights winning models per metric
- Benchmark code reformatted with black and redundancies removed
- Cleaned up unused functionality across estimators and benchmark runner

### Fixed
- Pylance type warnings in benchmark module

---

## [0.3.0] — 2025-07-12 — second-pass generalisation
### Added
- Second LLM pass that refines and extends the generated Python heuristic
- Riddle-style example demonstrating how the second pass improves edge cases
- Kaggle dataset example (added in 0.2.3, documented here)

---

## [0.2.3] — 2025-07-12 — housekeeping
### Added
- Kaggle dataset example

### Removed
- Unused dependencies trimmed from requirements

---

## [0.2.2] — 2025-07-10 — test coverage
### Added
- Full test suite covering classifier, regressor, and edge cases
- Test coverage for missing data, zero-row fitting, and boolean targets

---

## [0.2.1] — 2025-07-09 — refactor and sample
### Added
- Reintroduced `.sample(n)` method for generating synthetic rows from a fitted model

### Changed
- Classifier refactored for clarity and reduced duplication
- Common base logic simplified and consolidated
- Codebase reformatted with black

### Fixed
- Model benchmark code corrected

---

## [0.2.0] — 2025-07-08 — Python heuristics (major internal change)
### Changed
- Estimators now generate a **standalone Python function** as the heuristic instead of
  returning raw LLM text. Predictions run the generated function directly — no LLM call
  at predict time. This is the core architectural shift that makes inference fast and
  the model serializable.
- Regressor updated to output Python heuristics, benchmark fixed accordingly
- README updated to reflect Python heuristic approach

### Fixed
- Broken joblib test removed
- Classifier test corrected

---

## [0.1.0] — initial release
### Added
- `PromptClassifier` — sklearn-compatible classifier backed by LLM reasoning
- `PromptRegressor` — sklearn-compatible regressor backed by LLM reasoning
- Zero-row fitting: pass column names only, model infers from world knowledge
- `.sample(n)` method for generating synthetic rows from the fitted heuristic
- `joblib` serialization support (LLM client excluded, heuristic preserved)
- Pandas DataFrame support for both estimators
- Sliding window chunking for large datasets
- Benchmarks: mammal classification and falling-object regression vs traditional models
- Examples: Iris, Titanic, census, country flag colours, XOR, multioutput
- Type annotations throughout
- PyPI packaging
