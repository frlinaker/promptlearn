# Roadmap

Current stable release: **v0.5.0**

Released versions are documented in [CHANGELOG.md](CHANGELOG.md).

---

## The core thesis

promptlearn's accuracy is a **direct function of LLM capability** — it improves
for free every time a new model ships, with no code changes. The benchmark already
shows this: 54% (GPT-4o, May 2024) → 86% (GPT-5.5/Gemini 2.5 Flash, mid-2026),
closing in on XGBoost (92%) and TabPFN (94%). The roadmap below is organised
around amplifying that property.

---

## Shipped since v0.5.0 (unreleased)

- **Dataset context pre-pass**: `fit()` now accepts `dataset_description` and
  runs a pre-pass to generate a clean description + column decoder before the
  main code-generation prompt (`context` block injected above instructions)
- **Web search at fit time**: `web_search=True` on `fit()` enables real-time
  retrieval for GPT-5+ (OpenAI Responses API) and Gemini Vertex AI (grounding)
- **Prompt section markers**: centralized in `prompt_markers.py`; prompt assembly
  is now deterministic and inspectable
- **Prompt inspector**: HTML benchmark report with expandable per-model detail
  panels showing fit prompt, generated code, and per-stage timing
- **Benchmark extended**: TabPFN added as baseline; 13 datasets; web-search
  variants tracked separately; model-progression chart auto-sorts legend by
  accuracy; per-dataset timeline chart cleaned up

---

## Next: getting traction

### Living benchmark (the main growth driver)
The model-progression chart is our best marketing asset. It shows a clean rising
curve that updates automatically as new LLMs ship. Make it a public, auto-updating
leaderboard so the story tells itself.

- [ ] Host the benchmark chart at a stable URL (GitHub Pages or HuggingFace Space)
- [ ] Add a GitHub Action that re-runs the benchmark nightly/weekly and pushes
      updated charts — every new OpenAI/Google release becomes a press event
- [ ] Badge for README: live accuracy vs XGBoost gap

### Publicity (issues #9, #10, #11)
- [ ] Medium: *"Zero-shot classification closing in on XGBoost"*
      (issue #10) — progression chart is ready, no new code needed; this is the
      article most likely to drive GitHub stars
- [ ] Medium: *"How an LLM recovered F=ma from a CSV"* (issue #9) — regression
      story, generated code, explain(); targets a broader audience
- [ ] Kaggle notebook: `AdaptiveFeatureEngineer` + XGBoost (issue #11)

---

## v0.7.0 — confidence + ensemble

These two features amplify the "gets better with smarter models" thesis:

### Confidence / abstain
Let the model express uncertainty and refuse to predict on rows it can't handle.
Accuracy on *answered* questions climbs faster than raw accuracy, and the gap
to XGBoost narrows further.

- [ ] `predict_proba()` returns calibrated probabilities from the generated code
      (LLM annotates the heuristic with confidence weights per branch)
- [ ] `abstain_threshold` parameter: rows below confidence threshold return `None`
- [ ] Benchmark: accuracy@coverage curve (how does accuracy change as we
      allow more abstentions?)

### Multi-model ensemble
Run 2–3 cheap models + 1 frontier model, take majority vote. Approaches TabPFN
accuracy at lower per-row cost than a single frontier call.

- [ ] `EnsemblePromptClassifier(models=[...])` — fits each, majority-votes at predict
- [ ] Benchmark: does a GPT-4o-mini × 3 + GPT-5.5 × 1 ensemble beat GPT-5.5 alone?

---

## v0.8.0 — smarter prompting

Chain-of-thought and self-critique both benefit disproportionately from stronger
models — they widen the gap between weak and strong LLMs, making the progression
chart steeper.

- [ ] **Chain-of-thought fit**: ask the LLM to reason step-by-step before writing
      the predict function; store the reasoning in `clf.reasoning_`
- [ ] **Self-critique loop**: after generating code, ask the LLM to identify
      failure cases and patch them; retry up to N times
- [ ] **Contrastive examples**: when accuracy on a validation split is low, feed
      the wrong predictions back to the LLM for targeted refinement
- [ ] Benchmark: do these techniques close the gap on abstract datasets
      (kr-vs-kp, credit-g) where raw prompting struggles most?

---

## v1.0.0 — stable API

Declare the `fit` / `predict` / `predict_proba` / `score` / `sample` / `explain`
contract stable. Requires v0.7 confidence interface to be settled first.

---

## Ideas parking lot

Things worth exploring but not yet sequenced:

- **Regression: physical law recovery** — synthetic datasets (kinematics,
  thermodynamics, economics) where ground truth is a known formula; measure
  how well each LLM generation recovers the exact law (R², symbolic match)
- **Streaming leaderboard** — auto-submit new model results to a public HF dataset
  so the community can track progress without running the benchmark themselves
- **Plugin system for domain prompts** — let users inject domain knowledge into
  the fit prompt (e.g. "this is medical data; treat feature X as a risk factor")
- **Cross-lingual datasets** — do LLMs handle feature names in non-English languages?
  Tests whether world knowledge transfer is language-dependent
- **Time series support** — promptlearn's stateless `predict(**features)` contract
  is fundamentally row-independent and cannot model temporal dependencies. Supporting
  time series would require a different code generation target (e.g.
  `predict(history: list, horizon: int) -> list`) and a different evaluation
  framework. Natural fit would be a `PromptForecaster` class alongside sktime/darts
  conventions rather than sklearn's `predict(X)` interface.
