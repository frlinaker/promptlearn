
# promptlearn

[![GitHub last commit](https://img.shields.io/github/last-commit/frlinaker/promptlearn)](https://github.com/frlinaker/promptlearn)
[![PyPI - Version](https://img.shields.io/pypi/v/promptlearn)](https://pypi.org/project/promptlearn/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/promptlearn)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/promptlearn)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/promptlearn)
[![Licence](https://img.shields.io/github/license/frlinaker/promptlearn
)](https://mit-license.org/)

**promptlearn** brings large language models into your scikit-learn workflow. It is able to look at data, reason about the meaning of inputs and outputs, relate it to and identify relevant knowledge of the world, automatically building standalone executable Python code that augments the relationships of the original data with relevant materialized world-knowledge about categorical variables.

---

### 📊 Outperforming Traditional Models with Built-In Knowledge

Consider a simple binary classification task: predicting whether an [animal is a mammal](examples/quickstart.py) given things like its name, weight, and lifespan (`python examples/quickstart.py --demo compare --dataset mammal`).

Traditional models depend solely on the input features. But `promptlearn` models can use their internal understanding of zoology to form highly accurate rules, pulling in data about known mammals, and making that knowledge available in explicit reference tables for subsequent predictions.

|             model             | accuracy (higher is better) | fit_time_sec | predict_time_sec |
|:------------------------------|---------:|-------------:|-----------------:|
|      promptlearn_o3-mini      |   0.94   |   49.11      |      0.0028      |
|      promptlearn_o4-mini      |   0.86   |   60.96      |      0.0024      |
| promptlearn_gpt-3.5-turbo     |   0.66   |   20.25      |      0.0027      |
|      promptlearn_gpt-4o       |   0.66   |   43.93      |      0.0023      |
|      logistic_regression      |   0.60   |   0.02       |      0.0010      |
|        decision_tree          |   0.53   |   0.0014     |      0.0005      |
|      gradient_boosting        |   0.53   |   0.02       |      0.0011      |
|      promptlearn_gpt-4        |   0.40   |   12.49      |      0.0022      |
|            dummy              |   0.34   |   0.0006     |      0.0001      |
|        random_forest          |   0.28   |   0.01       |      0.0017      |

This type of semantic generalization is a powerful advantage for LLM-backed models.

---

Now compare performance on a regression task where the data contains samples of [objects falling from different heights, under different gravity](examples/quickstart.py) (`python examples/quickstart.py --demo compare --dataset fall`). This is a classic physics problem, with a well-known equation:

```
fall_time_s = sqrt((2 * height_m) / gravity_mps2)
```

`promptlearn` estimators are able to recover this exact formula, using just the dataframe itself, and use it to generate perfect predictions:

|              model             |   mse (lower is better)  | fit_time_sec | predict_time_sec |
|:-------------------------------|--------:|-------------:|-----------------:|
|      promptlearn_gpt-4o        |  0.000  |     2.92     |      0.001       |
|     promptlearn_o3-mini        |  0.000  |    10.80     |      0.001       |
|     promptlearn_o4-mini        |  0.000  |     7.96     |      0.001       |
|        random_forest           |  0.028  |     0.01     |      0.002       |
|      gradient_boosting         |  0.035  |     0.01     |      0.001       |
|        decision_tree           |  0.067  |     0.001    |      0.000       |
|      linear_regression         |  0.498  |     0.001    |      0.000       |
|             dummy              |  5.273  |     0.001    |      0.000       |
| promptlearn_gpt-3.5-turbo      | 18.193  |     3.01     |      0.002       |
|      promptlearn_gpt-4         | 855.445 |     2.43     |      0.001       |

No feature engineering was performed. No physics constants were added. The model discovered the rule and applied it directly. Classical regressors, by contrast, approximated a curve but missed the exact structure.

These results highlight the practical benefit of reasoning models: they learn compact, expressive heuristics and can outperform traditional systems when symbolic insight or background knowledge is essential.

---

### 🤖 Estimators Powered by Language

`promptlearn` provides scikit-learn-compatible estimators that use LLMs as the modeling engine:

- **`PromptClassifier`** – for predicting classes through generalized reasoning
- **`PromptRegressor`** – for modeling numeric relationships in data
- **`PromptFeatureEngineer`** – a transformer that derives new, world-knowledge-rich features for a downstream classical model

These estimators follow the same API as other `scikit-learn` models (`fit`, `predict`, `score`) but operate via dynamic prompt construction and few-shot abstraction.

---

### 🧪 LLM Feature Engineering (`PromptFeatureEngineer`)

`PromptFeatureEngineer` is a scikit-learn transformer that, at `fit`, asks the LLM to write a standalone `transform()` function deriving new features from semantically meaningful columns (e.g. mapping a country to its GDP tier, parsing a date into `is_weekend`, bucketing ages). At `transform` it just runs that generated code — **no per-row LLM calls** — and appends the engineered columns, so it drops straight into a `Pipeline` before any classical model:

```python
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from promptlearn import PromptFeatureEngineer

# PromptFeatureEngineer appends engineered columns to the original frame, so a
# downstream linear model still wants categoricals one-hot encoded.
encode = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), selector(dtype_exclude="number"))],
    remainder="passthrough",
)
pipe = Pipeline([
    ("features", PromptFeatureEngineer()),  # LLM-generated feature code
    ("encode", encode),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

This lets a fast, interpretable linear model benefit from the LLM's world knowledge while keeping inference cheap and serializable.

---

### 🚀 Try It

Everything runnable lives in a single guided tour, [`examples/quickstart.py`](examples/quickstart.py) — a menu of self-contained demos. Each makes live LLM calls, so run them one at a time:

```bash
python examples/quickstart.py --list             # see all the demos
python examples/quickstart.py --demo zero_row     # fit on column names only
python examples/quickstart.py --demo titanic --dump artifacts/   # deep tour: generated code, explain(), joblib
python examples/quickstart.py --demo compare --dataset mammal    # promptlearn vs sklearn/XGBoost
```

The demos cover zero-row fitting, `.sample()`, joblib round-tripping, world-knowledge reasoning, linear/nonlinear/multi-output regression, XOR, `GridSearchCV` tuning, a large real OpenML dataset, the side-by-side model `compare`, and the deep `titanic` walkthrough (generated `predict()` code, `explain()`, and artifact dumping).

The `compare` demo is powered by the reusable `promptlearn.compare_models(models, X_train, y_train, X_test, y_test)` helper, which works with any mix of promptlearn and sklearn/XGBoost estimators.

---

### 📈 Benchmark: feature engineering across 10 OpenML datasets

Accuracy on a held-out test split for 10 OpenML classification datasets with semantically meaningful categoricals. `promptlearn+FE` is `PromptFeatureEngineer → one-hot → LogisticRegression`; the promptlearn contenders use `gpt-5.4-mini`. Reproduce with [`benchmarks/run_openml_benchmark.py`](benchmarks/run_openml_benchmark.py).

| dataset | promptlearn | promptlearn+FE | logreg | xgboost |
| --- | ---: | ---: | ---: | ---: |
| adult | 0.782 | 0.858 | 0.864 | 0.850 |
| credit-g | 0.680 | 0.748 | 0.724 | 0.728 |
| bank-marketing | 0.678 | 0.876 | 0.868 | 0.878 |
| mushroom | 0.970 | 1.000 | 1.000 | 1.000 |
| car | 0.417 | 0.963 | 0.910 | 0.988 |
| nursery | 0.492 | 0.944 | 0.932 | 0.974 |
| vote | 0.193 | 0.945 | 0.954 | 0.982 |
| tic-tac-toe | 0.979 | 1.000 | 0.979 | 0.983 |
| kr-vs-kp | 0.598 | 0.966 | 0.964 | 0.992 |
| monks-2 | 0.616 | 0.623 | 0.583 | 0.874 |
| **mean** | **0.640** | **0.892** | **0.878** | **0.925** |

**Takeaway:** adding `PromptFeatureEngineer` in front of a plain logistic regression lifts mean accuracy from 0.878 to **0.892** — beating logistic regression on most datasets and even edging out XGBoost on `adult`, `credit-g`, and `tic-tac-toe`, while keeping a fully interpretable linear model and cheap inference. Using the LLM as a *direct* classifier (`promptlearn` alone) is weaker here: when the target's class encoding is arbitrary (e.g. party labels in `vote`), direct prediction has no semantic foothold, which is exactly the gap feature engineering closes.

---

### 🔌 Choose Your Provider

The LLM provider is selected by the `model` string and resolved via [LiteLLM](https://github.com/BerriAI/litellm), so you are not locked into OpenAI:

```python
PromptClassifier(model="gpt-5.5")            # OpenAI (the default)
PromptClassifier(model="claude-sonnet-4-6")  # Anthropic
PromptClassifier(model="ollama:llama3.1")    # local Ollama
```

API keys are read from the usual per-provider environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …); local providers like Ollama need none.

To change the default model without touching code, set `PROMPTLEARN_MODEL` (e.g. `export PROMPTLEARN_MODEL=gpt-5.4-mini` for faster, cheaper runs). An explicit `model=` argument always takes precedence.

---

### 🕳 Zero-Example Learning

If you call `.fit()` with no rows — just column names — `promptlearn` will still return a working model.

This is possible because the LLM can hallucinate a plausible mapping based on:

- Column names
- Prior knowledge
- Type hints or value patterns

This makes rapid prototyping and conceptual modeling trivial.

---

### 🧪 Native `.sample()` Support

You can generate synthetic rows directly from any trained model using `.sample(n)`:

```
>>> model.sample(3)
fruit    is_citrus
Lime     1
Banana   0
Orange   1
```

This is useful for:

- Understanding what the model believes
- Creating test sets or bootstrapped data
- Building readable examples from internal logic

---

### 🔎 Explain the Learned Rule

Call `.explain()` to get a plain-English description of the heuristic the model
learned — useful for interpretability reporting:

```python
>>> explanation = model.explain()
>>> print(explanation)
Predicts 1 (adult) when `age` is at least 18, otherwise 0.

>>> explanation.features_used
['age']
```

`explain()` returns an `Explanation` object with `meta` and `data` dicts (keys
also reachable as attributes) that is JSON round-trippable via `to_json()` /
`Explanation.from_json(...)`. A bare `explain()` describes the whole model
(global, and cached so it's deterministic); passing a single row, `explain(X)`,
describes that one prediction (local).

---

### 💾 Save and Reload with `joblib`

Like any scikit-learn model, `promptlearn` estimators can be serialized:

```python
import joblib

joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")
```

The compiled prediction function is excluded from the saved file and recompiled on load. The heuristic remains intact, interpretable, and ready to use.

---

## 📚 Related Work

### Scikit-LLM

[Scikit-LLM](https://github.com/BeastByteAI/scikit-llm) provides zero- and few-shot classification through template-based prompting.  
It is lightweight and NLP-focused.

**promptlearn** offers a broader modeling philosophy:

| Capability                  | Scikit-LLM         | promptlearn                |
|-----------------------------|--------------------|----------------------------|
| Produces runnable Python code | ❌ No               | ✅ Yes                     |
| Regression support          | ❌ No               | ✅ Yes                     |

---

## 🛠 Development

Install the dev dependencies and enable the git hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

The pre-commit hooks run [black](https://github.com/psf/black) and the full
test suite, and both must pass before a commit is allowed. Note the test suite
makes live LLM calls, so it needs a provider API key (e.g. `OPENAI_API_KEY`).
Bypass the hooks in an emergency with `git commit --no-verify`.

---

## 📁 License

MIT © 2025 Fredrik Linaker
