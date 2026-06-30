import logging
import os
import re
import warnings

from typing import Callable, Optional

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .explain import Explanation
from .utils import (
    generate_feature_dicts,
    make_predict_fn,
    prepare_training_data,
    extract_python_code,
    parse_tsv,
    sanitize_dataset_description,
)

logger = logging.getLogger("promptlearn")

# The library default model. Used when no model is passed and the
# PROMPTLEARN_MODEL environment variable is unset.
DEFAULT_MODEL = "gpt-5.5"


def resolve_model(model: Optional[str]) -> str:
    """Resolve the model string for an estimator.

    An explicit ``model`` always wins. Otherwise the ``PROMPTLEARN_MODEL``
    environment variable is used when set (handy for pointing tests/CI at a
    cheaper, faster model), falling back to :data:`DEFAULT_MODEL`.
    """
    if model is not None:
        return model
    return os.environ.get("PROMPTLEARN_MODEL", DEFAULT_MODEL)


class BasePromptEstimator(BaseEstimator):
    def __init__(
        self,
        model: str,
        verbose: bool,
        max_train_rows: int,
        max_retries: int = 2,
        web_search: bool = False,
    ):
        self.model = model
        self.verbose = verbose
        self.max_train_rows = max_train_rows
        self.max_retries = max_retries
        self.web_search = web_search
        self.predict_fn: Optional[Callable] = None
        self.target_name_: Optional[str] = None
        self.feature_names_: Optional[list] = None
        self.raw_python_code_: Optional[str] = None
        self.python_code_: Optional[str] = None
        self.explanation_: Optional[Explanation] = None

    # used by GridSearchCV
    def get_params(self, deep=True):
        # Only include arguments that are accepted by __init__
        return {
            "model": self.model,
            "verbose": self.verbose,
            "max_train_rows": self.max_train_rows,
            "max_retries": self.max_retries,
            "web_search": self.web_search,
        }

    # used by GridSearchCV
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # used by joblib
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("predict_fn", None)  # Remove predict_fn on serialization
        return state

    # used by joblib
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recompile the generated heuristic on re-creation of the object
        if getattr(self, "python_code_", None):
            try:
                self.predict_fn = make_predict_fn(self.python_code_)
            except Exception as e:
                warnings.warn(
                    f"Failed to recompile regression function: {e}", UserWarning
                )
                self.predict_fn = None

    # Models that use the Responses API for web search (tools=[{"type": "web_search"}]).
    # GPT-5+ models on OpenAI use the Responses API; web_search_options is NOT supported.
    _WEB_SEARCH_RESPONSES_API_MODELS = {
        "gpt-5.5",
        "gpt-5.4-mini",
        "gpt-5.4",
        "gpt-5.3",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
    }

    # Models that support web search via web_search_options in the Chat Completions API.
    # OpenAI: dedicated search-preview models.
    # Google: Gemini models via Vertex AI (Google Search grounding).
    _WEB_SEARCH_CHAT_COMPLETIONS_MODELS = {
        "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview",
        "vertex_ai/gemini-2.5-pro",
        "vertex_ai/gemini-2.5-flash",
        "vertex_ai/gemini-2.5-flash-lite",
        "vertex_ai/gemini-3.5-flash",
    }

    # Union for external checks (e.g. warnings when model is unsupported).
    _WEB_SEARCH_MODELS = (
        _WEB_SEARCH_RESPONSES_API_MODELS | _WEB_SEARCH_CHAT_COMPLETIONS_MODELS
    )

    def _call_llm(self, prompt: str, web_search: bool = False) -> str:
        """Call the language model via litellm, return the response text.

        The provider is selected by the model string, e.g. ``gpt-5.5`` (OpenAI),
        ``claude-sonnet-4-6`` (Anthropic), or ``ollama:llama3.1`` (local Ollama).
        API keys are read from the usual per-provider environment variables.
        """
        import litellm

        if self.verbose:
            logger.info("[Prompt to LLM]\n%s", prompt)
        # Accept the documented ``ollama:model`` syntax; litellm expects ``ollama/model``.
        model = self.model
        if model.startswith("ollama:"):
            model = "ollama/" + model[len("ollama:") :]

        if web_search and model in self._WEB_SEARCH_RESPONSES_API_MODELS:
            # GPT-5+ uses the Responses API with tools=[{"type": "web_search"}].
            response = litellm.responses(prompt, model, tools=[{"type": "web_search"}])
            content = ""
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            content += c.text
            content = content.strip()
            if self.verbose:
                logger.info("[LLM Response]\n%s", content)
            return content

        kwargs: dict = {}
        if web_search:
            if model in self._WEB_SEARCH_CHAT_COMPLETIONS_MODELS:
                kwargs["web_search_options"] = {}
            else:
                logger.warning(
                    "web_search=True requested but model %r is not in the known "
                    "supported list %s — proceeding without web search.",
                    model,
                    self._WEB_SEARCH_MODELS,
                )

        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            content = str(response.choices[0].message.content).strip()
            if self.verbose:
                logger.info("[LLM Response]\n%s", content)
            return content
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise RuntimeError(f"LLM call failed: {e}")

    def _fit(
        self,
        X,
        y,
        prompt: str,
        synthetic_features: Optional[list] = None,
        dataset_description: Optional[str] = None,
    ):
        data, self.feature_names_, self.target_name_ = prepare_training_data(X, y)
        self.explanation_ = None  # invalidate any cached explanation from a prior fit

        # Use a small sample for LLM to avoid expensive calls
        if len(data) > self.max_train_rows:
            logger.info(
                f"Reducing training data from {data.shape[0]:,} to {self.max_train_rows:,} rows for LLM."
            )
            sample_df = data.sample(self.max_train_rows, random_state=42)
        else:
            sample_df = data

        csv_data = sample_df.to_csv(index=False)

        if synthetic_features:
            synthetic_note = (
                f"\nNote: the following columns are SYNTHETIC features pre-computed by a "
                f"feature engineering step: {', '.join(synthetic_features)}. "
                "Treat them as weak hints only — do NOT build your primary logic around them. "
                "Base your prediction mainly on the original (non-synthetic) columns.\n"
            )
            prompt = synthetic_note + prompt

        base_prompt = prompt.replace("{data}", csv_data)

        if dataset_description:
            base_prompt = (
                f"--- Dataset context ---\n{sanitize_dataset_description(dataset_description)}\n"
                f"--- End context ---\n\n"
            ) + base_prompt

        if self.web_search:
            base_prompt = (
                "You may search the web to look up information about these features "
                "and their real-world relationships before writing the predict function.\n\n"
            ) + base_prompt

        self.fit_prompt_ = base_prompt
        logger.info(f"[LLM Prompt]\n{base_prompt}")

        validation_rows = (
            list(
                generate_feature_dicts(
                    sample_df[self.feature_names_], self.feature_names_
                )
            )
            if self.feature_names_
            else []
        )

        raw_code, extended_code, predict_fn = self._generate_code(
            base_prompt, validation_rows, web_search=self.web_search
        )
        self.raw_python_code_ = raw_code
        self.python_code_ = extended_code
        self.predict_fn = predict_fn
        return self

    def _generate_code(
        self, base_prompt: str, validation_rows: list, web_search: bool = False
    ):
        """Generate code from the LLM with a validation-and-retry loop.

        Returns ``(raw_code, extended_code, fn)``. Each attempt compiles the
        code and runs ``_validate_predict_fn`` over the sample rows; any failure
        triggers a retry with the error fed back to the LLM. Subclasses can
        override ``_validate_predict_fn`` to add stricter checks. Raises the last
        error if every attempt (one initial plus ``max_retries``) fails.
        """
        feedback = ""
        last_error: Optional[Exception] = None
        # One initial attempt plus up to max_retries corrective re-tries.
        for attempt in range(self.max_retries + 1):
            # Web search only on the first attempt; retries don't need it.
            code = self._call_llm(
                base_prompt + feedback, web_search=web_search and attempt == 0
            )
            if not isinstance(code, str):
                code = str(code)
            logger.info(f"[LLM Output]\n{code}")

            # Remove markdown/code block if present (triple backticks)
            code = extract_python_code(code)
            try:
                if not code.strip():
                    raise ValueError("No code to exec from LLM output.")
                raw_code = code
                extended_code = self._extend_code(code)
                fn = make_predict_fn(extended_code)
                self._validate_predict_fn(fn, validation_rows)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Validation] Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}"
                )
                feedback = (
                    "\n\nThe Python function you previously returned failed validation "
                    f"with this error:\n{e}\n\n"
                    "Fix the problem and return only the corrected, valid Python code."
                )
                continue

            return raw_code, extended_code, fn

        # Every attempt failed; surface the most recent error.
        assert last_error is not None
        raise last_error

    def _validate_predict_fn(self, predict_fn: Callable, rows: list) -> None:
        """Run the compiled function over the training sample to confirm it
        executes without raising. Any exception is treated as a validation
        failure so ``_fit`` can retry with the error fed back to the LLM."""
        for row in rows:
            predict_fn(**row)

    _EXTEND_MAX_RETRIES = 5

    def _extend_code(self, code: str) -> str:
        """Expand categorical mappings in the generated code via a second LLM pass.

        Validates the result and retries up to _EXTEND_MAX_RETRIES times when the
        LLM returns broken code, feeding the error back each time.  Falls back to
        the original code if all attempts fail.
        """
        logger.info("[Post-Process] Expanding code via second LLM pass...")
        base_prompt = (
            "The following function may use a dictionary, set, or mapping based on domain knowledge (e.g., country names, animal types).\n"
            "Please re-write the function to extend any such mappings with many more possible real-world keys, if applicable.\n"
            "Try to figure out the logic of the function based on the variable names and values that are processed in the function.\n"
            "Avoid changing the logic or structure beyond extending categorical support.\n"
            "Only return valid Python code.\n\n"
            f"{code}"
        )
        feedback = ""
        for attempt in range(self._EXTEND_MAX_RETRIES):
            try:
                refined_code = self._call_llm(base_prompt + feedback)
                refined_code = extract_python_code(str(refined_code))
                compile(refined_code, "<extend>", "exec")
                make_predict_fn(refined_code)
                logger.info("[Post-Process] Successfully extended function.")
                return refined_code
            except Exception as e:
                logger.warning(
                    "[Post-Process] Attempt %d/%d failed: %s",
                    attempt + 1, self._EXTEND_MAX_RETRIES, e,
                )
                feedback = (
                    f"\n\nThe code you returned has an error: {e}\n"
                    "Fix it and return only valid Python code."
                )
        logger.warning("[Post-Process] All extend attempts failed; using original code.")
        return code

    def sample(self, n: int = 5):
        """Generate n synthetic examples that illustrate the heuristic."""
        # Check that columns have some sort of names
        if (
            not hasattr(self, "feature_names_")
            or self.feature_names_ is None
            or not hasattr(self, "target_name_")
            or self.target_name_ is None
        ):
            raise RuntimeError(
                "Call fit() before sample(): feature names or target name not set."
            )
        prompt = (
            f"{self.python_code_}\n\n"
            f"Please generate {n} example rows in tabular format with the following columns:\n"
            f"{', '.join(self.feature_names_ + [self.target_name_])}.\n"
            f"Use tab-separated format. Do not explain."
        )
        text = self._call_llm(prompt)
        return parse_tsv(text)

    def explain(self, X=None) -> Explanation:
        """Return a plain-English explanation of the fitted heuristic.

        With no argument, returns a **global** explanation of the rule the model
        encodes (cached, so repeated calls are deterministic). Given a single-row
        ``X``, returns a **local** explanation of that one prediction.
        """
        if not getattr(self, "python_code_", None):
            raise NotFittedError("Call fit() before explain().")

        features_used = self._features_used()

        if X is not None:
            instance = next(iter(generate_feature_dicts(X, self.feature_names_)), {})
            summary = self._call_llm(self._local_explain_prompt(instance))
            return Explanation(
                meta=self._explanation_meta(["local"]),
                data={
                    "summary": summary.strip(),
                    "features_used": features_used,
                    "instance": instance,
                },
            )

        # Global explanation is computed once and cached for determinism.
        if self.explanation_ is None:
            summary = self._call_llm(self._global_explain_prompt())
            self.explanation_ = Explanation(
                meta=self._explanation_meta(["global"]),
                data={
                    "summary": summary.strip(),
                    "features_used": features_used,
                    "code": self.python_code_,
                },
            )
        return self.explanation_

    def _explanation_meta(self, scope: list) -> dict:
        # The generated Python heuristic is fully visible, so this is a whitebox
        # explanation in the Alibi sense.
        return {
            "name": type(self).__name__,
            "type": ["whitebox"],
            "explanations": scope,
            "params": self.get_params(),
        }

    def _features_used(self) -> list:
        """Features the heuristic actually references — never invents new ones."""
        names = self.feature_names_ or []
        code = self.python_code_ or ""
        used = [n for n in names if re.search(rf"\b{re.escape(n)}\b", code)]
        return used or list(names)

    def _global_explain_prompt(self) -> str:
        return (
            "You are documenting a trained model for an interpretability report. "
            "Below is the Python function it uses to make predictions. In clear, "
            "concise plain English, describe the rule it encodes: which input "
            "features it uses and how they determine the output. Be faithful to "
            "the code and do not mention features that are not present.\n\n"
            f"Target: {self.target_name_}\n"
            f"Features: {', '.join(self.feature_names_ or [])}\n\n"
            f"{self.python_code_}"
        )

    def _local_explain_prompt(self, instance: dict) -> str:
        return (
            "Below is the Python function a trained model uses to make "
            "predictions, followed by one specific input. In plain English, "
            "explain why this particular input yields its prediction, referring "
            "to the relevant feature values. Be faithful to the code.\n\n"
            f"{self.python_code_}\n\n"
            f"Input: {instance}"
        )
