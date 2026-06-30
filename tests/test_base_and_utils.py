import pytest
from promptlearn.base import BasePromptEstimator
from promptlearn.utils import sanitize_dataset_description


def test_get_set_params():
    est = BasePromptEstimator(model="gpt-4", verbose=True, max_train_rows=10)
    params = est.get_params()
    assert params["model"] == "gpt-4"
    est.set_params(model="gpt-3.5-turbo")
    assert est.model == "gpt-3.5-turbo"


def test_call_llm_raises(monkeypatch):
    import litellm

    est = BasePromptEstimator(model="gpt-4", verbose=False, max_train_rows=1)

    def boom(*args, **kwargs):
        raise RuntimeError("provider error")

    monkeypatch.setattr(litellm, "completion", boom)
    with pytest.raises(RuntimeError):
        est._call_llm("this should fail")


def test_call_llm_normalizes_ollama_model(monkeypatch):
    """The documented ``ollama:model`` syntax is mapped to litellm's ``ollama/model``."""
    import litellm

    captured = {}

    def fake_completion(model, messages, **kwargs):
        captured["model"] = model
        message = type("Message", (), {"content": "hello"})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})

    monkeypatch.setattr(litellm, "completion", fake_completion)
    est = BasePromptEstimator(model="ollama:llama3.1", verbose=False, max_train_rows=1)
    out = est._call_llm("hi")
    assert captured["model"] == "ollama/llama3.1"
    assert out == "hello"


def test_sanitize_description_strips_and_cleans():
    assert sanitize_dataset_description("  hello  ") == "hello"


def test_sanitize_description_removes_braces():
    result = sanitize_dataset_description("context {data} here")
    assert "{" not in result and "}" not in result
    assert "context" in result and "data" in result and "here" in result


def test_sanitize_description_truncates():
    long = "x" * 600
    result = sanitize_dataset_description(long)
    assert len(result) <= 512  # 500 chars + " [truncated]" (12 chars)
    assert result.endswith("[truncated]")


def test_sanitize_description_collapses_whitespace():
    assert sanitize_dataset_description("a  b\t\tc") == "a b c"


def test_extend_code_handles_llm_failure(monkeypatch):
    class DummyEstimator(BasePromptEstimator):
        def __init__(self):
            super().__init__(model="dummy-model", verbose=False, max_train_rows=10)

        def _call_llm(self, prompt: str):
            raise RuntimeError("Mocked LLM failure")

    estimator = DummyEstimator()
    # Should log a warning and return original code unchanged
    result = estimator._extend_code("def predict(**features): return 42")
    assert result.strip() == "def predict(**features): return 42"
