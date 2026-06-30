"""Tests for dataset_description and web_search features."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from promptlearn import PromptClassifier, PromptRegressor

SIMPLE_CODE = "def predict(**f): return 0"


def _mock_llm(code=SIMPLE_CODE):
    """Return a mock that makes _call_llm return valid Python code."""
    m = MagicMock(return_value=code)
    return m


# ---------------------------------------------------------------------------
# dataset_description
# ---------------------------------------------------------------------------


def test_classifier_description_in_prompt(monkeypatch):
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False)
    calls = []

    def fake_call_llm(prompt, web_search=False):
        calls.append({"prompt": prompt, "web_search": web_search})
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"age": [25, 40], "income": [30000, 80000]})
    y = pd.Series([0, 1])
    clf.fit(X, y, dataset_description="UCI Adult: predict income >50k.")

    first = calls[0]
    assert "Dataset context" in first["prompt"]
    assert "UCI Adult" in first["prompt"]
    assert first["web_search"] is False


def test_regressor_description_in_prompt(monkeypatch):
    reg = PromptRegressor(model="gpt-5.4-mini", verbose=False)
    calls = []

    def fake_call_llm(prompt, web_search=False):
        calls.append({"prompt": prompt})
        return "def predict(**f): return 1.0"

    monkeypatch.setattr(reg, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"height_m": [1.0, 5.0]})
    y = pd.Series([0.45, 1.01])
    reg.fit(X, y, dataset_description="Falling body: predict fall time in seconds.")

    assert "Dataset context" in calls[0]["prompt"]
    assert "Falling body" in calls[0]["prompt"]


def test_no_description_prompt_unchanged(monkeypatch):
    clf = PromptClassifier(model="gpt-5.4-mini", verbose=False)
    captured = {}

    def fake_call_llm(prompt, web_search=False):
        captured["prompt"] = prompt
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "Dataset context:" not in captured["prompt"]


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def test_web_search_passed_to_call_llm(monkeypatch):
    clf = PromptClassifier(model="gpt-5.5", verbose=False, web_search=True)
    captured = []

    def fake_call_llm(prompt, web_search=False):
        captured.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    # First call (fit) should have web_search=True; extend call should not.
    assert captured[0] is True
    assert all(not v for v in captured[1:])


def test_web_search_false_by_default(monkeypatch):
    clf = PromptClassifier(model="gpt-5.5", verbose=False)
    captured = []

    def fake_call_llm(prompt, web_search=False):
        captured.append(web_search)
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert all(not v for v in captured)


def test_web_search_prompt_prefix(monkeypatch):
    clf = PromptClassifier(model="gpt-5.5", verbose=False, web_search=True)
    captured = {}

    def fake_call_llm(prompt, web_search=False):
        if not captured:
            captured["prompt"] = prompt
        return SIMPLE_CODE

    monkeypatch.setattr(clf, "_call_llm", fake_call_llm)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "search the web" in captured["prompt"].lower()


def test_web_search_unsupported_model_warns(monkeypatch, caplog):
    import logging

    clf = PromptClassifier(model="claude-sonnet-4-6", verbose=False, web_search=True)

    calls = []

    def fake_completion(model, messages, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])

    with caplog.at_level(logging.WARNING, logger="promptlearn"):
        clf.fit(X, y)

    assert "not in the known supported list" in caplog.text
    # web_search_options should NOT have been passed to litellm
    assert all("web_search_options" not in c for c in calls)


def test_web_search_supported_model_passes_options(monkeypatch):
    """Chat Completions path: gpt-4o-search-preview gets web_search_options."""
    clf = PromptClassifier(
        model="gpt-4o-search-preview", verbose=False, web_search=True
    )

    calls = []

    def fake_completion(model, messages, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = SIMPLE_CODE
        return resp

    monkeypatch.setattr("litellm.completion", fake_completion)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert "web_search_options" in calls[0]


def test_web_search_responses_api_model(monkeypatch):
    """Responses API path: gpt-5.5 uses litellm.responses with web_search tool."""
    clf = PromptClassifier(model="gpt-5.5", verbose=False, web_search=True)

    responses_calls = []

    def fake_responses(prompt, model, **kwargs):
        responses_calls.append({"model": model, "tools": kwargs.get("tools")})
        # Build a minimal mock response matching ResponsesAPIResponse structure.
        msg = MagicMock()
        msg.type = "message"
        content_part = MagicMock()
        content_part.type = "output_text"
        content_part.text = SIMPLE_CODE
        msg.content = [content_part]
        resp = MagicMock()
        resp.output = [msg]
        return resp

    monkeypatch.setattr("litellm.responses", fake_responses)

    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])
    clf.fit(X, y)

    assert len(responses_calls) >= 1
    assert responses_calls[0]["model"] == "gpt-5.5"
    assert {"type": "web_search"} in responses_calls[0]["tools"]
