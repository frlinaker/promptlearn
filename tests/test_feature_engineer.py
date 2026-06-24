"""Tests for PromptFeatureEngineer.

The mocked tests stub the two LLM touchpoints (``_call_llm`` and
``_extend_code``) so they run without network. One live smoke test exercises a
real provider (the cheap test model from conftest).
"""

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from promptlearn import PromptFeatureEngineer

# A simple, valid transform the mocked LLM "returns".
GOOD_CODE = (
    "def transform(**features):\n"
    "    country = str(features.get('country', '')).lower()\n"
    "    gdp = {'sweden': 60000, 'india': 2500}.get(country, 0)\n"
    "    age = float(features.get('age', 0) or 0)\n"
    "    return {'gdp_per_capita': gdp, 'is_adult': 1 if age >= 18 else 0}\n"
)


def _mock(fe, code=GOOD_CODE):
    fe._call_llm = lambda prompt: code
    fe._extend_code = lambda c: c
    return fe


@pytest.fixture
def Xy():
    X = pd.DataFrame(
        {"country": ["sweden", "india", "sweden", "india"], "age": [30, 12, 41, 9]}
    )
    y = pd.Series([1, 0, 1, 0], name="label")
    return X, y


def test_fit_transform_appends_features(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X, y)
    assert fe.new_feature_names_ == ["gdp_per_capita", "is_adult"]
    out = fe.transform(X)
    # original columns preserved, new ones appended
    assert list(out.columns) == ["country", "age", "gdp_per_capita", "is_adult"]
    assert out.loc[0, "gdp_per_capita"] == 60000
    assert list(out["is_adult"]) == [1, 0, 1, 0]
    assert len(out) == len(X)


def test_fit_unsupervised_without_y(Xy):
    X, _ = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X)
    assert fe.target_name_ is None
    assert "gdp_per_capita" in fe.transform(X).columns


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        PromptFeatureEngineer(verbose=False).transform(pd.DataFrame({"a": [1]}))


def test_requires_dataframe(Xy):
    _, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False))
    with pytest.raises(ValueError, match="DataFrame"):
        fe.fit(np.array([[1, 2], [3, 4]]), y)


def test_transform_row_failure_yields_nan(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X, y)
    # Replace the compiled fn with one that raises, to exercise the safe path.
    fe.predict_fn = lambda **f: (_ for _ in ()).throw(RuntimeError("boom"))
    out = fe.transform(X)
    assert out["gdp_per_capita"].isna().all()


def test_validation_rejects_nondict_output(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False, max_retries=0))
    fe._call_llm = lambda prompt: "def transform(**features):\n    return 42\n"
    with pytest.raises(ValueError, match="must return a dict"):
        fe.fit(X, y)


def test_validation_rejects_inconsistent_keys(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False, max_retries=0))
    code = (
        "def transform(**features):\n"
        "    age = float(features.get('age', 0) or 0)\n"
        "    return {'a': 1} if age >= 18 else {'b': 2}\n"
    )
    fe._call_llm = lambda prompt: code
    with pytest.raises(ValueError, match="same set of feature keys"):
        fe.fit(X, y)


def test_pipeline_with_downstream_model(Xy):
    X, y = Xy
    fe = _mock(PromptFeatureEngineer(verbose=False)).fit(X, y)
    # Use only the engineered numeric features downstream.
    engineered = fe.transform(X)[fe.new_feature_names_]
    clf = LogisticRegression().fit(engineered, y)
    assert clf.score(engineered, y) == 1.0


def test_joblib_roundtrip_recompiles(Xy):
    X, _ = Xy
    # Build fitted state directly (no instance-level mocks that would pollute
    # __dict__ and break pickling).
    fe = PromptFeatureEngineer(verbose=False)
    fe.feature_names_ = ["country", "age"]
    fe.target_name_ = "label"
    fe.raw_python_code_ = GOOD_CODE
    fe.python_code_ = GOOD_CODE
    fe.new_feature_names_ = ["gdp_per_capita", "is_adult"]

    buffer = io.BytesIO()
    joblib.dump(fe, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    # predict_fn is dropped on dump and recompiled from python_code_ on load.
    out = restored.transform(X)
    assert out.loc[0, "gdp_per_capita"] == 60000
    assert restored.new_feature_names_ == ["gdp_per_capita", "is_adult"]


def test_live_feature_engineering():
    """Smoke test against a real provider via the conftest test model."""
    X = pd.DataFrame(
        {
            "country": ["Sweden", "India", "Japan", "Brazil"],
            "age": [30, 12, 41, 17],
        }
    )
    y = pd.Series([1, 0, 1, 0], name="label")
    fe = PromptFeatureEngineer(verbose=False).fit(X, y)
    out = fe.transform(X)
    # New columns were added, original rows preserved.
    assert len(out) == len(X)
    assert len(out.columns) > len(X.columns)
    assert fe.new_feature_names_
