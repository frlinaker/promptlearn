"""Tests for PromptFeatureEngineer and AdaptiveFeatureEngineer.

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

from promptlearn import AdaptiveFeatureEngineer, PromptFeatureEngineer

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


# ---------------------------------------------------------------------------
# AdaptiveFeatureEngineer tests
# ---------------------------------------------------------------------------


def _make_nonlinear_dataset(n=200):
    """XOR-like dataset: logreg struggles, xgboost handles it well."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        {
            "a": rng.randint(0, 2, n).astype(float),
            "b": rng.randint(0, 2, n).astype(float),
        }
    )
    y = pd.Series((X["a"].astype(int) ^ X["b"].astype(int)).astype(int), name="label")
    return X, y


def _make_linear_dataset(n=200):
    """Linearly separable dataset: both logreg and xgboost do equally well."""
    rng = np.random.RandomState(1)
    x = rng.randn(n)
    X = pd.DataFrame({"x": x, "noise": rng.randn(n) * 0.01})
    y = pd.Series((x > 0).astype(int), name="label")
    return X, y


def _mock_adaptive(afe, code=GOOD_CODE):
    """Patch LLM calls on an AdaptiveFeatureEngineer's inner FE after gap check."""
    original_fe_fit = afe.fit

    def patched_fit(X, y=None):
        result = original_fe_fit(X, y)
        if afe.fe_ is not None:
            afe.fe_._call_llm = lambda prompt: code
            afe.fe_._extend_code = lambda c: c
            # Re-run fit on the real PromptFeatureEngineer with mocked LLM.
            afe.fe_.fit(X, y)
        return result

    afe.fit = patched_fit
    return afe


def test_adaptive_fe_enabled_on_nonlinear():
    X, y = _make_nonlinear_dataset()
    afe = AdaptiveFeatureEngineer(gap_threshold=0.05, cv=3, verbose=False)
    afe.fit(X, y)
    assert hasattr(afe, "gap_")
    assert hasattr(afe, "fe_enabled_")
    # XOR is strongly non-linear; gap should exceed threshold
    assert afe.fe_enabled_, f"expected FE enabled, gap={afe.gap_:.4f}"
    assert afe.fe_ is not None


def test_adaptive_fe_disabled_on_linear():
    X, y = _make_linear_dataset()
    afe = AdaptiveFeatureEngineer(gap_threshold=0.05, cv=3, verbose=False)
    afe.fit(X, y)
    assert hasattr(afe, "gap_")
    assert not afe.fe_enabled_, f"expected FE disabled, gap={afe.gap_:.4f}"
    assert afe.fe_ is None
    # Pass-through: output identical to input
    out = afe.transform(X)
    assert out is X


def test_adaptive_passthrough_preserves_dataframe(Xy):
    X, y = Xy
    # Force gap below threshold with a high threshold value
    afe = AdaptiveFeatureEngineer(gap_threshold=1.0, cv=2, verbose=False)
    afe.fit(X, y)
    assert not afe.fe_enabled_
    out = afe.transform(X)
    pd.testing.assert_frame_equal(out, X)


def test_adaptive_transform_before_fit_raises(Xy):
    X, _ = Xy
    afe = AdaptiveFeatureEngineer(verbose=False)
    with pytest.raises(RuntimeError, match="fit"):
        afe.transform(X)


def test_adaptive_requires_dataframe():
    afe = AdaptiveFeatureEngineer(verbose=False)
    with pytest.raises(ValueError, match="DataFrame"):
        afe.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
