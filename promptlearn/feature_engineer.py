import logging

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .base import BasePromptEstimator, resolve_model
from .utils import generate_feature_dicts, normalize_feature_name

logger = logging.getLogger("promptlearn")

DEFAULT_FEATURE_ENGINEERING_PROMPT_TEMPLATE = """
You are doing feature engineering for a tabular machine-learning model.

Write a single valid Python function called 'transform' that, given the feature variables of ONE row (passed as keyword arguments), returns a dict of NEW engineered features derived from the inputs. These new features should make a downstream model more accurate by encoding domain knowledge.

Input columns: {columns}
{target_line}
Guidelines:
- Derive features using real-world/domain knowledge from semantically meaningful columns (e.g. map a country to its continent or GDP-per-capita tier, parse a date into is_weekend/month, bucket ages, combine related numeric columns into ratios).
- Prefer NUMERIC output features (ints/floats); booleans are fine encoded as 0/1. Avoid free-text outputs.
- Do NOT simply copy the input columns through; only return NEW features.
- Return the SAME set of dict keys for every possible input row. Use a sensible default (e.g. 0) when a value is unknown, missing, or out-of-vocabulary, so the function never raises.
- Coerce inputs with float(x)/int(x) as needed at the top of the function before using them.
- For categorical lookups, include an exhaustive mapping (aim for 100+ keys where relevant: countries, US states, common animals, colors, etc.) with a default fallback.

Every string literal MUST be valid, properly terminated Python. If a key or value contains an apostrophe (e.g. grevy's zebra), wrap that string in double quotes ("grevy's zebra"); if it contains a double quote, wrap it in single quotes. Never leave an unterminated string literal.

The function signature must be: def transform(**features): ...

Only output valid Python code, no markdown or explanations.

Data (sample rows; the last column is the target if one is present):
{data}
"""


class PromptFeatureEngineer(TransformerMixin, BasePromptEstimator):
    """LLM-powered feature engineering as a scikit-learn transformer.

    At ``fit`` the LLM writes a standalone ``transform(**features)`` function
    that derives new, more predictive columns from semantically meaningful ones
    (the same code-generation + validation-retry approach the estimators use, so
    there are no per-row LLM calls). ``transform`` runs that function over each
    row and appends the engineered columns to the input, making it a drop-in
    preprocessing step before any classical model in a ``Pipeline``.
    """

    def __init__(
        self,
        model=None,
        verbose: bool = True,
        max_train_rows: int = 100,
        max_retries: int = 2,
    ):
        super().__init__(
            model=resolve_model(model),
            verbose=verbose,
            max_train_rows=max_train_rows,
            max_retries=max_retries,
        )
        self.new_feature_names_ = None

    def fit(self, X, y=None) -> "PromptFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "PromptFeatureEngineer requires a pandas DataFrame with named columns."
            )
        self.explanation_ = None  # invalidate any cached explanation from a prior fit

        data = X.copy()
        data.columns = [normalize_feature_name(c) for c in data.columns]
        self.feature_names_ = list(data.columns)

        # Include the target column (when supervised) so the LLM can engineer
        # features that are actually relevant to what we are predicting.
        if y is not None:
            self.target_name_ = normalize_feature_name(
                getattr(y, "name", None) or "target"
            )
            sample_source = data.copy()
            sample_source[self.target_name_] = y.values if hasattr(y, "values") else y
            target_line = (
                f"The downstream model predicts the column '{self.target_name_}'. "
                "Engineer features that help predict it.\n"
            )
        else:
            self.target_name_ = None
            sample_source = data
            target_line = "No target is provided; engineer broadly useful features.\n"

        if len(sample_source) > self.max_train_rows:
            logger.info(
                f"Reducing training data from {sample_source.shape[0]:,} to "
                f"{self.max_train_rows:,} rows for LLM."
            )
            sample_df = sample_source.sample(self.max_train_rows, random_state=42)
        else:
            sample_df = sample_source

        base_prompt = DEFAULT_FEATURE_ENGINEERING_PROMPT_TEMPLATE.format(
            data=sample_df.to_csv(index=False),
            columns=", ".join(self.feature_names_),
            target_line=target_line,
        )
        logger.info(f"[LLM Prompt]\n{base_prompt}")

        validation_rows = list(
            generate_feature_dicts(data[self.feature_names_], self.feature_names_)
        )

        raw_code, extended_code, fn = self._generate_code(base_prompt, validation_rows)
        self.raw_python_code_ = raw_code
        self.python_code_ = extended_code
        self.predict_fn = fn
        self.new_feature_names_ = self._infer_new_feature_names(validation_rows)
        return self

    def transform(self, X) -> pd.DataFrame:
        if self.predict_fn is None:
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "PromptFeatureEngineer requires a pandas DataFrame with named columns."
            )

        X_norm = X.copy()
        X_norm.columns = [normalize_feature_name(c) for c in X_norm.columns]

        rows = [
            self._safe_transform_row(feats)
            for feats in generate_feature_dicts(
                X_norm[self.feature_names_], self.feature_names_
            )
        ]
        new_df = pd.DataFrame(rows, index=X.index)
        if self.new_feature_names_:
            new_df = new_df.reindex(columns=self.new_feature_names_)
        # Never overwrite an existing input column.
        new_df = new_df[[c for c in new_df.columns if c not in X.columns]]
        return pd.concat([X, new_df], axis=1)

    def get_feature_names_out(self, input_features=None):
        base = (
            list(input_features)
            if input_features is not None
            else list(self.feature_names_ or [])
        )
        extra = [c for c in (self.new_feature_names_ or []) if c not in base]
        return np.asarray(base + extra, dtype=object)

    def _validate_predict_fn(self, predict_fn, rows: list) -> None:
        """Confirm the generated function returns a consistent dict of features."""
        if not rows:
            raise ValueError(
                "PromptFeatureEngineer needs at least one training row to validate "
                "the generated features."
            )
        keysets = set()
        first = None
        for row in rows[:25]:
            out = predict_fn(**row)
            if not isinstance(out, dict):
                raise ValueError(
                    "transform() must return a dict of new features, but returned "
                    f"{type(out).__name__}."
                )
            keysets.add(tuple(sorted(out.keys())))
            first = out
        if len(keysets) > 1:
            raise ValueError(
                "transform() must return the same set of feature keys for every row."
            )
        if not first:
            raise ValueError("transform() returned no new features.")

    def _infer_new_feature_names(self, rows: list) -> list:
        for row in rows:
            try:
                out = self.predict_fn(**row)
            except Exception:
                continue
            if isinstance(out, dict) and out:
                return [k for k in out.keys() if k not in self.feature_names_]
        return []

    def _safe_transform_row(self, features: dict) -> dict:
        try:
            out = self.predict_fn(**features)
            if isinstance(out, dict):
                return out
        except Exception as e:
            logger.error(f"[FeatureEngineer ERROR] {e} on features={features}")
        return {k: np.nan for k in (self.new_feature_names_ or [])}


def _logreg_xgboost_gap(X: pd.DataFrame, y, cv: int) -> float:
    """Return mean CV accuracy of xgboost minus logreg on X, y."""
    try:
        from xgboost import XGBClassifier

        xgb_cls = XGBClassifier(n_estimators=100, eval_metric="logloss", verbosity=0)
    except ImportError:
        xgb_cls = None

    cat_cols = [
        c
        for c in X.columns
        if X[c].dtype == object
        or str(X[c].dtype) in ("category", "string", "str")
        or pd.api.types.is_string_dtype(X[c])
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]

    def make_pipeline(clf):
        from sklearn.compose import ColumnTransformer

        transformers = []
        if num_cols:
            transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            (
                                "enc",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    cat_cols,
                )
            )
        if not transformers:
            return Pipeline([("clf", clf)])
        ct = ColumnTransformer(transformers, remainder="drop")
        return Pipeline([("pre", ct), ("clf", clf)])

    lr = make_pipeline(LogisticRegression(max_iter=1000, solver="lbfgs"))
    lr_score = float(np.mean(cross_val_score(lr, X, y, cv=cv, scoring="accuracy")))

    if xgb_cls is None:
        return 0.0

    xgb = make_pipeline(xgb_cls)
    xgb_score = float(np.mean(cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")))
    return xgb_score - lr_score


class AdaptiveFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineer that decides at fit-time whether to apply FE.

    Estimates the logreg–xgboost accuracy gap via cross-validation on the fit
    data. If the gap exceeds ``gap_threshold``, feature engineering is likely
    to improve a downstream linear model and ``PromptFeatureEngineer`` is
    applied; otherwise the transformer passes data through unchanged.

    Attributes after fit:
        gap_ (float): estimated logreg–xgboost accuracy gap.
        fe_enabled_ (bool): whether FE was applied.
        fe_ (PromptFeatureEngineer | None): fitted FE instance, or None.
    """

    def __init__(
        self,
        model=None,
        gap_threshold: float = 0.05,
        cv: int = 3,
        verbose: bool = True,
        max_train_rows: int = 100,
        max_retries: int = 2,
    ):
        self.model = model
        self.gap_threshold = gap_threshold
        self.cv = cv
        self.verbose = verbose
        self.max_train_rows = max_train_rows
        self.max_retries = max_retries

    def fit(self, X: pd.DataFrame, y=None) -> "AdaptiveFeatureEngineer":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("AdaptiveFeatureEngineer requires a pandas DataFrame.")

        self.gap_ = _logreg_xgboost_gap(X, y, cv=self.cv)
        self.fe_enabled_ = self.gap_ > self.gap_threshold

        if self.verbose:
            logger.info(
                "[AdaptiveFE] gap=%.4f threshold=%.4f → FE %s",
                self.gap_,
                self.gap_threshold,
                "ENABLED" if self.fe_enabled_ else "DISABLED (pass-through)",
            )

        if self.fe_enabled_:
            self.fe_ = PromptFeatureEngineer(
                model=self.model,
                verbose=self.verbose,
                max_train_rows=self.max_train_rows,
                max_retries=self.max_retries,
            )
            self.fe_.fit(X, y)
        else:
            self.fe_ = None

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fe_enabled_"):
            raise RuntimeError("Call fit() before transform().")
        if self.fe_enabled_ and self.fe_ is not None:
            return self.fe_.transform(X)
        return X

    def get_feature_names_out(self, input_features=None):
        if self.fe_enabled_ and self.fe_ is not None:
            return self.fe_.get_feature_names_out(input_features)
        base = list(input_features) if input_features is not None else []
        return np.asarray(base, dtype=object)
