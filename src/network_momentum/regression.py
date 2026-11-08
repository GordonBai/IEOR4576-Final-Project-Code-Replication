"""Pooled OLS helpers for the future Stage-D milestone."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def build_next_day_scaled_target(
    returns: pd.DataFrame,
    daily_volatility: pd.DataFrame,
) -> pd.DataFrame:
    """At feature date t, label the next panel-date return divided by sigma_t."""
    if not returns.index.equals(daily_volatility.index) or not returns.columns.equals(
        daily_volatility.columns
    ):
        raise ValueError("returns and daily_volatility axes must match")
    target = returns.shift(-1) / daily_volatility
    return target.replace([np.inf, -np.inf], np.nan)


def fit_pooled_ols(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    expected_features: int | None = 8,
) -> LinearRegression:
    """Fit pooled OLS and attach classical finite-sample inference.

    The paper reports coefficient standard errors and t-test markers but does
    not disclose a covariance estimator.  We use the conventional
    homoskedastic OLS covariance, record the choice, and expose the resulting
    attributes on the fitted sklearn model.
    """
    if expected_features is not None and x.shape[1] != expected_features:
        raise ValueError(
            f"pooled OLS requires {expected_features} features, got {x.shape[1]}"
        )
    combined = x.join(y.rename("target"), how="inner").dropna()
    if combined.empty:
        raise ValueError("no complete training observations")
    model = LinearRegression(fit_intercept=True)
    model.fit(combined[x.columns], combined["target"])
    design = np.column_stack(
        [np.ones(len(combined), dtype=float), combined[x.columns].to_numpy(dtype=float)]
    )
    residual = combined["target"].to_numpy(dtype=float) - model.predict(combined[x.columns])
    degrees_of_freedom = max(len(combined) - design.shape[1], 1)
    residual_variance = float(residual @ residual / degrees_of_freedom)
    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    parameters = np.concatenate([[float(model.intercept_)], np.asarray(model.coef_, dtype=float)])
    t_statistics = np.divide(
        parameters,
        standard_errors,
        out=np.full_like(parameters, np.nan),
        where=standard_errors > 0,
    )
    p_values = 2.0 * stats.t.sf(np.abs(t_statistics), df=degrees_of_freedom)
    model.intercept_standard_error_ = float(standard_errors[0])
    model.standard_errors_ = standard_errors[1:]
    model.intercept_t_statistic_ = float(t_statistics[0])
    model.t_statistics_ = t_statistics[1:]
    model.intercept_p_value_ = float(p_values[0])
    model.p_values_ = p_values[1:]
    model.nobs_ = int(len(combined))
    model.residual_degrees_of_freedom_ = int(degrees_of_freedom)
    model.covariance_choice_ = "classical_homoskedastic_ols"
    return model


def predict_pooled_ols(model: LinearRegression, x: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=x.index, name="forecast", dtype=float)
    valid = x.notna().all(axis=1)
    if valid.any():
        result.loc[valid] = model.predict(x.loc[valid])
    return result
