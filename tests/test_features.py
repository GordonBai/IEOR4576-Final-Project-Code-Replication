from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from network_momentum.config import FeatureConfig
from network_momentum.features import (
    build_stage_a_features,
    compute_ewm_volatility,
    validate_stage_a_features,
)


def synthetic_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01", periods=900, name="date")
    assets = pd.Index(["A", "B", "C"], name="asset")
    returns = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(assets))), index=dates, columns=assets)
    returns.loc[dates[100], "B"] = np.nan
    returns.loc[dates[300], "C"] = np.nan
    prices = 100.0 * (1.0 + returns.fillna(0.0)).cumprod()
    prices = prices.mask(returns.isna())
    availability = returns.notna()
    return returns, prices, availability


def test_ewm_volatility_is_causal() -> None:
    returns, _, _ = synthetic_panels()
    config = FeatureConfig()
    full, _ = compute_ewm_volatility(returns, config)
    cutoff = returns.index[500]
    truncated, _ = compute_ewm_volatility(returns.loc[:cutoff], config)
    assert_frame_equal(full.loc[:cutoff], truncated)


def test_ewm_volatility_matches_appendix_normalized_weights() -> None:
    dates = pd.date_range("2020-01-01", periods=5, name="date")
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.00, 0.02]}, index=dates)
    config = FeatureConfig(vol_span=3, vol_min_periods=1)
    daily, _ = compute_ewm_volatility(returns, config)
    alpha = 2.0 / (config.vol_span + 1.0)
    values = returns["A"].to_numpy()
    weights = np.array([(1.0 - alpha) ** lag for lag in range(len(values))])
    weights = weights[::-1]
    mean = np.sum(weights * values) / np.sum(weights)
    expected = np.sqrt(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
    assert np.isclose(daily.iloc[-1, 0], expected, atol=1e-15)


def test_stage_a_has_exactly_eight_finite_bounded_features() -> None:
    returns, prices, availability = synthetic_panels()
    config = FeatureConfig()
    artifacts = build_stage_a_features(returns, prices, availability, config)
    report = validate_stage_a_features(artifacts, config)
    assert report["status"] == "PASS"
    assert artifacts.u_winsorized.shape[1] == 8
    assert list(artifacts.u_winsorized.columns) == list(config.feature_names)
    assert report["winsor_lower_bound_check"]
    assert report["winsor_upper_bound_check"]


def test_complete_feature_history_does_not_change_when_future_is_removed() -> None:
    returns, prices, availability = synthetic_panels()
    config = FeatureConfig()
    cutoff = returns.index[700]
    full = build_stage_a_features(returns, prices, availability, config).u_winsorized
    truncated = build_stage_a_features(
        returns.loc[:cutoff], prices.loc[:cutoff], availability.loc[:cutoff], config
    ).u_winsorized
    assert_frame_equal(full.loc[(slice(None, cutoff), slice(None)), :], truncated)
