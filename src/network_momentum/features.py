"""Causal Stage-A features for the Network Momentum replication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FeatureConfig


@dataclass(frozen=True)
class StageAFeatures:
    daily_volatility: pd.DataFrame
    annualized_volatility: pd.DataFrame
    vol_scaled_returns: pd.DataFrame
    macd_features: pd.DataFrame
    u_raw: pd.DataFrame
    winsor_lower: pd.DataFrame
    winsor_upper: pd.DataFrame
    u_winsorized: pd.DataFrame


def _observed_rolling(
    series: pd.Series,
    window: int,
    min_periods: int,
    operation: str,
    ddof: int = 1,
) -> pd.Series:
    """Roll over an asset's own observed trading-day sequence, then reindex."""
    observed = series.dropna()
    if operation == "std":
        transformed = observed.rolling(window, min_periods=min_periods).std(ddof=ddof)
    elif operation == "compound":
        transformed = (
            observed.add(1.0)
            .rolling(window, min_periods=min_periods)
            .apply(np.prod, raw=True)
            .sub(1.0)
        )
    else:
        raise ValueError(f"Unsupported rolling operation: {operation}")
    return transformed.reindex(series.index)


def _observed_rolling_frame(
    frame: pd.DataFrame,
    window: int,
    min_periods: int,
    operation: str,
    ddof: int = 1,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: _observed_rolling(
                frame[column], window, min_periods, operation, ddof=ddof
            )
            for column in frame.columns
        },
        index=frame.index,
    )


def compute_ewm_volatility(
    returns: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily and annualized 60-span EWM standard deviations."""
    daily = returns.ewm(
        span=config.vol_span,
        adjust=config.vol_ewm_adjust,
        ignore_na=config.vol_ewm_ignore_na,
        min_periods=config.vol_min_periods,
    ).std(bias=config.vol_bias)
    daily = daily.mask(~returns.notna()).replace([np.inf, -np.inf], np.nan)
    annualized = daily * np.sqrt(config.trading_days)
    daily.index.name = annualized.index.name = "date"
    daily.columns.name = annualized.columns.name = "asset"
    return daily, annualized


def compute_horizon_return(
    returns: pd.DataFrame,
    horizon: int,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Compute the configurable multi-period simple return ending at t."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if config.multi_period_return_method != "compound_observed":
        raise ValueError(
            "Only 'compound_observed' is implemented; choose explicitly in config"
        )
    if config.rolling_calendar != "asset_observations":
        raise ValueError(
            "Only 'asset_observations' is implemented; choose explicitly in config"
        )
    if horizon == 1:
        return returns.copy()
    return _observed_rolling_frame(
        returns,
        window=horizon,
        min_periods=horizon,
        operation="compound",
    )


def _wide_to_series(frame: pd.DataFrame, name: str) -> pd.Series:
    series = frame.stack(future_stack=True)
    series.index = series.index.set_names(["date", "asset"])
    return series.rename(name)


def compute_vol_scaled_return_features(
    returns: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    pieces: list[pd.Series] = []
    for horizon in config.momentum_horizons:
        horizon_return = compute_horizon_return(returns, horizon, config)
        feature = horizon_return / (daily_volatility * np.sqrt(horizon))
        if config.mask_features_to_return_availability:
            feature = feature.mask(~returns.notna())
        feature = feature.replace([np.inf, -np.inf], np.nan)
        pieces.append(_wide_to_series(feature, f"ret_{horizon}d_scaled"))
    panel = pd.concat(pieces, axis=1).sort_index()
    return panel


def _ewm_price(series: pd.Series, window: int, config: FeatureConfig) -> pd.Series:
    observed = series.dropna()
    smoothed = observed.ewm(
        alpha=1.0 / window,
        adjust=config.macd_ewm_adjust,
        ignore_na=config.macd_ewm_ignore_na,
        min_periods=config.macd_ewm_min_periods,
    ).mean()
    return smoothed.reindex(series.index)


def compute_macd_features(
    prices: pd.DataFrame,
    availability: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    price_std = _observed_rolling_frame(
        prices,
        window=config.macd_price_std_window,
        min_periods=config.macd_price_std_min_periods,
        operation="std",
        ddof=config.rolling_ddof,
    )

    pieces: list[pd.Series] = []
    for short, long in config.macd_windows:
        short_ewm = pd.DataFrame(
            {column: _ewm_price(prices[column], short, config) for column in prices.columns},
            index=prices.index,
        )
        long_ewm = pd.DataFrame(
            {column: _ewm_price(prices[column], long, config) for column in prices.columns},
            index=prices.index,
        )
        normalized_macd = (short_ewm - long_ewm) / price_std
        normalized_macd_std = _observed_rolling_frame(
            normalized_macd,
            window=config.macd_normalized_std_window,
            min_periods=config.macd_normalized_std_min_periods,
            operation="std",
            ddof=config.rolling_ddof,
        )
        feature = normalized_macd / normalized_macd_std
        if config.mask_features_to_return_availability:
            feature = feature.mask(~availability)
        feature = feature.replace([np.inf, -np.inf], np.nan)
        pieces.append(_wide_to_series(feature, f"macd_{short}_{long}"))

    return pd.concat(pieces, axis=1).sort_index()


def build_raw_feature_panel(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    availability: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaled = compute_vol_scaled_return_features(returns, daily_volatility, config)
    macd = compute_macd_features(prices, availability, config)
    raw = pd.concat([scaled, macd], axis=1).sort_index()
    raw = raw.loc[:, list(config.feature_names)]
    return scaled, macd, raw


def causal_winsorize(
    feature_panel: pd.DataFrame,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Winsorize each asset/feature using only information available by t."""
    if list(feature_panel.index.names) != ["date", "asset"]:
        raise ValueError("feature_panel index must be ['date', 'asset']")

    winsorized: dict[str, pd.Series] = {}
    lowers: dict[str, pd.Series] = {}
    uppers: dict[str, pd.Series] = {}

    for feature in feature_panel.columns:
        wide = feature_panel[feature].unstack("asset")
        mean = wide.ewm(
            halflife=config.winsor_halflife,
            adjust=config.winsor_ewm_adjust,
            ignore_na=config.winsor_ewm_ignore_na,
            min_periods=config.winsor_min_periods,
        ).mean()
        std = wide.ewm(
            halflife=config.winsor_halflife,
            adjust=config.winsor_ewm_adjust,
            ignore_na=config.winsor_ewm_ignore_na,
            min_periods=config.winsor_min_periods,
        ).std(bias=config.winsor_bias)
        if not config.winsor_include_current:
            mean = mean.shift(1)
            std = std.shift(1)
        lower = mean - config.winsor_multiple * std
        upper = mean + config.winsor_multiple * std
        clipped = wide.clip(lower=lower, upper=upper, axis=None).mask(wide.isna())
        winsorized[feature] = _wide_to_series(clipped, feature)
        lowers[feature] = _wide_to_series(lower.mask(wide.isna()), feature)
        uppers[feature] = _wide_to_series(upper.mask(wide.isna()), feature)

    return (
        pd.DataFrame(winsorized).sort_index(),
        pd.DataFrame(lowers).sort_index(),
        pd.DataFrame(uppers).sort_index(),
    )


def build_stage_a_features(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    availability: pd.DataFrame,
    config: FeatureConfig,
) -> StageAFeatures:
    daily_vol, annual_vol = compute_ewm_volatility(returns, config)
    scaled, macd, raw = build_raw_feature_panel(
        returns, prices, availability, daily_vol, config
    )
    winsorized, lower, upper = causal_winsorize(raw, config)
    return StageAFeatures(
        daily_volatility=daily_vol,
        annualized_volatility=annual_vol,
        vol_scaled_returns=scaled,
        macd_features=macd,
        u_raw=raw,
        winsor_lower=lower,
        winsor_upper=upper,
        u_winsorized=winsorized,
    )


def validate_stage_a_features(
    artifacts: StageAFeatures,
    config: FeatureConfig,
) -> dict[str, Any]:
    panel = artifacts.u_winsorized
    raw = artifacts.u_raw
    expected = list(config.feature_names)
    finite = panel.replace([np.inf, -np.inf], np.nan).notna().equals(panel.notna())
    lower_ok = (
        panel.ge(artifacts.winsor_lower) | panel.isna() | artifacts.winsor_lower.isna()
    ).all().all()
    upper_ok = (
        panel.le(artifacts.winsor_upper) | panel.isna() | artifacts.winsor_upper.isna()
    ).all().all()
    raw_missing_preserved = bool((~raw.notna() | panel.notna()).all().all())

    failures: list[str] = []
    if list(panel.columns) != expected:
        failures.append(f"expected feature columns {expected}, got {list(panel.columns)}")
    if list(panel.index.names) != ["date", "asset"]:
        failures.append(f"unexpected panel index names: {panel.index.names}")
    if not panel.index.is_monotonic_increasing or not panel.index.is_unique:
        failures.append("feature panel index is not sorted and unique")
    if not finite:
        failures.append("feature panel contains infinite values")
    if not lower_ok or not upper_ok:
        failures.append("winsorized features violate stored bounds")
    if not raw_missing_preserved:
        failures.append("winsorization created a value where the raw feature was missing")
    if failures:
        raise AssertionError("Stage-A feature QA failed: " + "; ".join(failures))

    complete_rows = panel.notna().all(axis=1)
    complete_by_date = complete_rows.groupby(level="date").sum()
    return {
        "status": "PASS",
        "feature_count": int(panel.shape[1]),
        "feature_names": expected,
        "panel_rows": int(panel.shape[0]),
        "nonmissing_by_feature": {
            column: int(value) for column, value in panel.notna().sum().items()
        },
        "complete_asset_dates": int(complete_rows.sum()),
        "first_date_with_complete_u": complete_by_date[complete_by_date.gt(0)].index.min().isoformat(),
        "last_date_with_complete_u": complete_by_date[complete_by_date.gt(0)].index.max().isoformat(),
        "max_assets_with_complete_u_on_one_date": int(complete_by_date.max()),
        "finite_observed_features": bool(finite),
        "winsor_lower_bound_check": bool(lower_ok),
        "winsor_upper_bound_check": bool(upper_ok),
        "raw_missingness_preserved": raw_missing_preserved,
    }


def validate_stage_a_causality_by_truncation(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    availability: pd.DataFrame,
    full_artifacts: StageAFeatures,
    config: FeatureConfig,
    cutoff: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Prove that deleting the future leaves all Stage-A values through t unchanged."""
    if cutoff is None:
        cutoff = returns.index[int(len(returns.index) * 0.8)]
    cutoff = pd.Timestamp(cutoff)
    truncated = build_stage_a_features(
        returns.loc[:cutoff],
        prices.loc[:cutoff],
        availability.loc[:cutoff],
        config,
    )

    comparisons = {
        "u_raw": (full_artifacts.u_raw, truncated.u_raw),
        "winsor_lower": (full_artifacts.winsor_lower, truncated.winsor_lower),
        "winsor_upper": (full_artifacts.winsor_upper, truncated.winsor_upper),
        "u_winsorized": (full_artifacts.u_winsorized, truncated.u_winsorized),
    }
    details: dict[str, Any] = {}
    failures: list[str] = []
    for name, (full, short) in comparisons.items():
        historical = full.loc[(slice(None, cutoff), slice(None)), :]
        same_axes = historical.index.equals(short.index) and historical.columns.equals(short.columns)
        same_missingness = same_axes and historical.isna().equals(short.isna())
        difference = (historical - short).abs() if same_axes else pd.DataFrame()
        max_difference = (
            float(difference.max().max()) if not difference.empty and difference.notna().any().any() else 0.0
        )
        exact_values = bool(max_difference == 0.0)
        details[name] = {
            "same_axes": bool(same_axes),
            "same_missingness": bool(same_missingness),
            "max_abs_difference": max_difference,
        }
        if not (same_axes and same_missingness and exact_values):
            failures.append(name)
    if failures:
        raise AssertionError(
            "Stage-A truncation causality check failed for: " + ", ".join(failures)
        )
    return {
        "status": "PASS",
        "cutoff": cutoff.isoformat(),
        "rule": "removing every observation after cutoff leaves history exactly unchanged",
        "comparisons": details,
    }
