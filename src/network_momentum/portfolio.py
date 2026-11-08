"""Position and portfolio-return construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_to_direction(forecast: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Main GMOM direction: sign of forecast, preserving missing values."""
    direction = np.sign(forecast)
    return direction.where(forecast.notna())


def asset_strategy_returns(
    direction: pd.DataFrame,
    next_day_returns: pd.DataFrame,
    annualized_volatility: pd.DataFrame,
    annual_target: float = 0.15,
) -> pd.DataFrame:
    if not (
        direction.index.equals(next_day_returns.index)
        and direction.columns.equals(next_day_returns.columns)
        and direction.index.equals(annualized_volatility.index)
        and direction.columns.equals(annualized_volatility.columns)
    ):
        raise ValueError("direction, returns, and volatility axes must match")
    multiplier = annual_target / annualized_volatility
    return direction * multiplier * next_day_returns


def equal_weight_available(asset_returns: pd.DataFrame) -> pd.Series:
    """Average only across actually available asset strategy returns at t."""
    return asset_returns.mean(axis=1, skipna=True).rename("portfolio_return")


def causal_portfolio_vol_rescale(
    returns: pd.Series,
    annual_target: float = 0.15,
    span: int = 60,
    min_periods: int = 20,
    trading_days: int = 252,
    adjust: bool = False,
    leverage_cap: float | None = None,
) -> pd.Series:
    """Apply a second, strictly lagged portfolio-level volatility target."""
    clean = returns.astype(float)
    trailing_annual_vol = (
        clean.ewm(span=span, adjust=adjust, min_periods=min_periods).std(bias=False).shift(1)
        * np.sqrt(trading_days)
    )
    multiplier = annual_target / trailing_annual_vol
    if leverage_cap is not None:
        multiplier = multiplier.clip(upper=leverage_cap)
    return (clean * multiplier).rename(returns.name)
