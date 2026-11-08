"""Performance metrics with explicit daily arithmetic-return conventions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    clean = returns.dropna().astype(float)
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    underwater = drawdown.lt(0)
    groups = underwater.ne(underwater.shift()).cumsum()
    duration = int(underwater.groupby(groups).sum().max()) if not underwater.empty else 0
    return max_dd, duration


def performance_summary(returns: pd.Series, trading_days: int = 252) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("no returns available")
    annual_return = float(clean.mean() * trading_days)
    annual_vol = float(clean.std(ddof=1) * np.sqrt(trading_days))
    downside = clean.clip(upper=0)
    downside_deviation = float(np.sqrt((downside.pow(2)).mean()) * np.sqrt(trading_days))
    max_dd, max_dd_duration = max_drawdown(clean)
    profits = clean[clean > 0]
    losses = clean[clean < 0]
    average_profit = float(profits.mean()) if not profits.empty else np.nan
    average_loss = float(losses.mean()) if not losses.empty else np.nan
    return {
        "annualized_mean_return": annual_return,
        "annualized_volatility": annual_vol,
        "sharpe": annual_return / annual_vol if annual_vol else np.nan,
        "downside_deviation": downside_deviation,
        "sortino": annual_return / downside_deviation if downside_deviation else np.nan,
        "maximum_drawdown": max_dd,
        "maximum_drawdown_duration_days": float(max_dd_duration),
        # Pu et al. define MDD duration as the longest underwater spell as a
        # percentage of the full out-of-sample period.
        "maximum_drawdown_duration_fraction": float(max_dd_duration / len(clean)),
        "calmar": annual_return / abs(max_dd) if max_dd else np.nan,
        "hit_rate": float((clean > 0).mean()),
        "average_profit": average_profit,
        "average_loss": average_loss,
        "average_profit_loss_ratio": (
            average_profit / abs(average_loss)
            if np.isfinite(average_profit) and np.isfinite(average_loss) and average_loss != 0
            else np.nan
        ),
    }
