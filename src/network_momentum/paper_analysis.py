"""Paper-facing tables and figures for Sections 4.2--4.4 and Table 4."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import os

import numpy as np
import pandas as pd

from .metrics import performance_summary


STRATEGY_ORDER = ["Long Only", "MACD", "LinReg", "GMOM", "RegCombo", "SignCombo"]
PAPER_METRIC_ORDER = [
    "annualized_mean_return",
    "annualized_volatility",
    "sharpe",
    "downside_deviation",
    "maximum_drawdown",
    "maximum_drawdown_duration_fraction",
    "sortino",
    "calmar",
    "hit_rate",
    "average_profit_loss_ratio",
]


def paper_metric_values(summary: dict[str, float]) -> dict[str, float]:
    """Format internal metrics using the paper's positive MDD convention."""
    values = {metric: summary[metric] for metric in PAPER_METRIC_ORDER}
    values["maximum_drawdown"] = abs(values["maximum_drawdown"])
    return values


def build_complete_metrics_table(
    raw_returns: pd.DataFrame,
    scaled_returns: pd.DataFrame,
    *,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Return paper Table 1 in long form with the paper's ten metrics."""
    rows: list[dict[str, float | str]] = []
    for panel_name, returns in (("raw", raw_returns), ("scaled_15pct", scaled_returns)):
        for strategy in STRATEGY_ORDER:
            if strategy not in returns:
                raise ValueError(f"missing Table-1 strategy: {strategy}")
            summary = performance_summary(returns[strategy], trading_days=trading_days)
            rows.append(
                {
                    "panel": panel_name,
                    "strategy": strategy,
                    **paper_metric_values(summary),
                }
            )
    return pd.DataFrame(rows).set_index(["panel", "strategy"])


def return_correlations(
    raw_returns: pd.DataFrame,
    scaled_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = [name for name in STRATEGY_ORDER if name in raw_returns.columns]
    return raw_returns[order].corr(), scaled_returns[order].corr()


def sign_agreement(asset_signals: pd.DataFrame) -> pd.DataFrame:
    """Asset-date sign agreement used in paper Figure 2(c)."""
    required = {"strategy", "position"}
    if not required.issubset(asset_signals.columns):
        raise ValueError(f"asset_signals requires {sorted(required)}")
    wide = (
        asset_signals.reset_index()
        .pivot_table(index=["date", "asset"], columns="strategy", values="position", aggfunc="last")
    )
    order = [name for name in STRATEGY_ORDER if name in wide.columns]
    result = pd.DataFrame(np.nan, index=order, columns=order, dtype=float)
    for left in order:
        for right in order:
            left_signal = wide[left].rename("left")
            right_signal = wide[right].rename("right")
            pair = pd.concat([left_signal, right_signal], axis=1).dropna()
            result.loc[left, right] = float(
                (np.sign(pair["left"]) == np.sign(pair["right"])).mean()
            ) if len(pair) else np.nan
    return result


def turnover_and_cost_adjusted_sharpe(
    asset_signals: pd.DataFrame,
    raw_returns: pd.DataFrame,
    *,
    annual_target: float = 0.15,
    costs_bps: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0),
    trading_days: int = 252,
    strategies: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Implement paper equations (13)--(14).

    ``t-1`` is the preceding valid trading observation for each asset.  This
    avoids treating a local-market holiday as a liquidation and re-entry, in
    line with the project's asset-observation calendar convention.
    """
    required = {"strategy", "position", "annual_volatility"}
    if not required.issubset(asset_signals.columns):
        raise ValueError(f"asset_signals requires {sorted(required)}")
    daily_turnover: dict[str, pd.Series] = {}
    cost_returns: dict[tuple[str, float], pd.Series] = {}
    summary_rows: list[dict[str, float | str]] = []
    strategy_order = list(strategies) if strategies is not None else STRATEGY_ORDER
    for strategy in strategy_order:
        subset = asset_signals.loc[asset_signals["strategy"] == strategy]
        if subset.empty:
            raise ValueError(f"missing signals for {strategy}")
        exposure = (
            annual_target * subset["position"] / subset["annual_volatility"]
        ).replace([np.inf, -np.inf], np.nan).dropna()
        previous = exposure.groupby(level="asset").shift(1)
        asset_turnover = (exposure - previous).abs()
        turnover = asset_turnover.groupby(level="date").mean().reindex(raw_returns.index)
        turnover = turnover.rename(strategy)
        daily_turnover[strategy] = turnover
        summary_rows.append(
            {"strategy": strategy, "average_turnover": float(turnover.mean())}
        )
        for cost in costs_bps:
            adjusted = raw_returns[strategy] - float(cost) * 1e-4 * turnover
            adjusted.name = strategy
            cost_returns[(strategy, float(cost))] = adjusted
    turnover_panel = pd.DataFrame(daily_turnover)
    cost_return_panel = pd.concat(cost_returns, axis=1)
    sharpe_rows: list[dict[str, float | str]] = []
    for strategy in strategy_order:
        for cost in costs_bps:
            series = cost_return_panel[(strategy, float(cost))]
            metric = performance_summary(series, trading_days=trading_days)
            sharpe_rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": float(cost),
                    "cost_adjusted_sharpe": metric["sharpe"],
                }
            )
    return (
        pd.DataFrame(summary_rows).set_index("strategy"),
        pd.DataFrame(sharpe_rows).set_index(["strategy", "cost_bps"]),
        cost_return_panel,
    )


def _plot_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_main_paper_outputs(
    raw_returns: pd.DataFrame,
    scaled_returns: pd.DataFrame,
    asset_signals: pd.DataFrame,
    output_dir: Path,
    *,
    trading_days: int = 252,
) -> dict[str, Path]:
    """Save complete Table 1 and Figures 1--3 data/plots."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(output_dir / "artifacts" / "matplotlib_cache")
    )
    import matplotlib.pyplot as plt

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "artifacts" / "figure_data"
    for directory in (table_dir, figure_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    table1 = build_complete_metrics_table(
        raw_returns, scaled_returns, trading_days=trading_days
    )
    raw_corr, scaled_corr = return_correlations(raw_returns, scaled_returns)
    agreement = sign_agreement(asset_signals)
    turnover, cost_sharpe, cost_returns = turnover_and_cost_adjusted_sharpe(
        asset_signals,
        raw_returns,
        trading_days=trading_days,
    )
    table1.to_csv(table_dir / "table_1_portfolio_performance.csv")
    raw_corr.to_csv(data_dir / "figure_2a_raw_return_correlation.csv")
    scaled_corr.to_csv(data_dir / "figure_2b_scaled_return_correlation.csv")
    agreement.to_csv(data_dir / "figure_2c_sign_agreement.csv")
    turnover.to_csv(data_dir / "figure_3a_average_turnover.csv")
    cost_sharpe.to_csv(data_dir / "figure_3b_cost_adjusted_sharpe.csv")
    cost_returns.to_parquet(data_dir / "figure_3_cost_adjusted_returns.parquet")

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for axis, panel, title in (
        (axes[0], raw_returns, "Raw signals"),
        (axes[1], scaled_returns, "Signals scaled to 15% volatility"),
    ):
        wealth = (1.0 + panel[STRATEGY_ORDER].fillna(0.0)).cumprod()
        wealth.plot(ax=axis, linewidth=1.1)
        axis.set_title(title)
        axis.set_ylabel("Cumulative wealth")
        axis.grid(alpha=0.2)
    axes[1].legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "figure_1_cumulative_returns.png", dpi=180)
    plt.close(figure)

    _plot_heatmap(raw_corr, "Raw return correlations", figure_dir / "figure_2a_raw_correlation.png")
    _plot_heatmap(
        scaled_corr,
        "15%-scaled return correlations",
        figure_dir / "figure_2b_scaled_correlation.png",
    )
    _plot_heatmap(agreement, "Asset-date sign agreement", figure_dir / "figure_2c_sign_agreement.png")

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    turnover.loc[STRATEGY_ORDER].plot.bar(ax=axes[0], legend=False)
    axes[0].set_title("Average turnover")
    axes[0].set_ylabel("Average daily turnover")
    cost_plot = cost_sharpe["cost_adjusted_sharpe"].unstack("strategy")
    cost_plot[STRATEGY_ORDER].plot(ax=axes[1], marker="o")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_title("Cost-adjusted Sharpe")
    axes[1].set_xlabel("Cost (bps)")
    axes[1].set_ylabel("Sharpe")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "figure_3_turnover_and_costs.png", dpi=180)
    plt.close(figure)

    return {
        "table_1": table_dir / "table_1_portfolio_performance.csv",
        "figure_1": figure_dir / "figure_1_cumulative_returns.png",
        "figure_2a": figure_dir / "figure_2a_raw_correlation.png",
        "figure_2b": figure_dir / "figure_2b_scaled_correlation.png",
        "figure_2c": figure_dir / "figure_2c_sign_agreement.png",
        "figure_3": figure_dir / "figure_3_turnover_and_costs.png",
    }


def save_table4_regression_inference(
    inference: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save numeric and paper-formatted LinReg/GMOM Table 4 outputs."""
    required = {
        "block",
        "model",
        "feature",
        "coefficient",
        "standard_error",
        "t_statistic",
        "p_value",
    }
    if not required.issubset(inference.columns):
        raise ValueError(f"regression inference requires {sorted(required)}")
    selected = inference.loc[
        inference["model"].isin(["LinReg", "GMOM"])
        & inference["feature"].ne("intercept")
    ].copy()
    selected["significant_5pct"] = selected["p_value"] < 0.05
    selected["paper_value"] = selected.apply(
        lambda row: (
            f"{row['coefficient']:.3f}"
            f"{'*' if row['significant_5pct'] else ''} "
            f"({row['standard_error']:.3f})"
        ),
        axis=1,
    )
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    numeric_path = table_dir / "table_4_regression_inference_long.csv"
    formatted_path = table_dir / "table_4_regression_coefficients_formatted.csv"
    selected.to_csv(numeric_path, index=False)
    selected.pivot(index=["model", "block"], columns="feature", values="paper_value").to_csv(
        formatted_path
    )
    return numeric_path, formatted_path
