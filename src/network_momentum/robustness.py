"""Graph-topology and robustness outputs for paper Section 5."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any
import os

import numpy as np
import pandas as pd

from .network_features import DynamicNetworkFeatures
from .config import PortfolioConfig
from .metrics import performance_summary
from .paper_analysis import (
    PAPER_METRIC_ORDER,
    paper_metric_values,
    turnover_and_cost_adjusted_sharpe,
)
from .portfolio import causal_portfolio_vol_rescale
from .regression import fit_pooled_ols, predict_pooled_ols


EQUITIES = {"AX", "CA", "EN", "ER", "ES", "HS", "LX", "MD", "SC", "SP", "XU", "XX", "YM"}
FIXED_INCOME = {"AP", "DT", "FB", "GS", "TU", "TY", "UB", "US"}
FX = {"AN", "BN", "CB", "CN", "DX", "FN", "JN", "MP", "NK", "SN"}


def paper_asset_classes(assets: list[str] | pd.Index) -> dict[str, str]:
    """Return the paper's four-class assignment (including CB in FX)."""
    result: dict[str, str] = {}
    for asset in map(str, assets):
        if asset in EQUITIES:
            result[asset] = "Equities"
        elif asset in FIXED_INCOME:
            result[asset] = "Fixed Income"
        elif asset in FX:
            result[asset] = "FX"
        else:
            result[asset] = "Commodities"
    return result


def _save_heatmap(graph: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    classes = paper_asset_classes(graph.index)
    order = sorted(graph.index, key=lambda asset: (classes[str(asset)], str(asset)))
    ordered = graph.loc[order, order]
    figure, axis = plt.subplots(figsize=(9.0, 8.0))
    image = axis.imshow(ordered.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(order)), order, rotation=90, fontsize=5)
    axis.set_yticks(range(len(order)), order, fontsize=5)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _save_cluster_plot(graph: pd.DataFrame, title: str, path: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from sklearn.cluster import SpectralClustering
    from sklearn.manifold import SpectralEmbedding, TSNE

    adjacency = graph.to_numpy(dtype=float)
    n = len(graph)
    clusters = SpectralClustering(
        n_clusters=4,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=7,
    ).fit_predict(adjacency)
    embedding_dimensions = min(4, n - 1)
    spectral = SpectralEmbedding(
        n_components=embedding_dimensions,
        affinity="precomputed",
        random_state=7,
    ).fit_transform(adjacency)
    coordinates = TSNE(
        n_components=2,
        perplexity=min(30.0, max(2.0, n / 4.0)),
        init="random",
        learning_rate="auto",
        random_state=7,
    ).fit_transform(spectral)
    classes = paper_asset_classes(graph.index)
    colors = {
        "Commodities": "#d97706",
        "Equities": "#2563eb",
        "Fixed Income": "#16a34a",
        "FX": "#9333ea",
    }
    markers = ["o", "s", "^", "D"]
    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    for position, asset in enumerate(graph.index.astype(str)):
        axis.scatter(
            coordinates[position, 0],
            coordinates[position, 1],
            c=colors[classes[asset]],
            marker=markers[int(clusters[position])],
            s=42,
        )
        axis.annotate(asset, coordinates[position], fontsize=6, xytext=(2, 2), textcoords="offset points")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return pd.DataFrame(
        {
            "asset": graph.index.astype(str),
            "asset_class": [classes[str(asset)] for asset in graph.index],
            "cluster": clusters,
            "tsne_x": coordinates[:, 0],
            "tsne_y": coordinates[:, 1],
        }
    ).set_index("asset")


def save_graph_topology_outputs(
    artifacts: DynamicNetworkFeatures,
    output_dir: Path,
) -> dict[str, Path]:
    """Save data and Figures 4--6 and Figure 9 from retained graphs."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(output_dir / "artifacts" / "matplotlib_cache")
    )
    import matplotlib.pyplot as plt

    if not artifacts.selected_graphs:
        raise ValueError("selected graphs were not retained")
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    graph_dir = output_dir / "artifacts" / "selected_graphs"
    for directory in (table_dir, figure_dir, graph_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for key, graph in artifacts.selected_graphs.items():
        graph.to_parquet(graph_dir / f"{key}.parquet")

    figure4_dates = ["2006-12-22", "2022-04-05"]
    for date in figure4_dates:
        key = f"{date}__ensemble"
        if key not in artifacts.selected_graphs:
            raise ValueError(f"missing selected graph {key}")
        graph = artifacts.selected_graphs[key]
        _save_heatmap(
            graph,
            f"Ensemble network {date}",
            figure_dir / f"figure_4_network_{date}.png",
        )
        clusters = _save_cluster_plot(
            graph,
            f"Spectral clusters and t-SNE {date}",
            figure_dir / f"figure_5_clusters_{date}.png",
        )
        clusters.to_csv(table_dir / f"figure_5_clusters_{date}.csv")

    lookback_date = "2020-03-23"
    for lookback in (252, 504, 756, 1008, 1260):
        key = f"{lookback_date}__lookback_{lookback}"
        if key not in artifacts.selected_graphs:
            raise ValueError(f"missing selected graph {key}")
        _save_heatmap(
            artifacts.selected_graphs[key],
            f"Network {lookback_date}, lookback={lookback}",
            figure_dir / f"figure_9_network_{lookback}.png",
        )

    topology_columns = [
        "node_count",
        "edge_sparsity",
        "average_node_degree",
        "average_clustering_coefficient",
        "community_ratio",
        "jaccard_index",
    ]
    missing = [column for column in topology_columns if column not in artifacts.diagnostics]
    if missing:
        raise ValueError(f"topology diagnostics missing columns: {missing}")
    topology = artifacts.diagnostics[topology_columns]
    topology.to_csv(table_dir / "figure_6_daily_graph_topology.csv")
    figure, axes = plt.subplots(3, 2, figsize=(13.0, 10.0), sharex=True)
    for axis, column in zip(axes.ravel(), topology_columns):
        topology[column].plot(ax=axis, linewidth=0.8)
        axis.set_title(column.replace("_", " ").title())
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "figure_6_graph_topology.png", dpi=180)
    plt.close(figure)
    return {
        "figure_4_dir": figure_dir,
        "figure_5_dir": figure_dir,
        "figure_6": figure_dir / "figure_6_graph_topology.png",
        "figure_9_dir": figure_dir,
        "topology_data": table_dir / "figure_6_daily_graph_topology.csv",
    }


@dataclass(frozen=True)
class NetworkPanelBacktest:
    raw_returns: pd.Series
    scaled_returns: pd.Series
    asset_signals: pd.DataFrame
    inference: pd.DataFrame


DEFAULT_SCHEDULES = (
    ("2000-2004", "1999-12-31", "2000-01-01", "2004-12-31"),
    ("2005-2009", "2004-12-31", "2005-01-01", "2009-12-31"),
    ("2010-2014", "2009-12-31", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2014-12-31", "2015-01-01", "2019-12-31"),
    ("2020-2022", "2019-12-31", "2020-01-01", "2022-12-31"),
)


def backtest_network_panel(
    network_panel: pd.DataFrame,
    target: pd.Series,
    returns: pd.DataFrame,
    annual_volatility: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> NetworkPanelBacktest:
    """Apply the paper's five expanding OLS blocks to one network panel."""
    available_dates = network_panel.index.get_level_values("date").unique()
    next_date_map = pd.Series(returns.index[1:].to_numpy(), index=returns.index[:-1])
    predictions: list[pd.Series] = []
    inference_rows: list[dict[str, Any]] = []
    for block, train_end_text, test_start_text, test_end_text in DEFAULT_SCHEDULES:
        train_end = pd.Timestamp(train_end_text)
        test_start = pd.Timestamp(test_start_text)
        test_end = pd.Timestamp(test_end_text)
        train_dates = pd.DatetimeIndex(
            [
                date
                for date in available_dates[available_dates <= train_end]
                if date in next_date_map.index
                and pd.Timestamp(next_date_map.loc[date]) <= train_end
            ]
        )
        test_dates = available_dates[
            (available_dates >= test_start) & (available_dates <= test_end)
        ]
        x_train = network_panel.loc[(train_dates, slice(None)), :]
        x_test = network_panel.loc[(test_dates, slice(None)), :]
        model = fit_pooled_ols(x_train, target)
        predictions.append(predict_pooled_ols(model, x_test))
        for feature, coefficient, standard_error, t_statistic, p_value in zip(
            x_train.columns,
            model.coef_,
            model.standard_errors_,
            model.t_statistics_,
            model.p_values_,
        ):
            inference_rows.append(
                {
                    "block": block,
                    "feature": feature,
                    "coefficient": coefficient,
                    "standard_error": standard_error,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                }
            )
    forecast = pd.concat(predictions).sort_index()
    direction = np.sign(forecast).where(forecast.notna())
    next_return = returns.shift(-1).stack(future_stack=True).rename("next_return")
    next_return.index = next_return.index.set_names(["date", "asset"])
    annual_vol = annual_volatility.stack(future_stack=True).rename("annual_volatility")
    annual_vol.index = annual_vol.index.set_names(["date", "asset"])
    signals = (
        direction.rename("position")
        .to_frame()
        .join(next_return, how="left")
        .join(annual_vol, how="left")
    )
    signals["asset_strategy_return"] = (
        signals["position"]
        * portfolio_config.annual_asset_vol_target
        / signals["annual_volatility"]
        * signals["next_return"]
    )
    raw = signals["asset_strategy_return"].groupby(level="date").mean().dropna()
    scaled = causal_portfolio_vol_rescale(
        raw,
        annual_target=portfolio_config.annual_asset_vol_target,
        span=portfolio_config.portfolio_vol_span,
        min_periods=portfolio_config.portfolio_vol_min_periods,
        trading_days=portfolio_config.trading_days,
        adjust=portfolio_config.portfolio_vol_ewm_adjust,
        leverage_cap=portfolio_config.portfolio_vol_leverage_cap,
    )
    return NetworkPanelBacktest(
        raw_returns=raw,
        scaled_returns=scaled,
        asset_signals=signals,
        inference=pd.DataFrame(inference_rows),
    )


def save_lookback_robustness_outputs(
    lookback_panels: dict[int, pd.DataFrame],
    ensemble_scaled_returns: pd.Series,
    target: pd.Series,
    returns: pd.DataFrame,
    annual_volatility: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """Create paper Table 3 and Figure 10 for individual lookbacks."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(output_dir / "artifacts" / "matplotlib_cache")
    )
    import matplotlib.pyplot as plt

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "artifacts" / "figure_data"
    for directory in (table_dir, figure_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    results: dict[int, NetworkPanelBacktest] = {}
    for lookback, panel in sorted(lookback_panels.items()):
        results[lookback] = backtest_network_panel(
            panel, target, returns, annual_volatility, portfolio_config
        )
    table_rows = [
        {
            "portfolio": "GMOM-Ensemble",
            **paper_metric_values(
                performance_summary(
                    ensemble_scaled_returns,
                    trading_days=portfolio_config.trading_days,
                )
            ),
        }
    ]
    for lookback, result in results.items():
        summary = performance_summary(
            result.scaled_returns, trading_days=portfolio_config.trading_days
        )
        table_rows.append(
            {
                "portfolio": f"delta_{lookback}",
                **paper_metric_values(summary),
            }
        )
    table3 = pd.DataFrame(table_rows).set_index("portfolio")
    table3.to_csv(table_dir / "table_3_lookback_performance.csv")

    names = [f"delta_{lookback}" for lookback in sorted(results)]
    raw_panel = pd.concat(
        {f"delta_{lookback}": result.raw_returns for lookback, result in results.items()},
        axis=1,
    )
    scaled_panel = pd.concat(
        {f"delta_{lookback}": result.scaled_returns for lookback, result in results.items()},
        axis=1,
    )
    signal_frames = []
    for lookback, result in results.items():
        frame = result.asset_signals.copy()
        frame["strategy"] = f"delta_{lookback}"
        signal_frames.append(frame)
        result.inference.assign(lookback=lookback).to_csv(
            table_dir / f"lookback_{lookback}_regression_inference.csv", index=False
        )
    signals = pd.concat(signal_frames)
    turnover, costs, _ = turnover_and_cost_adjusted_sharpe(
        signals,
        raw_panel,
        annual_target=portfolio_config.annual_asset_vol_target,
        trading_days=portfolio_config.trading_days,
        strategies=names,
    )
    correlation = scaled_panel.corr()
    turnover.to_csv(data_dir / "figure_10a_lookback_turnover.csv")
    costs.to_csv(data_dir / "figure_10b_lookback_cost_sharpe.csv")
    correlation.to_csv(data_dir / "figure_10c_lookback_correlation.csv")
    raw_panel.to_parquet(data_dir / "lookback_raw_returns.parquet")
    scaled_panel.to_parquet(data_dir / "lookback_scaled_returns.parquet")

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    turnover.loc[names].plot.bar(ax=axes[0], legend=False)
    axes[0].set_title("Average turnover")
    costs["cost_adjusted_sharpe"].unstack("strategy")[names].plot(
        ax=axes[1], marker="o"
    )
    axes[1].set_title("Cost-adjusted Sharpe")
    image = axes[2].imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[2].set_xticks(range(len(names)), names, rotation=45, ha="right")
    axes[2].set_yticks(range(len(names)), names)
    axes[2].set_title("Scaled return correlation")
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(figure_dir / "figure_10_lookback_robustness.png", dpi=180)
    plt.close(figure)
    return {
        "table_3": table_dir / "table_3_lookback_performance.csv",
        "figure_10": figure_dir / "figure_10_lookback_robustness.png",
    }


def save_asset_class_robustness_outputs(
    class_panels: dict[str, pd.DataFrame],
    full_gmom_signals: pd.DataFrame,
    full_gmom_scaled_returns: pd.Series,
    target: pd.Series,
    returns: pd.DataFrame,
    annual_volatility: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """Create paper Table 2 and Figures 7--8."""
    os.environ.setdefault(
        "MPLCONFIGDIR", str(output_dir / "artifacts" / "matplotlib_cache")
    )
    import matplotlib.pyplot as plt

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "artifacts" / "figure_data"
    for directory in (table_dir, figure_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    class_map = paper_asset_classes(returns.columns)
    results = {
        name: backtest_network_panel(
            panel, target, returns, annual_volatility, portfolio_config
        )
        for name, panel in class_panels.items()
    }
    scaled_series: dict[str, pd.Series] = {"GMOM": full_gmom_scaled_returns}
    for name in ("GMOM-Intra", "GMOM-Inter"):
        scaled_series[name] = results[name].scaled_returns

    gmom_signals = full_gmom_signals.loc[full_gmom_signals["strategy"] == "GMOM"]
    single_signals = results["Single-Class"].asset_signals
    for class_name, short_name in (
        ("Commodities", "Commd"),
        ("Equities", "Equity"),
        ("Fixed Income", "FI"),
        ("FX", "FX"),
    ):
        assets = {asset for asset, label in class_map.items() if label == class_name}
        cross_raw = (
            gmom_signals.loc[
                gmom_signals.index.get_level_values("asset").isin(assets),
                "asset_strategy_return",
            ]
            .groupby(level="date")
            .mean()
        )
        single_raw = (
            single_signals.loc[
                single_signals.index.get_level_values("asset").isin(assets),
                "asset_strategy_return",
            ]
            .groupby(level="date")
            .mean()
        )
        for prefix, raw in (("M", cross_raw), ("S", single_raw)):
            scaled_series[f"{prefix}-{short_name}"] = causal_portfolio_vol_rescale(
                raw,
                annual_target=portfolio_config.annual_asset_vol_target,
                span=portfolio_config.portfolio_vol_span,
                min_periods=portfolio_config.portfolio_vol_min_periods,
                trading_days=portfolio_config.trading_days,
                adjust=portfolio_config.portfolio_vol_ewm_adjust,
                leverage_cap=portfolio_config.portfolio_vol_leverage_cap,
            )
    scaled_panel = pd.concat(scaled_series, axis=1)
    rows = []
    for name, series in scaled_series.items():
        summary = performance_summary(series, trading_days=portfolio_config.trading_days)
        rows.append(
            {"portfolio": name, **paper_metric_values(summary)}
        )
    pd.DataFrame(rows).set_index("portfolio").to_csv(
        table_dir / "table_2_asset_class_performance.csv"
    )
    scaled_panel.to_parquet(data_dir / "asset_class_scaled_returns.parquet")
    correlation = scaled_panel.corr()
    correlation.to_csv(data_dir / "figure_8_asset_class_correlation.csv")

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    wealth = (1.0 + scaled_panel.fillna(0.0)).cumprod()
    wealth.plot(ax=axes[0], linewidth=0.9)
    axes[0].set_title("Asset-class robustness cumulative wealth")
    axes[0].legend(fontsize=6, ncol=2)
    image = axes[1].imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_xticks(
        range(len(correlation.columns)), correlation.columns, rotation=45, ha="right", fontsize=7
    )
    axes[1].set_yticks(range(len(correlation.index)), correlation.index, fontsize=7)
    axes[1].set_title("Scaled return correlations")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(figure_dir / "figures_7_8_asset_class_robustness.png", dpi=180)
    plt.close(figure)
    return {
        "table_2": table_dir / "table_2_asset_class_performance.csv",
        "figures_7_8": figure_dir / "figures_7_8_asset_class_robustness.png",
    }
