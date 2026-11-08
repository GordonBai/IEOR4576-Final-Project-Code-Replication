"""Disclosed 10% validation and 11x11 graph-parameter grid search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .config import GraphConfig
from .graph_learning import (
    build_graph_lookback,
    ensemble_graphs,
    learn_graph_fast,
    normalize_graph,
    prepare_graph_feature_history,
)
from .metrics import performance_summary
from .network_features import propagate_network_features
from .regression import fit_pooled_ols, predict_pooled_ols


@dataclass(frozen=True)
class GridSearchResult:
    alpha: float
    beta: float
    validation_dates: pd.DatetimeIndex
    scores: pd.DataFrame
    selected_network_features: pd.DataFrame


def _align_warm_start(
    old_assets: list[str], old_adjacency: np.ndarray, new_assets: list[str]
) -> np.ndarray:
    old_position = {asset: position for position, asset in enumerate(old_assets)}
    aligned = np.full((len(new_assets), len(new_assets)), 1e-12, dtype=float)
    for i, asset_i in enumerate(new_assets):
        old_i = old_position.get(asset_i)
        if old_i is None:
            continue
        for j, asset_j in enumerate(new_assets):
            old_j = old_position.get(asset_j)
            if old_j is not None:
                aligned[i, j] = old_adjacency[old_i, old_j]
    np.fill_diagonal(aligned, 0.0)
    return aligned


def select_validation_dates(
    train_dates: pd.DatetimeIndex,
    config: GraphConfig,
) -> pd.DatetimeIndex:
    """Take the most recent 10% and evenly sample it for the full grid."""
    unique = pd.DatetimeIndex(train_dates).unique().sort_values()
    count = max(int(np.ceil(len(unique) * config.validation_fraction)), 1)
    validation = unique[-count:]
    if len(validation) < config.validation_min_days:
        raise ValueError(
            f"validation window has {len(validation)} dates; need {config.validation_min_days}"
        )
    if len(validation) <= config.validation_max_dates:
        return validation
    positions = np.linspace(
        0, len(validation) - 1, config.validation_max_dates, dtype=int
    )
    return validation[np.unique(positions)]


def _prepare_validation_graph_inputs(
    u_panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: GraphConfig,
    prepared_history: Any,
) -> list[dict[str, Any]]:
    """Cache pandas lookback extraction once; alpha/beta only affect solves."""
    prepared_inputs: list[dict[str, Any]] = []
    for target in dates:
        longest_input = build_graph_lookback(
            u_panel,
            target,
            max(config.lookbacks),
            calendar_policy=config.calendar_policy,
            ffill_limit=config.feature_ffill_limit,
            require_current_unfilled_features=config.require_current_unfilled_features,
            prepared_history=prepared_history,
        )
        assets = longest_input.assets
        if len(assets) < 2:
            continue
        values_by_lookback = {}
        for lookback in config.lookbacks:
            graph_input = build_graph_lookback(
                u_panel,
                target,
                lookback,
                calendar_policy=config.calendar_policy,
                ffill_limit=config.feature_ffill_limit,
                require_current_unfilled_features=config.require_current_unfilled_features,
                prepared_history=prepared_history,
                asset_subset=assets,
            )
            values_by_lookback[lookback] = graph_input.values
        prepared_inputs.append(
            {
                "date": pd.Timestamp(target),
                "assets": assets,
                "values": values_by_lookback,
                "current": u_panel.xs(target, level="date").loc[assets],
            }
        )
    return prepared_inputs


def _candidate_network_features(
    prepared_inputs: list[dict[str, Any]],
    config: GraphConfig,
    alpha: float,
    beta: float,
) -> pd.DataFrame:
    state: dict[int, tuple[list[str], np.ndarray]] = {}
    feature_rows: list[pd.DataFrame] = []
    feature_dates: list[pd.Timestamp] = []
    for prepared in prepared_inputs:
        target = prepared["date"]
        assets = prepared["assets"]
        graphs: list[np.ndarray] = []
        for lookback in config.lookbacks:
            warm = None
            if lookback in state:
                old_assets, old_adjacency = state[lookback]
                warm = _align_warm_start(old_assets, old_adjacency, assets)
            solution = learn_graph_fast(
                prepared["values"][lookback],
                alpha,
                beta,
                warm_start=warm,
                max_iterations=300,
                tolerance=1e-7,
            )
            state[lookback] = (assets, solution.adjacency)
            graphs.append(solution.adjacency)
        normalized = normalize_graph(ensemble_graphs(graphs))
        current = prepared["current"]
        propagated = propagate_network_features(normalized, current)
        if not isinstance(propagated, pd.DataFrame):
            raise AssertionError("grid-search propagation lost labels")
        feature_rows.append(propagated)
        feature_dates.append(pd.Timestamp(target))
    if not feature_rows:
        raise ValueError("candidate produced no validation network features")
    return pd.concat(feature_rows, keys=feature_dates, names=["date", "asset"]).sort_index()


def grid_search_graph_parameters(
    u_panel: pd.DataFrame,
    target: pd.Series,
    next_return: pd.Series,
    annual_volatility: pd.Series,
    train_dates: pd.DatetimeIndex,
    config: GraphConfig,
    *,
    annual_target: float = 0.15,
    trading_days: int = 252,
) -> GridSearchResult:
    """Evaluate all 121 disclosed alpha/beta pairs on the latest 10%.

    The validation score is an internally out-of-sample strategy Sharpe.  The
    paper does not disclose its score or validation-date density; both choices
    are saved with the result.
    """
    if config.validation_metric not in {"strategy_sharpe", "forecast_correlation"}:
        raise ValueError("unsupported validation metric")
    dates = select_validation_dates(train_dates, config)
    split = int(np.floor(len(dates) * config.validation_internal_train_fraction))
    split = min(max(split, 2), len(dates) - 2)
    prepared = prepare_graph_feature_history(u_panel, config.feature_ffill_limit)
    validation_inputs = _prepare_validation_graph_inputs(
        u_panel, dates, config, prepared
    )
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_pair: tuple[float, float] | None = None
    best_features: pd.DataFrame | None = None
    for alpha, beta in product(config.parameter_grid, repeat=2):
        row: dict[str, Any] = {"alpha": alpha, "beta": beta}
        try:
            features = _candidate_network_features(
                validation_inputs, config, alpha, beta
            )
            feature_dates = features.index.get_level_values("date").unique()
            train_part = feature_dates[:split]
            test_part = feature_dates[split:]
            model = fit_pooled_ols(
                features.loc[(train_part, slice(None)), :], target
            )
            forecast = predict_pooled_ols(
                model, features.loc[(test_part, slice(None)), :]
            )
            frame = (
                forecast.rename("forecast")
                .to_frame()
                .join(target.rename("target"), how="left")
                .join(next_return.rename("next_return"), how="left")
                .join(annual_volatility.rename("annual_volatility"), how="left")
            ).dropna()
            frame["asset_return"] = (
                np.sign(frame["forecast"])
                * annual_target
                / frame["annual_volatility"]
                * frame["next_return"]
            )
            portfolio = frame["asset_return"].groupby(level="date").mean()
            summary = performance_summary(portfolio, trading_days=trading_days)
            forecast_correlation = float(frame[["forecast", "target"]].corr().iloc[0, 1])
            score = (
                forecast_correlation
                if config.validation_metric == "forecast_correlation"
                else float(summary["sharpe"])
            )
            row.update(
                {
                    "status": "PASS",
                    "score": score,
                    "validation_portfolio_days": int(len(portfolio)),
                    "validation_forecasts": int(len(frame)),
                    "forecast_correlation": forecast_correlation,
                    "strategy_sharpe": float(summary["sharpe"]),
                }
            )
            if np.isfinite(score) and (
                best_pair is None
                or score > best_score
                or (score == best_score and (alpha, beta) < best_pair)
            ):
                best_score = score
                best_pair = (float(alpha), float(beta))
                best_features = features
        except Exception as error:
            row.update({"status": "FAIL", "score": -np.inf, "error": str(error)})
        rows.append(row)
    if best_pair is None or best_features is None:
        raise RuntimeError("all graph hyperparameter candidates failed")
    scores = pd.DataFrame(rows).sort_values(
        ["score", "alpha", "beta"], ascending=[False, True, True]
    )
    return GridSearchResult(
        alpha=best_pair[0],
        beta=best_pair[1],
        validation_dates=dates,
        scores=scores,
        selected_network_features=best_features,
    )
