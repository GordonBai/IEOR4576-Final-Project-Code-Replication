"""Network-momentum feature propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .config import GraphConfig
from .graph_learning import (
    build_graph_lookback,
    ensemble_graphs,
    learn_graph_cvxpy_edges_from_distances,
    learn_graph_cvxpy_edges,
    learn_graph_fast_from_distances,
    learn_graph_fast,
    normalize_graph,
    prepare_graph_feature_history,
    validate_adjacency,
)


@dataclass(frozen=True)
class DynamicNetworkFeatures:
    panel: pd.DataFrame
    diagnostics: pd.DataFrame
    lookback_panels: dict[int, pd.DataFrame] | None = None
    selected_graphs: dict[str, pd.DataFrame] | None = None
    class_ablation_panels: dict[str, pd.DataFrame] | None = None


def _graph_topology_metrics(
    adjacency: np.ndarray,
    assets: list[str],
    edge_tolerance: float,
    asset_classes: dict[str, str] | None,
    previous_edges: set[tuple[str, str]] | None,
) -> tuple[dict[str, float], set[tuple[str, str]]]:
    binary = np.asarray(adjacency, dtype=float) > edge_tolerance
    np.fill_diagonal(binary, False)
    n = len(assets)
    rows, columns = np.triu_indices(n, k=1)
    edge_mask = binary[rows, columns]
    edges = {
        (assets[int(i)], assets[int(j)])
        for i, j, present in zip(rows, columns, edge_mask)
        if present
    }
    degree = binary.sum(axis=1)
    local_cluster: list[float] = []
    for node in range(n):
        neighbours = np.flatnonzero(binary[node])
        if len(neighbours) < 2:
            local_cluster.append(0.0)
            continue
        subgraph = binary[np.ix_(neighbours, neighbours)]
        triangles = float(subgraph.sum() / 2.0)
        local_cluster.append(2.0 * triangles / (len(neighbours) * (len(neighbours) - 1)))
    community_ratio = np.nan
    if asset_classes is not None and n > 1:
        agreements = 0
        comparisons = 0
        for i, j in zip(rows, columns):
            if assets[int(i)] not in asset_classes or assets[int(j)] not in asset_classes:
                continue
            same_class = asset_classes[assets[int(i)]] == asset_classes[assets[int(j)]]
            agreements += int((binary[int(i), int(j)] and same_class) or (not binary[int(i), int(j)] and not same_class))
            comparisons += 1
        community_ratio = agreements / comparisons if comparisons else np.nan
    jaccard = np.nan
    if previous_edges is not None:
        union = edges | previous_edges
        jaccard = len(edges & previous_edges) / len(union) if union else 1.0
    possible_edges = n * (n - 1) / 2.0
    return (
        {
            "node_count": float(n),
            "edge_sparsity": len(edges) / possible_edges if possible_edges else np.nan,
            "average_node_degree": float(degree.mean()) if n else np.nan,
            "average_clustering_coefficient": float(np.mean(local_cluster)) if n else np.nan,
            "community_ratio": float(community_ratio),
            "jaccard_index": float(jaccard),
        },
        edges,
    )


def _solve_cvx_with_fallback(
    values: np.ndarray,
    config: GraphConfig,
) -> tuple[Any, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return (
                learn_graph_cvxpy_edges(
                    values,
                    config.prototype_alpha,
                    config.prototype_beta,
                    solver=config.prototype_solver,
                    solver_options={"max_iter": 1000},
                ),
                config.prototype_solver,
            )
        except Exception:
            return (
                learn_graph_cvxpy_edges(
                    values,
                    config.prototype_alpha,
                    config.prototype_beta,
                    solver="SCS",
                    solver_options={"max_iters": 20000, "eps": 1e-5},
                ),
                "SCS",
            )


def _solve_distance_cvx_with_fallback(
    distances: np.ndarray,
    config: GraphConfig,
) -> tuple[Any, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            return (
                learn_graph_cvxpy_edges_from_distances(
                    distances,
                    config.prototype_alpha,
                    config.prototype_beta,
                    solver=config.prototype_solver,
                    solver_options={"max_iter": 1000},
                ),
                config.prototype_solver,
            )
        except Exception:
            return (
                learn_graph_cvxpy_edges_from_distances(
                    distances,
                    config.prototype_alpha,
                    config.prototype_beta,
                    solver="SCS",
                    solver_options={"max_iters": 20000, "eps": 1e-5},
                ),
                "SCS",
            )


def _align_warm_start(
    old_assets: list[str],
    old_adjacency: np.ndarray,
    new_assets: list[str],
) -> np.ndarray:
    aligned = np.full((len(new_assets), len(new_assets)), 1e-12, dtype=float)
    old_position = {asset: i for i, asset in enumerate(old_assets)}
    for new_i, asset_i in enumerate(new_assets):
        old_i = old_position.get(asset_i)
        if old_i is None:
            continue
        for new_j, asset_j in enumerate(new_assets):
            old_j = old_position.get(asset_j)
            if old_j is not None:
                aligned[new_i, new_j] = old_adjacency[old_i, old_j]
    np.fill_diagonal(aligned, 0.0)
    return aligned


def propagate_network_features(
    normalized_adjacency: np.ndarray,
    u_t: pd.DataFrame | np.ndarray,
) -> pd.DataFrame | np.ndarray:
    """Return U_tilde = A_tilde @ U_t with dimension checks."""
    a = np.asarray(normalized_adjacency, dtype=float)
    values = u_t.to_numpy(dtype=float) if isinstance(u_t, pd.DataFrame) else np.asarray(u_t, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("normalized_adjacency must be square")
    if values.ndim != 2 or values.shape[0] != a.shape[1]:
        raise ValueError(f"shape mismatch: adjacency {a.shape}, features {values.shape}")
    propagated = a @ values
    if isinstance(u_t, pd.DataFrame):
        return pd.DataFrame(propagated, index=u_t.index, columns=u_t.columns)
    return propagated


def build_dynamic_network_features(
    u_panel: pd.DataFrame,
    dates: pd.DatetimeIndex | list[pd.Timestamp],
    config: GraphConfig,
) -> DynamicNetworkFeatures:
    """Construct dynamic five-graph ensemble features for selected dates."""
    requested = pd.DatetimeIndex(dates)
    available = u_panel.index.get_level_values("date").unique()
    requested = requested.intersection(available).sort_values()
    if len(requested) == 0:
        raise ValueError("no requested dates exist in u_panel")
    prepared = prepare_graph_feature_history(u_panel, config.feature_ffill_limit)
    longest = max(config.lookbacks)
    feature_rows: list[pd.DataFrame] = []
    feature_dates: list[pd.Timestamp] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for target in requested:
        longest_input = build_graph_lookback(
            u_panel,
            target,
            longest,
            calendar_policy=config.calendar_policy,
            ffill_limit=config.feature_ffill_limit,
            require_current_unfilled_features=config.require_current_unfilled_features,
            prepared_history=prepared,
        )
        common_assets = longest_input.assets
        if len(common_assets) < 2:
            diagnostic_rows.append(
                {
                    "date": pd.Timestamp(target),
                    "asset_count": len(common_assets),
                    "all_solver_statuses_acceptable": False,
                    "solver_statuses": "SKIPPED_FEWER_THAN_TWO_ASSETS",
                    "solvers_used": "NONE",
                    "fallback_solver_count": 0,
                    "minimum_edges": 0,
                    "maximum_edges": 0,
                    "causal_short_gap_fills_across_lookbacks": 0,
                }
            )
            continue
        graphs: list[np.ndarray] = []
        statuses: list[str] = []
        solvers_used: list[str] = []
        edge_counts: list[int] = []
        total_fills = 0
        for lookback in config.lookbacks:
            graph_input = build_graph_lookback(
                u_panel,
                target,
                lookback,
                calendar_policy=config.calendar_policy,
                ffill_limit=config.feature_ffill_limit,
                require_current_unfilled_features=config.require_current_unfilled_features,
                prepared_history=prepared,
                asset_subset=common_assets,
            )
            solution, solver_used = _solve_cvx_with_fallback(graph_input.values, config)
            validate_adjacency(solution.adjacency, tolerance=1e-8)
            graphs.append(solution.adjacency)
            statuses.append(solution.status)
            solvers_used.append(solver_used)
            upper = solution.adjacency[np.triu_indices_from(solution.adjacency, k=1)]
            edge_counts.append(int((upper > config.edge_tolerance).sum()))
            total_fills += graph_input.fill_count

        ensemble = ensemble_graphs(graphs)
        normalized = normalize_graph(ensemble)
        current = u_panel.xs(target, level="date").loc[common_assets]
        propagated = propagate_network_features(normalized, current)
        if not isinstance(propagated, pd.DataFrame):
            raise AssertionError("propagated features lost labels")
        propagated.index.name = "asset"
        feature_rows.append(propagated)
        feature_dates.append(pd.Timestamp(target))
        diagnostic_rows.append(
            {
                "date": pd.Timestamp(target),
                "asset_count": len(common_assets),
                "all_solver_statuses_acceptable": all(
                    status in {"optimal", "optimal_inaccurate"} for status in statuses
                ),
                "solver_statuses": "|".join(statuses),
                "solvers_used": "|".join(solvers_used),
                "fallback_solver_count": int(sum(solver != config.prototype_solver for solver in solvers_used)),
                "minimum_edges": min(edge_counts),
                "maximum_edges": max(edge_counts),
                "causal_short_gap_fills_across_lookbacks": total_fills,
            }
        )

    if not feature_rows:
        raise ValueError("no dates produced a graph with at least two assets")
    panel = pd.concat(feature_rows, keys=feature_dates, names=["date", "asset"])
    panel = panel.sort_index()
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("date").sort_index()
    return DynamicNetworkFeatures(panel=panel, diagnostics=diagnostics)


def build_dynamic_network_features_production(
    u_panel: pd.DataFrame,
    dates: pd.DatetimeIndex | list[pd.Timestamp],
    config: GraphConfig,
    *,
    reference_interval: int = 60,
) -> DynamicNetworkFeatures:
    """Historical graph path using exact reference solves plus warm starts.

    Every ``reference_interval`` dates and whenever a fast solve fails, the
    state is reset with CVXPY/CLARABEL (or recorded SCS fallback).
    """
    requested = pd.DatetimeIndex(dates)
    available = u_panel.index.get_level_values("date").unique()
    requested = requested.intersection(available).sort_values()
    if len(requested) == 0:
        raise ValueError("no requested dates exist in u_panel")
    prepared = prepare_graph_feature_history(u_panel, config.feature_ffill_limit)
    longest = max(config.lookbacks)
    state: dict[int, tuple[list[str], np.ndarray]] = {}
    feature_rows: list[pd.DataFrame] = []
    feature_dates: list[pd.Timestamp] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for date_number, target in enumerate(requested):
        longest_input = build_graph_lookback(
            u_panel,
            target,
            longest,
            calendar_policy=config.calendar_policy,
            ffill_limit=config.feature_ffill_limit,
            require_current_unfilled_features=config.require_current_unfilled_features,
            prepared_history=prepared,
        )
        assets = longest_input.assets
        if len(assets) < 2:
            diagnostic_rows.append(
                {"date": target, "asset_count": len(assets), "status": "SKIPPED", "cvx_solves": 0}
            )
            continue
        graphs: list[np.ndarray] = []
        cvx_solves = 0
        fast_solves = 0
        scs_fallbacks = 0
        for lookback in config.lookbacks:
            graph_input = build_graph_lookback(
                u_panel,
                target,
                lookback,
                calendar_policy=config.calendar_policy,
                ffill_limit=config.feature_ffill_limit,
                require_current_unfilled_features=config.require_current_unfilled_features,
                prepared_history=prepared,
                asset_subset=assets,
            )
            use_reference = lookback not in state or date_number % reference_interval == 0
            solution = None
            if not use_reference:
                old_assets, old_adjacency = state[lookback]
                warm = _align_warm_start(old_assets, old_adjacency, assets)
                fast = learn_graph_fast(
                    graph_input.values,
                    config.prototype_alpha,
                    config.prototype_beta,
                    warm_start=warm,
                    max_iterations=3000,
                    tolerance=1e-10,
                )
                if fast.success:
                    solution = fast
                    fast_solves += 1
            if solution is None:
                solution, solver_used = _solve_cvx_with_fallback(graph_input.values, config)
                cvx_solves += 1
                scs_fallbacks += int(solver_used == "SCS")
            validate_adjacency(solution.adjacency, tolerance=1e-8)
            state[lookback] = (assets, solution.adjacency)
            graphs.append(solution.adjacency)

        normalized = normalize_graph(ensemble_graphs(graphs))
        current = u_panel.xs(target, level="date").loc[assets]
        propagated = propagate_network_features(normalized, current)
        if not isinstance(propagated, pd.DataFrame):
            raise AssertionError("propagated features lost labels")
        propagated.index.name = "asset"
        feature_rows.append(propagated)
        feature_dates.append(pd.Timestamp(target))
        diagnostic_rows.append(
            {
                "date": target,
                "asset_count": len(assets),
                "status": "PASS",
                "cvx_solves": cvx_solves,
                "fast_solves": fast_solves,
                "scs_fallbacks": scs_fallbacks,
            }
        )

    if not feature_rows:
        raise ValueError("no dates produced production network features")
    panel = pd.concat(feature_rows, keys=feature_dates, names=["date", "asset"]).sort_index()
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("date").sort_index()
    return DynamicNetworkFeatures(panel=panel, diagnostics=diagnostics)


def build_dynamic_network_features_rolling(
    u_panel: pd.DataFrame,
    dates: pd.DatetimeIndex | list[pd.Timestamp],
    config: GraphConfig,
    *,
    reference_interval: int = 60,
    retain_lookback_features: bool = False,
    selected_graph_dates: set[pd.Timestamp] | None = None,
    asset_classes: dict[str, str] | None = None,
    retain_class_ablation_features: bool = False,
) -> DynamicNetworkFeatures:
    """Production path with rolling pairwise distances and warm graph solves."""
    requested = pd.DatetimeIndex(dates)
    date_index = u_panel.index.get_level_values("date").unique()
    requested = requested.intersection(date_index).sort_values()
    if len(requested) == 0:
        raise ValueError("no requested dates exist in u_panel")
    original, filled = prepare_graph_feature_history(u_panel, config.feature_ffill_limit)
    features = list(u_panel.columns)
    asset_index = original[features[0]].columns.astype(str)
    original_values = np.stack(
        [original[feature].to_numpy(dtype=float) for feature in features], axis=2
    )
    filled_values = np.stack(
        [filled[feature].to_numpy(dtype=float) for feature in features], axis=2
    )
    original_valid = np.isfinite(original_values).all(axis=2)
    filled_valid = np.isfinite(filled_values).all(axis=2)
    filled_cumulative = np.vstack(
        [np.zeros((1, len(asset_index)), dtype=np.int32), np.cumsum(filled_valid, axis=0)]
    )
    position = pd.Series(np.arange(len(date_index)), index=date_index)
    requested_positions = {int(position.loc[date]): pd.Timestamp(date) for date in requested}
    longest = max(config.lookbacks)
    start_position = max(0, min(requested_positions) - longest + 1)
    end_position = max(requested_positions)
    buffer_size = longest + 1
    distance_buffer = np.zeros(
        (buffer_size, len(asset_index), len(asset_index)), dtype=np.float64
    )
    rolling_sums = {
        lookback: np.zeros((len(asset_index), len(asset_index)), dtype=np.float64)
        for lookback in config.lookbacks
    }
    state: dict[int, tuple[list[str], np.ndarray]] = {}
    feature_rows: list[pd.DataFrame] = []
    feature_dates: list[pd.Timestamp] = []
    diagnostic_rows: list[dict[str, Any]] = []
    lookback_rows: dict[int, list[pd.DataFrame]] = {
        lookback: [] for lookback in config.lookbacks
    }
    lookback_dates: dict[int, list[pd.Timestamp]] = {
        lookback: [] for lookback in config.lookbacks
    }
    selected_graphs: dict[str, pd.DataFrame] = {}
    previous_edges: set[tuple[str, str]] | None = None
    class_rows: dict[str, list[pd.DataFrame]] = {
        "GMOM-Intra": [],
        "GMOM-Inter": [],
        "Single-Class": [],
    }
    class_dates: dict[str, list[pd.Timestamp]] = {key: [] for key in class_rows}
    class_state: dict[tuple[str, int], tuple[list[str], np.ndarray]] = {}
    if retain_class_ablation_features and asset_classes is None:
        raise ValueError("asset_classes are required for class ablations")
    solved_date_number = 0

    for local_step, global_position in enumerate(range(start_position, end_position + 1)):
        values = filled_values[global_position]
        valid = filled_valid[global_position]
        safe = np.where(np.isfinite(values), values, 0.0)
        norms = np.sum(safe * safe, axis=1)
        daily_distance = np.maximum(
            norms[:, None] + norms[None, :] - 2.0 * (safe @ safe.T), 0.0
        )
        pair_valid = valid[:, None] & valid[None, :]
        daily_distance = np.where(pair_valid, daily_distance, 0.0)
        for lookback in config.lookbacks:
            rolling_sums[lookback] += daily_distance
            if local_step >= lookback:
                rolling_sums[lookback] -= distance_buffer[(local_step - lookback) % buffer_size]
        distance_buffer[local_step % buffer_size] = daily_distance

        if global_position not in requested_positions or local_step + 1 < longest:
            continue
        target = requested_positions[global_position]
        window_start = global_position + 1 - longest
        valid_counts = filled_cumulative[global_position + 1] - filled_cumulative[window_start]
        asset_mask = (valid_counts == longest) & original_valid[global_position]
        asset_positions = np.flatnonzero(asset_mask)
        assets = asset_index[asset_positions].tolist()
        if len(assets) < 2:
            diagnostic_rows.append(
                {"date": target, "asset_count": len(assets), "status": "SKIPPED", "cvx_solves": 0}
            )
            continue

        graphs: list[np.ndarray] = []
        cvx_solves = 0
        fast_solves = 0
        scs_fallbacks = 0
        for lookback in config.lookbacks:
            distances = rolling_sums[lookback][np.ix_(asset_positions, asset_positions)]
            use_reference = lookback not in state or solved_date_number % reference_interval == 0
            solution = None
            if not use_reference:
                old_assets, old_adjacency = state[lookback]
                warm = _align_warm_start(old_assets, old_adjacency, assets)
                fast = learn_graph_fast_from_distances(
                    distances,
                    config.prototype_alpha,
                    config.prototype_beta,
                    warm_start=warm,
                    max_iterations=3000,
                    tolerance=1e-10,
                )
                if fast.success:
                    solution = fast
                    fast_solves += 1
            if solution is None:
                solution, solver_used = _solve_distance_cvx_with_fallback(distances, config)
                cvx_solves += 1
                scs_fallbacks += int(solver_used == "SCS")
            validate_adjacency(solution.adjacency, tolerance=1e-8)
            state[lookback] = (assets, solution.adjacency)
            graphs.append(solution.adjacency)

        ensemble = ensemble_graphs(graphs)
        normalized = normalize_graph(ensemble)
        current = pd.DataFrame(
            original_values[global_position, asset_positions, :],
            index=pd.Index(assets, name="asset"),
            columns=features,
        )
        propagated = propagate_network_features(normalized, current)
        if not isinstance(propagated, pd.DataFrame):
            raise AssertionError("propagated features lost labels")
        feature_rows.append(propagated)
        feature_dates.append(target)
        if retain_lookback_features:
            for lookback, graph in zip(config.lookbacks, graphs):
                lookback_propagated = propagate_network_features(
                    normalize_graph(graph), current
                )
                if not isinstance(lookback_propagated, pd.DataFrame):
                    raise AssertionError("lookback propagation lost labels")
                lookback_rows[lookback].append(lookback_propagated)
                lookback_dates[lookback].append(target)
        if retain_class_ablation_features:
            labels = [asset_classes[asset] for asset in assets]
            same_class = np.equal.outer(labels, labels)
            intra_graph = np.where(same_class, ensemble, 0.0)
            inter_graph = np.where(~same_class, ensemble, 0.0)
            np.fill_diagonal(intra_graph, 0.0)
            np.fill_diagonal(inter_graph, 0.0)
            for name, graph in (
                ("GMOM-Intra", intra_graph),
                ("GMOM-Inter", inter_graph),
            ):
                class_propagated = propagate_network_features(
                    normalize_graph(graph), current
                )
                if not isinstance(class_propagated, pd.DataFrame):
                    raise AssertionError("class-ablation propagation lost labels")
                class_rows[name].append(class_propagated)
                class_dates[name].append(target)

            single_class_parts: list[pd.DataFrame] = []
            for class_name in sorted(set(labels)):
                local_indices = [
                    position for position, label in enumerate(labels) if label == class_name
                ]
                if len(local_indices) < 2:
                    continue
                global_class_positions = asset_positions[np.asarray(local_indices, dtype=int)]
                class_assets = [assets[position] for position in local_indices]
                class_graphs: list[np.ndarray] = []
                for lookback in config.lookbacks:
                    distances = rolling_sums[lookback][
                        np.ix_(global_class_positions, global_class_positions)
                    ]
                    state_key = (class_name, lookback)
                    use_reference = (
                        state_key not in class_state
                        or solved_date_number % reference_interval == 0
                    )
                    solution = None
                    if not use_reference:
                        old_assets, old_adjacency = class_state[state_key]
                        warm = _align_warm_start(
                            old_assets, old_adjacency, class_assets
                        )
                        fast = learn_graph_fast_from_distances(
                            distances,
                            config.prototype_alpha,
                            config.prototype_beta,
                            warm_start=warm,
                            max_iterations=3000,
                            tolerance=1e-10,
                        )
                        if fast.success:
                            solution = fast
                    if solution is None:
                        solution, _ = _solve_distance_cvx_with_fallback(
                            distances, config
                        )
                    class_state[state_key] = (class_assets, solution.adjacency)
                    class_graphs.append(solution.adjacency)
                class_current = current.loc[class_assets]
                class_propagated = propagate_network_features(
                    normalize_graph(ensemble_graphs(class_graphs)), class_current
                )
                if not isinstance(class_propagated, pd.DataFrame):
                    raise AssertionError("single-class propagation lost labels")
                single_class_parts.append(class_propagated)
            if single_class_parts:
                class_rows["Single-Class"].append(
                    pd.concat(single_class_parts).sort_index()
                )
                class_dates["Single-Class"].append(target)
        if selected_graph_dates and pd.Timestamp(target) in selected_graph_dates:
            index = pd.Index(assets, name="asset")
            selected_graphs[f"{pd.Timestamp(target).date()}__ensemble"] = pd.DataFrame(
                ensemble, index=index, columns=index
            )
            for lookback, graph in zip(config.lookbacks, graphs):
                selected_graphs[
                    f"{pd.Timestamp(target).date()}__lookback_{lookback}"
                ] = pd.DataFrame(graph, index=index, columns=index)
        topology, previous_edges = _graph_topology_metrics(
            ensemble,
            assets,
            config.edge_tolerance,
            asset_classes,
            previous_edges,
        )
        diagnostic_rows.append(
            {
                "date": target,
                "asset_count": len(assets),
                "status": "PASS",
                "cvx_solves": cvx_solves,
                "fast_solves": fast_solves,
                "scs_fallbacks": scs_fallbacks,
                **topology,
            }
        )
        solved_date_number += 1

    if not feature_rows:
        raise ValueError("no dates produced rolling network features")
    panel = pd.concat(feature_rows, keys=feature_dates, names=["date", "asset"]).sort_index()
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("date").sort_index()
    lookback_panels = None
    if retain_lookback_features:
        lookback_panels = {
            lookback: pd.concat(
                lookback_rows[lookback],
                keys=lookback_dates[lookback],
                names=["date", "asset"],
            ).sort_index()
            for lookback in config.lookbacks
            if lookback_rows[lookback]
        }
    class_ablation_panels = None
    if retain_class_ablation_features:
        class_ablation_panels = {
            name: pd.concat(rows, keys=class_dates[name], names=["date", "asset"]).sort_index()
            for name, rows in class_rows.items()
            if rows
        }
    return DynamicNetworkFeatures(
        panel=panel,
        diagnostics=diagnostics,
        lookback_panels=lookback_panels,
        selected_graphs=selected_graphs or None,
        class_ablation_panels=class_ablation_panels,
    )
