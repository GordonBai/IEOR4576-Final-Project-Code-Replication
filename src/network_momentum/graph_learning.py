"""Stage-B graph utilities.

This module is implemented for the next milestone, but the Stage-A command does
not call the optimizer. Hyperparameter selection remains intentionally blocked
until the paper/code selection metric is confirmed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphLookback:
    values: np.ndarray
    assets: list[str]
    dates: pd.DatetimeIndex
    fill_count: int


@dataclass(frozen=True)
class GraphSolution:
    adjacency: np.ndarray
    objective: float
    success: bool
    status: str
    iterations: int


def prepare_graph_feature_history(
    u_panel: pd.DataFrame,
    ffill_limit: int = 5,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return original and causally short-gap-filled feature matrices.

    The fill is restricted to graph history. Stage-A artifacts and the current
    propagated U_t remain unfilled.
    """
    if list(u_panel.index.names) != ["date", "asset"]:
        raise ValueError("u_panel index must be ['date', 'asset']")
    if ffill_limit < 0:
        raise ValueError("ffill_limit must be non-negative")
    original = {feature: u_panel[feature].unstack("asset") for feature in u_panel.columns}
    filled = {
        feature: frame.ffill(limit=ffill_limit) if ffill_limit else frame.copy()
        for feature, frame in original.items()
    }
    return original, filled


def build_graph_lookback_matrix(
    u_panel: pd.DataFrame,
    date: pd.Timestamp | str,
    lookback: int,
    *,
    calendar_policy: str = "strict",
    ffill_limit: int = 5,
    require_current_unfilled_features: bool = True,
    prepared_history: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None = None,
    asset_subset: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Create V_t with rows=assets and columns=8*lookback.

    Assets are retained only when all eight features are present for every one
    of the last ``lookback`` panel dates through ``date``.
    """
    if list(u_panel.index.names) != ["date", "asset"]:
        raise ValueError("u_panel index must be ['date', 'asset']")
    target = pd.Timestamp(date)
    dates = u_panel.index.get_level_values("date").unique()
    dates = dates[dates <= target]
    if len(dates) < lookback:
        raise ValueError(f"Need {lookback} dates through {target.date()}, got {len(dates)}")
    selected_dates = pd.DatetimeIndex(dates[-lookback:])
    if calendar_policy not in {"strict", "causal_ffill"}:
        raise ValueError(f"Unsupported calendar_policy: {calendar_policy}")
    if prepared_history is None:
        original, filled = prepare_graph_feature_history(u_panel, ffill_limit=ffill_limit)
    else:
        original, filled = prepared_history
    source = original if calendar_policy == "strict" else filled
    assets = (
        pd.Index(asset_subset)
        if asset_subset is not None
        else original[next(iter(original))].columns
    )
    valid_assets: list[str] = []
    for asset in assets:
        history_complete = all(
            source[feature].loc[selected_dates, asset].notna().all()
            for feature in u_panel.columns
        )
        current_complete = all(
            original[feature].at[target, asset] == original[feature].at[target, asset]
            for feature in u_panel.columns
        )
        if history_complete and (current_complete or not require_current_unfilled_features):
            valid_assets.append(str(asset))
    if not valid_assets:
        raise ValueError("No assets have a complete feature history for this lookback")
    # Date-major flattening: [u_1,t-lookback+1, ..., u_8,t,].
    v = np.vstack([
        np.column_stack([
            source[feature].loc[selected_dates, asset].to_numpy(dtype=float)
            for feature in u_panel.columns
        ]).reshape(-1)
        for asset in valid_assets
    ])
    return v, valid_assets


def build_graph_lookback(
    u_panel: pd.DataFrame,
    date: pd.Timestamp | str,
    lookback: int,
    *,
    calendar_policy: str = "causal_ffill",
    ffill_limit: int = 5,
    require_current_unfilled_features: bool = True,
    prepared_history: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]] | None = None,
    asset_subset: Sequence[str] | None = None,
) -> GraphLookback:
    target = pd.Timestamp(date)
    dates = u_panel.index.get_level_values("date").unique()
    selected_dates = pd.DatetimeIndex(dates[dates <= target][-lookback:])
    if prepared_history is None:
        prepared_history = prepare_graph_feature_history(u_panel, ffill_limit)
    original, filled = prepared_history
    values, assets = build_graph_lookback_matrix(
        u_panel,
        target,
        lookback,
        calendar_policy=calendar_policy,
        ffill_limit=ffill_limit,
        require_current_unfilled_features=require_current_unfilled_features,
        prepared_history=prepared_history,
        asset_subset=asset_subset,
    )
    fill_count = 0
    if calendar_policy == "causal_ffill":
        fill_count = int(sum(
            (
                original[feature].loc[selected_dates, assets].isna()
                & filled[feature].loc[selected_dates, assets].notna()
            ).sum().sum()
            for feature in u_panel.columns
        ))
    return GraphLookback(values=values, assets=assets, dates=selected_dates, fill_count=fill_count)


def learn_graph(
    v: np.ndarray,
    alpha: float,
    beta: float,
    solver: str | None = None,
    solver_options: dict | None = None,
) -> np.ndarray:
    """Solve the paper's nonnegative symmetric graph-learning objective."""
    import cvxpy as cp

    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2:
        raise ValueError("v must be a 2D array with at least two assets")
    if not np.isfinite(v).all():
        raise ValueError("v contains non-finite values")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")

    n_assets = v.shape[0]
    squared_norms = np.sum(v * v, axis=1)
    pairwise_squared_distance = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (v @ v.T)
    )
    pairwise_squared_distance = np.maximum(pairwise_squared_distance, 0.0)
    a = cp.Variable((n_assets, n_assets), symmetric=True)
    degree = cp.sum(a, axis=1)
    objective = cp.Minimize(
        0.5 * cp.sum(cp.multiply(pairwise_squared_distance, a))
        - alpha * cp.sum(cp.log(degree))
        + beta * cp.sum_squares(a)
    )
    constraints = [a >= 0, cp.diag(a) == 0]
    problem = cp.Problem(objective, constraints)
    options = dict(solver_options or {})
    problem.solve(solver=solver, **options)
    if a.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Graph optimization failed with status {problem.status}")
    result = np.asarray(a.value, dtype=float)
    result[np.abs(result) < 1e-12] = 0.0
    return result


def graph_objective(
    a: np.ndarray,
    v: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Evaluate Eq. (4) exactly for a symmetric adjacency matrix."""
    a = np.asarray(a, dtype=float)
    v = np.asarray(v, dtype=float)
    degree = a.sum(axis=1)
    if np.any(degree <= 0):
        return float("inf")
    squared_norms = np.sum(v * v, axis=1)
    distances = np.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (v @ v.T),
        0.0,
    )
    smoothness = 0.5 * float(np.sum(a * distances))
    return smoothness - alpha * float(np.log(degree).sum()) + beta * float(np.sum(a * a))


def graph_objective_from_distances(
    a: np.ndarray,
    pairwise_squared_distance: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    a = np.asarray(a, dtype=float)
    distances = np.asarray(pairwise_squared_distance, dtype=float)
    if a.shape != distances.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency and distance matrices must be equally sized square matrices")
    degree = a.sum(axis=1)
    if np.any(degree <= 0):
        return float("inf")
    return (
        0.5 * float(np.sum(a * distances))
        - alpha * float(np.log(degree).sum())
        + beta * float(np.sum(a * a))
    )


def learn_graph_fast_from_distances(
    pairwise_squared_distance: np.ndarray,
    alpha: float,
    beta: float,
    *,
    warm_start: np.ndarray | None = None,
    max_iterations: int = 3000,
    tolerance: float = 1e-10,
) -> GraphSolution:
    """L-BFGS-B graph solve when rolling pairwise distances are precomputed."""
    from scipy.optimize import minimize

    distances = np.asarray(pairwise_squared_distance, dtype=float)
    if (
        distances.ndim != 2
        or distances.shape[0] < 2
        or distances.shape[0] != distances.shape[1]
        or not np.isfinite(distances).all()
    ):
        raise ValueError("distance matrix must be finite, square, and at least 2x2")
    n = distances.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    z = np.maximum(distances[rows, cols], 0.0)
    positive_z = z[z > np.finfo(float).eps * 100]
    distance_scale = float(np.median(positive_z)) if positive_z.size else 1.0
    weight_scale = alpha / max(distance_scale, np.finfo(float).tiny)
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=float)
        if warm.shape != (n, n):
            raise ValueError(f"warm_start must have shape {(n, n)}")
        initial = np.maximum(warm[rows, cols] / weight_scale, 1e-10)
    else:
        initial = np.full(len(rows), 1e-6 / max(n - 1, 1), dtype=float)
        full_z = np.full((n, n), np.inf, dtype=float)
        full_z[rows, cols] = z
        full_z[cols, rows] = z
        lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}
        for i in range(n):
            j = int(np.argmin(full_z[i]))
            initial[lookup[(min(i, j), max(i, j))]] += 1.0

    linear = (weight_scale / alpha) * z
    quadratic = 2.0 * beta * weight_scale * weight_scale / alpha

    def value_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        degree = np.bincount(
            np.concatenate([rows, cols]),
            weights=np.concatenate([x, x]),
            minlength=n,
        )
        if np.any(degree <= 0) or not np.isfinite(degree).all():
            return 1e100, np.zeros_like(x)
        value = float(linear @ x) - float(np.log(degree).sum()) + quadratic * float(x @ x)
        gradient = linear - (1.0 / degree[rows] + 1.0 / degree[cols]) + 2.0 * quadratic * x
        return value, gradient

    result = minimize(
        value_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * len(rows),
        options={"maxiter": max_iterations, "ftol": tolerance, "gtol": tolerance, "maxls": 50},
    )
    weights = weight_scale * np.maximum(np.asarray(result.x, dtype=float), 0.0)
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[rows, cols] = weights
    adjacency[cols, rows] = weights
    validate_adjacency(adjacency, tolerance=1e-10)
    if adjacency.sum(axis=1).min() <= 0:
        raise AssertionError("distance-based graph solution contains an isolated node")
    return GraphSolution(
        adjacency=adjacency,
        objective=graph_objective_from_distances(adjacency, distances, alpha, beta),
        success=bool(result.success),
        status=str(result.message),
        iterations=int(result.nit),
    )


def learn_graph_cvxpy_edges_from_distances(
    pairwise_squared_distance: np.ndarray,
    alpha: float,
    beta: float,
    *,
    solver: str = "CLARABEL",
    solver_options: dict | None = None,
) -> GraphSolution:
    """CVXPY reference solve using a precomputed rolling distance matrix."""
    import cvxpy as cp
    from scipy import sparse

    distances = np.asarray(pairwise_squared_distance, dtype=float)
    if (
        distances.ndim != 2
        or distances.shape[0] < 2
        or distances.shape[0] != distances.shape[1]
        or not np.isfinite(distances).all()
    ):
        raise ValueError("distance matrix must be finite, square, and at least 2x2")
    n = distances.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    z = np.maximum(distances[rows, cols], 0.0)
    positive_z = z[z > np.finfo(float).eps * 100]
    distance_scale = float(np.median(positive_z)) if positive_z.size else 1.0
    weight_scale = alpha / max(distance_scale, np.finfo(float).tiny)
    m = len(rows)
    incidence = sparse.coo_matrix(
        (np.ones(2 * m), (np.concatenate([rows, cols]), np.tile(np.arange(m), 2))),
        shape=(n, m),
    ).tocsr()
    x = cp.Variable(m, nonneg=True)
    degree = incidence @ x
    objective = cp.Minimize(
        (weight_scale / alpha) * z @ x
        - cp.sum(cp.log(degree))
        + (2.0 * beta * weight_scale * weight_scale / alpha) * cp.sum_squares(x)
    )
    problem = cp.Problem(objective)
    problem.solve(solver=solver, **dict(solver_options or {}))
    if x.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPY distance solve failed with status {problem.status}")
    weights = weight_scale * np.maximum(np.asarray(x.value), 0.0)
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[rows, cols] = weights
    adjacency[cols, rows] = weights
    return GraphSolution(
        adjacency=adjacency,
        objective=graph_objective_from_distances(adjacency, distances, alpha, beta),
        success=problem.status == cp.OPTIMAL,
        status=str(problem.status),
        iterations=int(getattr(problem.solver_stats, "num_iters", 0) or 0),
    )


def learn_graph_fast(
    v: np.ndarray,
    alpha: float,
    beta: float,
    *,
    warm_start: np.ndarray | None = None,
    max_iterations: int = 10000,
    tolerance: float = 1e-10,
) -> GraphSolution:
    """Solve Eq. (4) over upper-triangular nonnegative edge weights.

    This parameterization enforces symmetry and a zero diagonal exactly. The
    objective is convex; L-BFGS-B supplies nonnegative box constraints.
    """
    from scipy.optimize import minimize

    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2 or not np.isfinite(v).all():
        raise ValueError("v must be a finite 2D array with at least two assets")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")

    n = v.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    m = len(rows)
    squared_norms = np.sum(v * v, axis=1)
    distances = np.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (v @ v.T),
        0.0,
    )
    z = distances[rows, cols]

    positive_z = z[z > np.finfo(float).eps * 100]
    distance_scale = float(np.median(positive_z)) if positive_z.size else 1.0
    weight_scale = alpha / max(distance_scale, np.finfo(float).tiny)

    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=float)
        if warm.shape != (n, n):
            raise ValueError(f"warm_start must have shape {(n, n)}")
        initial = np.maximum(warm[rows, cols] / weight_scale, 1e-10)
    else:
        initial = np.full(m, 1e-6 / max(n - 1, 1), dtype=float)
        # A nearest-neighbour initialization gives every node positive degree
        # without starting from a numerically oversized dense graph.
        full_z = np.full((n, n), np.inf, dtype=float)
        full_z[rows, cols] = z
        full_z[cols, rows] = z
        edge_lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}
        for i in range(n):
            j = int(np.argmin(full_z[i]))
            key = (min(i, j), max(i, j))
            initial[edge_lookup[key]] += 1.0

    def value_and_gradient(scaled_weights: np.ndarray) -> tuple[float, np.ndarray]:
        degree = np.bincount(
            np.concatenate([rows, cols]),
            weights=np.concatenate([scaled_weights, scaled_weights]),
            minlength=n,
        )
        if np.any(degree <= 0) or not np.isfinite(degree).all():
            return 1e100, np.zeros_like(scaled_weights)
        linear_coefficient = (weight_scale / alpha) * z
        quadratic_coefficient = 2.0 * beta * weight_scale * weight_scale / alpha
        value = (
            float(linear_coefficient @ scaled_weights)
            - float(np.log(degree).sum())
            + quadratic_coefficient * float(scaled_weights @ scaled_weights)
        )
        gradient = (
            linear_coefficient
            - (1.0 / degree[rows] + 1.0 / degree[cols])
            + 2.0 * quadratic_coefficient * scaled_weights
        )
        return value, gradient

    result = minimize(
        value_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * m,
        options={
            "maxiter": max_iterations,
            "ftol": tolerance,
            "gtol": tolerance,
            "maxls": 50,
        },
    )
    weights = weight_scale * np.maximum(np.asarray(result.x, dtype=float), 0.0)
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[rows, cols] = weights
    adjacency[cols, rows] = weights
    solution = GraphSolution(
        adjacency=adjacency,
        objective=graph_objective(adjacency, v, alpha, beta),
        success=bool(result.success),
        status=str(result.message),
        iterations=int(result.nit),
    )
    validate_adjacency(adjacency, tolerance=max(tolerance, 1e-10))
    if adjacency.sum(axis=1).min() <= 0:
        raise AssertionError("fast graph solution contains an isolated node")
    return solution


def learn_graph_cvxpy_edges(
    v: np.ndarray,
    alpha: float,
    beta: float,
    *,
    solver: str = "CLARABEL",
    solver_options: dict | None = None,
) -> GraphSolution:
    """Numerical cross-check using CVXPY on scaled upper-triangular weights."""
    import cvxpy as cp
    from scipy import sparse

    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2 or not np.isfinite(v).all():
        raise ValueError("v must be a finite 2D array with at least two assets")
    n = v.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    m = len(rows)
    squared_norms = np.sum(v * v, axis=1)
    distances = np.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (v @ v.T),
        0.0,
    )
    z = distances[rows, cols]
    positive_z = z[z > np.finfo(float).eps * 100]
    distance_scale = float(np.median(positive_z)) if positive_z.size else 1.0
    weight_scale = alpha / max(distance_scale, np.finfo(float).tiny)
    incidence = sparse.coo_matrix(
        (
            np.ones(2 * m),
            (np.concatenate([rows, cols]), np.tile(np.arange(m), 2)),
        ),
        shape=(n, m),
    ).tocsr()
    scaled_weights = cp.Variable(m, nonneg=True)
    degree = incidence @ scaled_weights
    objective = cp.Minimize(
        (weight_scale / alpha) * z @ scaled_weights
        - cp.sum(cp.log(degree))
        + (2.0 * beta * weight_scale * weight_scale / alpha)
        * cp.sum_squares(scaled_weights)
    )
    problem = cp.Problem(objective)
    options = dict(solver_options or {})
    problem.solve(solver=solver, **options)
    if scaled_weights.value is None or problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:
        raise RuntimeError(f"CVXPY edge solve failed with status {problem.status}")
    weights = weight_scale * np.maximum(np.asarray(scaled_weights.value), 0.0)
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[rows, cols] = weights
    adjacency[cols, rows] = weights
    return GraphSolution(
        adjacency=adjacency,
        objective=graph_objective(adjacency, v, alpha, beta),
        success=problem.status == cp.OPTIMAL,
        status=str(problem.status),
        iterations=int(getattr(problem.solver_stats, "num_iters", 0) or 0),
    )


def learn_graph_primal_dual(
    v: np.ndarray,
    alpha: float,
    beta: float,
    *,
    warm_start: np.ndarray | None = None,
    max_iterations: int = 20000,
    tolerance: float = 1e-7,
    check_every: int = 25,
) -> GraphSolution:
    """Chambolle–Pock solver specialized to the log-degree graph objective.

    The variable is the nonnegative upper-triangular edge vector. Symmetry and
    the zero diagonal are therefore exact by construction.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2 or not np.isfinite(v).all():
        raise ValueError("v must be a finite 2D array with at least two assets")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    n = v.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    m = len(rows)
    squared_norms = np.sum(v * v, axis=1)
    distances = np.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (v @ v.T),
        0.0,
    )
    z = distances[rows, cols]
    positive_z = z[z > np.finfo(float).eps * 100]
    distance_scale = float(np.median(positive_z)) if positive_z.size else 1.0
    weight_scale = alpha / max(distance_scale, np.finfo(float).tiny)
    linear = (weight_scale / alpha) * z
    quadratic = 2.0 * beta * weight_scale * weight_scale / alpha

    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=float)
        if warm.shape != (n, n):
            raise ValueError(f"warm_start must have shape {(n, n)}")
        x = np.maximum(warm[rows, cols] / weight_scale, 1e-12)
    else:
        x = np.full(m, 1e-8, dtype=float)
        full_z = np.full((n, n), np.inf, dtype=float)
        full_z[rows, cols] = z
        full_z[cols, rows] = z
        lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}
        for i in range(n):
            j = int(np.argmin(full_z[i]))
            x[lookup[(min(i, j), max(i, j))]] += 1.0

    def degrees(weights: np.ndarray) -> np.ndarray:
        return np.bincount(
            np.concatenate([rows, cols]),
            weights=np.concatenate([weights, weights]),
            minlength=n,
        )

    degree = np.maximum(degrees(x), 1e-12)
    dual = -1.0 / degree
    x_bar = x.copy()
    operator_norm = np.sqrt(2.0 * max(n - 1, 1))
    tau = 0.99 / operator_norm
    sigma = 0.99 / operator_norm
    success = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        dual_argument = dual + sigma * degrees(x_bar)
        dual_new = 0.5 * (
            dual_argument - np.sqrt(dual_argument * dual_argument + 4.0 * sigma)
        )
        primal_argument = x - tau * (dual_new[rows] + dual_new[cols])
        x_new = np.maximum(
            0.0,
            (primal_argument - tau * linear) / (1.0 + 2.0 * tau * quadratic),
        )
        x_bar = 2.0 * x_new - x
        if iteration % check_every == 0:
            relative_change = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1.0)
            if relative_change < tolerance and np.all(degrees(x_new) > 0):
                success = True
                x = x_new
                dual = dual_new
                break
        x = x_new
        dual = dual_new

    weights = weight_scale * np.maximum(x, 0.0)
    adjacency = np.zeros((n, n), dtype=float)
    adjacency[rows, cols] = weights
    adjacency[cols, rows] = weights
    validate_adjacency(adjacency, tolerance=1e-10)
    if adjacency.sum(axis=1).min() <= 0:
        raise AssertionError("primal-dual graph solution contains an isolated node")
    return GraphSolution(
        adjacency=adjacency,
        objective=graph_objective(adjacency, v, alpha, beta),
        success=success,
        status="CONVERGED" if success else "MAX_ITERATIONS",
        iterations=iteration,
    )


def validate_adjacency(a: np.ndarray, tolerance: float = 1e-8) -> dict[str, float | bool]:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise AssertionError("adjacency must be square")
    report = {
        "finite": bool(np.isfinite(a).all()),
        "nonnegative": bool((a >= -tolerance).all()),
        "symmetric": bool(np.allclose(a, a.T, atol=tolerance, rtol=0)),
        "zero_diagonal": bool(np.allclose(np.diag(a), 0, atol=tolerance, rtol=0)),
        "minimum_degree": float(a.sum(axis=1).min()),
        "maximum_degree": float(a.sum(axis=1).max()),
    }
    if not all(report[key] for key in ("finite", "nonnegative", "symmetric", "zero_diagonal")):
        raise AssertionError(f"Invalid adjacency: {report}")
    return report


def ensemble_graphs(graphs: Sequence[np.ndarray]) -> np.ndarray:
    if not graphs:
        raise ValueError("at least one graph is required")
    shapes = {np.asarray(graph).shape for graph in graphs}
    if len(shapes) != 1:
        raise ValueError(f"graph shapes differ: {shapes}")
    return np.mean(np.stack(graphs, axis=0), axis=0)


def normalize_graph(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    validate_adjacency(a)
    degree = a.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    return inverse_sqrt[:, None] * a * inverse_sqrt[None, :]
