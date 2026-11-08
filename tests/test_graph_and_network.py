from __future__ import annotations

import numpy as np
import pandas as pd

from network_momentum.graph_learning import (
    ensemble_graphs,
    graph_objective,
    graph_objective_from_distances,
    learn_graph_fast,
    normalize_graph,
    validate_adjacency,
)
from network_momentum.network_features import propagate_network_features


def test_graph_ensemble_normalization_and_propagation() -> None:
    a1 = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.25], [0.5, 0.25, 0.0]])
    a2 = a1 * 2.0
    ensemble = ensemble_graphs([a1, a2])
    report = validate_adjacency(ensemble)
    assert report["symmetric"]
    normalized = normalize_graph(ensemble)
    u_t = pd.DataFrame(np.arange(24, dtype=float).reshape(3, 8), index=["A", "B", "C"])
    propagated = propagate_network_features(normalized, u_t)
    assert propagated.shape == u_t.shape
    np.testing.assert_allclose(propagated.to_numpy(), normalized @ u_t.to_numpy())


def test_fast_graph_solver_enforces_constraints() -> None:
    rng = np.random.default_rng(11)
    v = rng.normal(size=(8, 40))
    solution = learn_graph_fast(v, alpha=0.1, beta=0.1, max_iterations=5000)
    report = validate_adjacency(solution.adjacency)
    assert report["finite"]
    assert report["nonnegative"]
    assert report["symmetric"]
    assert report["zero_diagonal"]
    assert report["minimum_degree"] > 0


def test_pairwise_distance_objective_matches_stacked_feature_objective() -> None:
    rng = np.random.default_rng(19)
    v = rng.normal(size=(6, 32))
    raw = rng.uniform(0.0, 1.0, size=(6, 6))
    adjacency = (raw + raw.T) / 2.0
    np.fill_diagonal(adjacency, 0.0)
    norms = np.sum(v * v, axis=1)
    distances = np.maximum(norms[:, None] + norms[None, :] - 2.0 * (v @ v.T), 0.0)
    direct = graph_objective(adjacency, v, alpha=0.1, beta=0.1)
    rolling = graph_objective_from_distances(adjacency, distances, alpha=0.1, beta=0.1)
    np.testing.assert_allclose(direct, rolling, rtol=1e-12, atol=1e-12)
