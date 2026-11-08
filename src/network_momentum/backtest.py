"""Replication orchestration.

The only executable milestone in this revision is Stage A (validated U_t).
Graph learning and the full backtest are deliberately not launched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from .config import DEFAULT_CONFIG, UNDOCUMENTED_CHOICES, ReplicationConfig
from .data_utils import audit_long_model_data, load_model_inputs, validate_model_inputs
from .features import (
    StageAFeatures,
    build_stage_a_features,
    validate_stage_a_causality_by_truncation,
    validate_stage_a_features,
)
from .graph_learning import (
    build_graph_lookback,
    ensemble_graphs,
    learn_graph_cvxpy_edges,
    learn_graph_fast,
    normalize_graph,
    prepare_graph_feature_history,
    validate_adjacency,
)
from .metrics import performance_summary
from .hyperparameter import grid_search_graph_parameters
from .network_features import (
    DynamicNetworkFeatures,
    build_dynamic_network_features,
    build_dynamic_network_features_rolling,
    propagate_network_features,
)
from .portfolio import causal_portfolio_vol_rescale
from .paper_analysis import save_main_paper_outputs, save_table4_regression_inference
from .robustness import (
    paper_asset_classes,
    save_asset_class_robustness_outputs,
    save_graph_topology_outputs,
    save_lookback_robustness_outputs,
)
from .regression import build_next_day_scaled_target, fit_pooled_ols, predict_pooled_ols


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_stage_a_artifacts(
    artifacts: StageAFeatures,
    config: ReplicationConfig,
    input_report: dict[str, Any],
    long_audit_report: dict[str, Any],
    feature_report: dict[str, Any],
    causality_report: dict[str, Any],
) -> None:
    output = config.paths.output_dir
    feature_dir = output / "features"
    qa_dir = output / "qa"
    feature_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "daily_volatility.parquet": artifacts.daily_volatility,
        "annualized_volatility.parquet": artifacts.annualized_volatility,
        "vol_scaled_return_features.parquet": artifacts.vol_scaled_returns,
        "macd_features.parquet": artifacts.macd_features,
        "U_raw.parquet": artifacts.u_raw,
        "winsor_lower_bounds.parquet": artifacts.winsor_lower,
        "winsor_upper_bounds.parquet": artifacts.winsor_upper,
        "U_winsorized.parquet": artifacts.u_winsorized,
    }
    for name, frame in files.items():
        frame.to_parquet(feature_dir / name)

    coverage = artifacts.u_winsorized.notna().groupby(level="asset").sum()
    coverage.to_csv(qa_dir / "feature_coverage_by_asset.csv")
    _write_json(qa_dir / "input_qa.json", input_report)
    _write_json(qa_dir / "long_model_data_audit.json", long_audit_report)
    _write_json(qa_dir / "feature_qa.json", feature_report)
    _write_json(qa_dir / "causality_qa.json", causality_report)
    _write_json(
        qa_dir / "stage_a_manifest.json",
        {
            "milestone": "validated_U_t",
            "status": "PASS",
            "authoritative_inputs": {
                "returns": str(config.paths.returns_path),
                "availability": str(config.paths.availability_path),
                "supporting_prices_for_macd": str(config.paths.prices_path),
                "long_audit_source": str(config.paths.model_data_path),
            },
            "config": config.to_dict(),
            "undocumented_choices": UNDOCUMENTED_CHOICES,
            "outputs": {name: str(feature_dir / name) for name in files},
            "next_stage": "Stage B: one-date graph prototype; not executed",
        },
    )


def run_stage_a(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> tuple[StageAFeatures, dict[str, Any]]:
    inputs = load_model_inputs(config.paths)
    input_report = validate_model_inputs(inputs, config.data)
    long_audit_report = audit_long_model_data(config.paths, inputs)
    artifacts = build_stage_a_features(
        inputs.returns,
        inputs.prices,
        inputs.availability,
        config.features,
    )
    feature_report = validate_stage_a_features(artifacts, config.features)
    causality_report = validate_stage_a_causality_by_truncation(
        inputs.returns,
        inputs.prices,
        inputs.availability,
        artifacts,
        config.features,
    )
    reports = {
        "input_qa": input_report,
        "long_model_data_audit": long_audit_report,
        "feature_qa": feature_report,
        "causality_qa": causality_report,
    }
    if save:
        save_stage_a_artifacts(
            artifacts,
            config,
            input_report,
            long_audit_report,
            feature_report,
            causality_report,
        )
    return artifacts, reports


def run_stage_b_prototype(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Learn five graphs for one date, ensemble them, and construct U_tilde."""
    u_path = config.paths.output_dir / "features" / "U_winsorized.parquet"
    if not u_path.exists():
        run_stage_a(config, save=True)
    u_panel = pd.read_parquet(u_path)
    target = pd.Timestamp(config.graph.prototype_date)
    available_dates = u_panel.index.get_level_values("date").unique()
    if target not in available_dates:
        eligible = available_dates[available_dates <= target]
        if len(eligible) == 0:
            raise ValueError(f"No feature date on or before {target.date()}")
        target = pd.Timestamp(eligible[-1])

    prepared = prepare_graph_feature_history(
        u_panel,
        ffill_limit=config.graph.feature_ffill_limit,
    )
    longest = max(config.graph.lookbacks)
    longest_input = build_graph_lookback(
        u_panel,
        target,
        longest,
        calendar_policy=config.graph.calendar_policy,
        ffill_limit=config.graph.feature_ffill_limit,
        require_current_unfilled_features=config.graph.require_current_unfilled_features,
        prepared_history=prepared,
    )
    common_assets = longest_input.assets

    graph_frames: dict[int, pd.DataFrame] = {}
    graph_reports: dict[str, Any] = {}
    graph_arrays: list[np.ndarray] = []
    inputs_by_lookback: dict[int, Any] = {}
    for lookback in config.graph.lookbacks:
        graph_input = build_graph_lookback(
            u_panel,
            target,
            lookback,
            calendar_policy=config.graph.calendar_policy,
            ffill_limit=config.graph.feature_ffill_limit,
            require_current_unfilled_features=config.graph.require_current_unfilled_features,
            prepared_history=prepared,
            asset_subset=common_assets,
        )
        inputs_by_lookback[lookback] = graph_input
        fast_solution = learn_graph_fast(
            graph_input.values,
            config.graph.prototype_alpha,
            config.graph.prototype_beta,
            max_iterations=20000,
        )
        solution = learn_graph_cvxpy_edges(
            graph_input.values,
            config.graph.prototype_alpha,
            config.graph.prototype_beta,
            solver=config.graph.prototype_solver,
            solver_options={"max_iter": 1000},
        )
        adjacency = solution.adjacency
        validation = validate_adjacency(adjacency, tolerance=1e-8)
        upper = adjacency[np.triu_indices_from(adjacency, k=1)]
        graph_arrays.append(adjacency)
        graph_frames[lookback] = pd.DataFrame(
            adjacency,
            index=pd.Index(common_assets, name="asset"),
            columns=pd.Index(common_assets, name="asset"),
        )
        graph_reports[str(lookback)] = {
            "assets": len(common_assets),
            "history_start": graph_input.dates.min().isoformat(),
            "history_end": graph_input.dates.max().isoformat(),
            "causal_short_gap_fills": graph_input.fill_count,
            "solver_success": solution.success,
            "solver_status": solution.status,
            "solver_iterations": solution.iterations,
            "objective": solution.objective,
            "fast_solver_success": fast_solution.success,
            "fast_solver_status": fast_solution.status,
            "fast_solver_iterations": fast_solution.iterations,
            "fast_solver_objective": fast_solution.objective,
            "fast_solver_relative_objective_gap": abs(
                fast_solution.objective - solution.objective
            ) / max(abs(solution.objective), 1e-12),
            "nonzero_edges_upper_at_1e-8": int((upper > 1e-8).sum()),
            "adjacency_checks": validation,
        }

    ensemble = ensemble_graphs(graph_arrays)
    ensemble_validation = validate_adjacency(ensemble, tolerance=1e-8)
    normalized = normalize_graph(ensemble)
    u_t = u_panel.xs(target, level="date").loc[common_assets]
    u_tilde = propagate_network_features(normalized, u_t)
    if not isinstance(u_tilde, pd.DataFrame):
        raise AssertionError("network feature propagation did not preserve DataFrame labels")
    manual = normalized[0] @ u_t.to_numpy(dtype=float)
    manual_error = float(np.max(np.abs(manual - u_tilde.iloc[0].to_numpy(dtype=float))))
    if manual_error > 1e-12 or u_tilde.shape != u_t.shape:
        raise AssertionError(
            f"network feature check failed: shape={u_tilde.shape}, error={manual_error}"
        )

    ensemble_frame = pd.DataFrame(
        ensemble,
        index=pd.Index(common_assets, name="asset"),
        columns=pd.Index(common_assets, name="asset"),
    )
    normalized_frame = pd.DataFrame(
        normalized,
        index=pd.Index(common_assets, name="asset"),
        columns=pd.Index(common_assets, name="asset"),
    )
    u_tilde.index.name = "asset"
    u_tilde.columns = u_t.columns
    qa = {
        "status": "PASS",
        "milestone": "single_date_five_graph_ensemble_and_network_features",
        "date": target.isoformat(),
        "common_asset_count": len(common_assets),
        "common_assets": common_assets,
        "calendar_policy": config.graph.calendar_policy,
        "feature_ffill_limit": config.graph.feature_ffill_limit,
        "alpha": config.graph.prototype_alpha,
        "beta": config.graph.prototype_beta,
        "solver": f"CVXPY/{config.graph.prototype_solver} on scaled upper-triangle weights",
        "paper_solver": "MOSEK (not installed)",
        "fast_solver_role": "diagnostic only; not approved for historical production",
        "graphs": graph_reports,
        "ensemble_checks": ensemble_validation,
        "network_feature_shape": list(u_tilde.shape),
        "manual_propagation_max_abs_error": manual_error,
        "next_stage": "historical dynamic network features and pooled OLS prototype",
    }

    if save:
        graph_dir = config.paths.output_dir / "graphs"
        feature_dir = config.paths.output_dir / "features"
        qa_dir = config.paths.output_dir / "qa"
        graph_dir.mkdir(parents=True, exist_ok=True)
        feature_dir.mkdir(parents=True, exist_ok=True)
        qa_dir.mkdir(parents=True, exist_ok=True)
        date_tag = target.strftime("%Y-%m-%d")
        for lookback, frame in graph_frames.items():
            frame.to_parquet(graph_dir / f"prototype_{date_tag}_A_{lookback}.parquet")
        ensemble_frame.to_parquet(graph_dir / f"prototype_{date_tag}_A_ensemble.parquet")
        normalized_frame.to_parquet(graph_dir / f"prototype_{date_tag}_A_normalized.parquet")
        u_tilde.to_parquet(feature_dir / f"network_features_prototype_{date_tag}.parquet")
        _write_json(qa_dir / "stage_b_prototype_qa.json", qa)
    return u_tilde, qa


def run_stage_cd_prototype(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> tuple[pd.Series, dict[str, Any]]:
    """Build a short dynamic history, fit pooled OLS, and verify P&L timing."""
    u_path = config.paths.output_dir / "features" / "U_winsorized.parquet"
    daily_vol_path = config.paths.output_dir / "features" / "daily_volatility.parquet"
    annual_vol_path = config.paths.output_dir / "features" / "annualized_volatility.parquet"
    if not (u_path.exists() and daily_vol_path.exists() and annual_vol_path.exists()):
        run_stage_a(config, save=True)
    u_panel = pd.read_parquet(u_path)
    daily_vol = pd.read_parquet(daily_vol_path)
    annual_vol = pd.read_parquet(annual_vol_path)
    inputs = load_model_inputs(config.paths)

    all_dates = u_panel.index.get_level_values("date").unique()
    end = pd.Timestamp(config.graph.prototype_date)
    selected = all_dates[all_dates <= end][-config.graph.dynamic_prototype_dates :]
    if len(selected) < 10:
        raise ValueError("dynamic prototype needs at least 10 feature dates")
    dynamic = build_dynamic_network_features(u_panel, pd.DatetimeIndex(selected), config.graph)

    dynamic_dates = dynamic.panel.index.get_level_values("date").unique()
    split = int(len(dynamic_dates) * config.graph.dynamic_prototype_train_fraction)
    split = min(max(split, 2), len(dynamic_dates) - 2)
    test_start = pd.Timestamp(dynamic_dates[split])
    test_dates = pd.DatetimeIndex(dynamic_dates[split:])

    return_dates = inputs.returns.index
    next_date_map = pd.Series(return_dates[1:].to_numpy(), index=return_dates[:-1])
    eligible_train_dates = pd.DatetimeIndex(
        [
            date
            for date in dynamic_dates[:split]
            if date in next_date_map.index and pd.Timestamp(next_date_map.loc[date]) < test_start
        ]
    )
    if len(eligible_train_dates) == 0:
        raise AssertionError("no causally eligible training dates")
    realized_train_max = max(pd.Timestamp(next_date_map.loc[date]) for date in eligible_train_dates)
    if not realized_train_max < test_start:
        raise AssertionError("training labels reach into the test period")

    target_wide = build_next_day_scaled_target(inputs.returns, daily_vol)
    target = target_wide.stack(future_stack=True).rename("target")
    target.index = target.index.set_names(["date", "asset"])
    x_train = dynamic.panel.loc[(eligible_train_dates, slice(None)), :]
    x_test = dynamic.panel.loc[(test_dates, slice(None)), :]
    model = fit_pooled_ols(x_train, target)
    forecast = predict_pooled_ols(model, x_test)
    if len(model.coef_) != 8 or x_train.shape[1] != 8:
        raise AssertionError("pooled OLS does not have exactly eight coefficients")

    next_return = inputs.returns.shift(-1).stack(future_stack=True).rename("next_return")
    next_return.index = next_return.index.set_names(["date", "asset"])
    annual_vol_long = annual_vol.stack(future_stack=True).rename("annual_volatility")
    annual_vol_long.index = annual_vol_long.index.set_names(["date", "asset"])
    pnl_frame = (
        forecast.rename("forecast")
        .to_frame()
        .join(next_return, how="left")
        .join(annual_vol_long, how="left")
    )
    pnl_frame["direction"] = np.sign(pnl_frame["forecast"])
    pnl_frame["asset_strategy_return"] = (
        pnl_frame["direction"]
        * config.portfolio.annual_asset_vol_target
        / pnl_frame["annual_volatility"]
        * pnl_frame["next_return"]
    )
    portfolio_return = (
        pnl_frame["asset_strategy_return"]
        .groupby(level="date")
        .mean()
        .dropna()
        .rename("gmom_prototype_return")
    )
    if portfolio_return.empty:
        raise AssertionError("prototype produced no portfolio returns")

    timing_rows = []
    for date in portfolio_return.index[:5]:
        realized = pd.Timestamp(next_date_map.loc[date])
        if not realized > date:
            raise AssertionError("realized return date is not after feature date")
        timing_rows.append(
            {
                "feature_date": date.isoformat(),
                "signal_date": date.isoformat(),
                "position_date": date.isoformat(),
                "realized_return_date": realized.isoformat(),
            }
        )

    complete_training_rows = int(x_train.join(target).dropna().shape[0])
    complete_test_forecasts = int(forecast.notna().sum())
    report = {
        "status": "PASS",
        "milestone": "dynamic_network_features_pooled_ols_and_pnl_prototype",
        "date_start": pd.Timestamp(dynamic_dates.min()).isoformat(),
        "date_end": pd.Timestamp(dynamic_dates.max()).isoformat(),
        "requested_dates": int(len(selected)),
        "dynamic_dates": int(len(dynamic_dates)),
        "training_feature_date_end": eligible_train_dates.max().isoformat(),
        "maximum_training_label_realization_date": realized_train_max.isoformat(),
        "test_start": test_start.isoformat(),
        "train_before_test": bool(realized_train_max < test_start),
        "training_rows_after_complete_case_filter": complete_training_rows,
        "test_forecasts": complete_test_forecasts,
        "network_feature_count": int(x_train.shape[1]),
        "ols_intercept": float(model.intercept_),
        "ols_coefficients": {
            feature: float(value)
            for feature, value in zip(x_train.columns, model.coef_)
        },
        "timing_examples": timing_rows,
        "test_portfolio_days": int(len(portfolio_return)),
        "test_performance_for_mechanics_only": performance_summary(
            portfolio_return,
            trading_days=config.portfolio.trading_days,
        ),
        "solver_status_counts": dynamic.diagnostics["solver_statuses"].value_counts().to_dict(),
        "next_stage": "first 2000-2004 walk-forward block after hyperparameter protocol is frozen",
    }
    if save:
        feature_dir = config.paths.output_dir / "features"
        signal_dir = config.paths.output_dir / "signals"
        result_dir = config.paths.output_dir / "results"
        qa_dir = config.paths.output_dir / "qa"
        for directory in (feature_dir, signal_dir, result_dir, qa_dir):
            directory.mkdir(parents=True, exist_ok=True)
        dynamic.panel.to_parquet(feature_dir / "network_features_dynamic_prototype.parquet")
        dynamic.diagnostics.to_csv(qa_dir / "dynamic_graph_prototype_diagnostics.csv")
        pnl_frame.to_parquet(signal_dir / "gmom_dynamic_prototype_signals.parquet")
        portfolio_return.to_frame().to_parquet(
            result_dir / "gmom_dynamic_prototype_returns.parquet"
        )
        _write_json(qa_dir / "stage_cd_prototype_qa.json", report)
    return portfolio_return, report


def _strategy_pnl_from_position(
    position: pd.Series,
    next_return: pd.Series,
    annual_volatility: pd.Series,
    annual_target: float,
) -> tuple[pd.Series, pd.DataFrame]:
    frame = (
        position.rename("position")
        .to_frame()
        .join(next_return.rename("next_return"), how="left")
        .join(annual_volatility.rename("annual_volatility"), how="left")
    )
    frame["asset_strategy_return"] = (
        frame["position"]
        * annual_target
        / frame["annual_volatility"]
        * frame["next_return"]
    )
    portfolio = frame["asset_strategy_return"].groupby(level="date").mean().dropna()
    return portfolio, frame


def _regression_inference_rows(
    block: str,
    model_name: str,
    model: Any,
    columns: pd.Index | list[str],
) -> list[dict[str, Any]]:
    """Return the coefficient/SE/t/p fields required by paper Table 4."""
    rows = [
        {
            "block": block,
            "model": model_name,
            "feature": "intercept",
            "coefficient": float(model.intercept_),
            "standard_error": float(model.intercept_standard_error_),
            "t_statistic": float(model.intercept_t_statistic_),
            "p_value": float(model.intercept_p_value_),
            "significant_5pct": bool(model.intercept_p_value_ < 0.05),
            "nobs": int(model.nobs_),
            "covariance": model.covariance_choice_,
        }
    ]
    for feature, coefficient, standard_error, t_statistic, p_value in zip(
        columns,
        model.coef_,
        model.standard_errors_,
        model.t_statistics_,
        model.p_values_,
    ):
        rows.append(
            {
                "block": block,
                "model": model_name,
                "feature": str(feature),
                "coefficient": float(coefficient),
                "standard_error": float(standard_error),
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
                "significant_5pct": bool(p_value < 0.05),
                "nobs": int(model.nobs_),
                "covariance": model.covariance_choice_,
            }
        )
    return rows


def run_first_walkforward_block(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Engineering validation of Train<=1999 -> Test 2000-2004.

    Graph alpha/beta are fixed at the configured prototype values because the
    paper does not disclose its validation scoring rule. Results are therefore
    labelled approximate, not a final faithful Table-1 replication.
    """
    u_path = config.paths.output_dir / "features" / "U_winsorized.parquet"
    daily_vol_path = config.paths.output_dir / "features" / "daily_volatility.parquet"
    annual_vol_path = config.paths.output_dir / "features" / "annualized_volatility.parquet"
    if not (u_path.exists() and daily_vol_path.exists() and annual_vol_path.exists()):
        run_stage_a(config, save=True)
    u_panel = pd.read_parquet(u_path)
    daily_vol = pd.read_parquet(daily_vol_path)
    annual_vol = pd.read_parquet(annual_vol_path)
    inputs = load_model_inputs(config.paths)
    dates = u_panel.index.get_level_values("date").unique()
    block_dates = dates[(dates >= "1994-01-01") & (dates <= "2004-12-31")]
    dynamic = build_dynamic_network_features_rolling(
        u_panel,
        pd.DatetimeIndex(block_dates),
        config.graph,
        reference_interval=20,
    )
    network = dynamic.panel
    network_dates = network.index.get_level_values("date")
    train_end = pd.Timestamp("1999-12-31")
    test_start = pd.Timestamp("2000-01-01")
    test_end = pd.Timestamp("2004-12-31")

    next_date_map = pd.Series(inputs.returns.index[1:].to_numpy(), index=inputs.returns.index[:-1])
    train_dates = network_dates.unique()
    train_dates = pd.DatetimeIndex(
        [
            date
            for date in train_dates[train_dates <= train_end]
            if date in next_date_map.index and pd.Timestamp(next_date_map.loc[date]) <= train_end
        ]
    )
    test_dates = network_dates.unique()
    test_dates = test_dates[(test_dates >= test_start) & (test_dates <= test_end)]
    if len(train_dates) == 0 or len(test_dates) == 0:
        raise AssertionError("first block lacks train or test network dates")
    x_gmom_train = network.loc[(train_dates, slice(None)), :]
    x_gmom_test = network.loc[(test_dates, slice(None)), :]
    x_linreg_train = u_panel.reindex(x_gmom_train.index)
    x_linreg_test = u_panel.reindex(x_gmom_test.index)

    target_wide = build_next_day_scaled_target(inputs.returns, daily_vol)
    target = target_wide.stack(future_stack=True).rename("target")
    target.index = target.index.set_names(["date", "asset"])
    gmom_model = fit_pooled_ols(x_gmom_train, target)
    linreg_model = fit_pooled_ols(x_linreg_train, target)
    gmom_forecast = predict_pooled_ols(gmom_model, x_gmom_test)
    linreg_forecast = predict_pooled_ols(linreg_model, x_linreg_test)

    next_return = inputs.returns.shift(-1).stack(future_stack=True).rename("next_return")
    next_return.index = next_return.index.set_names(["date", "asset"])
    annual_vol_long = annual_vol.stack(future_stack=True).rename("annual_volatility")
    annual_vol_long.index = annual_vol_long.index.set_names(["date", "asset"])
    gmom_position = np.sign(gmom_forecast).where(gmom_forecast.notna())
    linreg_position = np.sign(linreg_forecast).where(linreg_forecast.notna())
    baseline_features = x_linreg_test
    macd_columns = [f"macd_{short}_{long}" for short, long in config.features.macd_windows]
    macd_transformed = baseline_features[macd_columns].apply(
        lambda column: column * np.exp(-(column * column) / 4.0) / 0.89
    )
    macd_position = macd_transformed.mean(axis=1, skipna=False).rename("position")
    long_only_position = pd.Series(1.0, index=x_gmom_test.index, name="position")

    positions = {
        "Long Only": long_only_position,
        "MACD": macd_position,
        "LinReg": linreg_position,
        "GMOM": gmom_position,
    }
    raw_returns: dict[str, pd.Series] = {}
    asset_frames: list[pd.DataFrame] = []
    for strategy, position_series in positions.items():
        portfolio, asset_frame = _strategy_pnl_from_position(
            position_series,
            next_return,
            annual_vol_long,
            config.portfolio.annual_asset_vol_target,
        )
        raw_returns[strategy] = portfolio.rename(strategy)
        asset_frame = asset_frame.copy()
        asset_frame["strategy"] = strategy
        asset_frames.append(asset_frame)
    raw_panel = pd.concat(raw_returns, axis=1).sort_index()
    scaled_panel = pd.concat(
        {
            strategy: causal_portfolio_vol_rescale(
                raw_panel[strategy],
                annual_target=config.portfolio.annual_asset_vol_target,
                span=config.portfolio.portfolio_vol_span,
                min_periods=config.portfolio.portfolio_vol_min_periods,
                trading_days=config.portfolio.trading_days,
                adjust=config.portfolio.portfolio_vol_ewm_adjust,
                leverage_cap=config.portfolio.portfolio_vol_leverage_cap,
            )
            for strategy in raw_panel.columns
        },
        axis=1,
    )

    raw_metrics = {strategy: performance_summary(raw_panel[strategy]) for strategy in raw_panel}
    scaled_metrics = {
        strategy: performance_summary(scaled_panel[strategy]) for strategy in scaled_panel
    }
    metrics_rows = []
    for panel_name, metrics_by_strategy in (("raw", raw_metrics), ("portfolio_vol_scaled", scaled_metrics)):
        for strategy, values in metrics_by_strategy.items():
            metrics_rows.append({"panel": panel_name, "strategy": strategy, **values})
    metrics_table = pd.DataFrame(metrics_rows).set_index(["panel", "strategy"])

    report = {
        "status": "PASS_APPROXIMATE",
        "milestone": "first_walkforward_block_2000_2004",
        "methodology_deviation": (
            "alpha/beta fixed at prototype values; paper does not disclose graph-grid validation score"
        ),
        "alpha": config.graph.prototype_alpha,
        "beta": config.graph.prototype_beta,
        "training_network_date_start": train_dates.min().isoformat(),
        "training_feature_date_end": train_dates.max().isoformat(),
        "maximum_training_label_realization_date": max(
            pd.Timestamp(next_date_map.loc[date]) for date in train_dates
        ).isoformat(),
        "test_date_start": test_dates.min().isoformat(),
        "test_date_end": test_dates.max().isoformat(),
        "train_before_test": bool(
            max(pd.Timestamp(next_date_map.loc[date]) for date in train_dates) < test_dates.min()
        ),
        "gmom_training_rows": int(x_gmom_train.join(target).dropna().shape[0]),
        "gmom_test_forecasts": int(gmom_forecast.notna().sum()),
        "linreg_training_rows": int(x_linreg_train.join(target).dropna().shape[0]),
        "graph_diagnostics": {
            "dates_passed": int((dynamic.diagnostics["status"] == "PASS").sum()),
            "dates_skipped": int((dynamic.diagnostics["status"] == "SKIPPED").sum()),
            "cvx_solves": int(dynamic.diagnostics.get("cvx_solves", pd.Series(dtype=float)).fillna(0).sum()),
            "fast_solves": int(dynamic.diagnostics.get("fast_solves", pd.Series(dtype=float)).fillna(0).sum()),
            "scs_fallbacks": int(dynamic.diagnostics.get("scs_fallbacks", pd.Series(dtype=float)).fillna(0).sum()),
        },
        "raw_metrics": raw_metrics,
        "portfolio_vol_scaled_metrics": scaled_metrics,
        "gmom_coefficients": {
            feature: float(value) for feature, value in zip(x_gmom_train.columns, gmom_model.coef_)
        },
        "linreg_coefficients": {
            feature: float(value) for feature, value in zip(x_linreg_train.columns, linreg_model.coef_)
        },
        "next_stage": "freeze graph validation rule, then run all five OOS blocks",
    }
    if not report["train_before_test"]:
        raise AssertionError("first block training labels overlap the test period")

    if save:
        feature_dir = config.paths.output_dir / "features"
        signal_dir = config.paths.output_dir / "signals"
        result_dir = config.paths.output_dir / "results"
        qa_dir = config.paths.output_dir / "qa"
        for directory in (feature_dir, signal_dir, result_dir, qa_dir):
            directory.mkdir(parents=True, exist_ok=True)
        network.to_parquet(feature_dir / "network_features_first_block.parquet")
        dynamic.diagnostics.to_csv(qa_dir / "first_block_graph_diagnostics.csv")
        pd.concat(asset_frames).to_parquet(signal_dir / "first_block_asset_signals.parquet")
        raw_panel.to_parquet(result_dir / "first_block_raw_returns.parquet")
        scaled_panel.to_parquet(result_dir / "first_block_vol_scaled_returns.parquet")
        metrics_table.to_csv(result_dir / "first_block_metrics.csv")
        _write_json(qa_dir / "first_block_qa.json", report)
    return metrics_table, report


def run_full_backtest(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run all five OOS blocks with fixed, explicitly approximate graph parameters."""
    first_network_path = config.paths.output_dir / "features" / "network_features_first_block.parquet"
    if not first_network_path.exists():
        run_first_walkforward_block(config, save=True)
    u_panel = pd.read_parquet(config.paths.output_dir / "features" / "U_winsorized.parquet")
    daily_vol = pd.read_parquet(config.paths.output_dir / "features" / "daily_volatility.parquet")
    annual_vol = pd.read_parquet(config.paths.output_dir / "features" / "annualized_volatility.parquet")
    inputs = load_model_inputs(config.paths)
    early_network = pd.read_parquet(first_network_path)
    all_dates = u_panel.index.get_level_values("date").unique()
    remaining_dates = all_dates[(all_dates >= "2005-01-01") & (all_dates <= "2022-12-31")]
    later_dynamic = build_dynamic_network_features_rolling(
        u_panel,
        pd.DatetimeIndex(remaining_dates),
        config.graph,
        reference_interval=20,
    )
    network = pd.concat([early_network, later_dynamic.panel]).sort_index()
    network = network[~network.index.duplicated(keep="last")]

    target_wide = build_next_day_scaled_target(inputs.returns, daily_vol)
    target = target_wide.stack(future_stack=True).rename("target")
    target.index = target.index.set_names(["date", "asset"])
    next_date_map = pd.Series(inputs.returns.index[1:].to_numpy(), index=inputs.returns.index[:-1])
    schedules = [
        ("2000-2004", pd.Timestamp("1999-12-31"), pd.Timestamp("2000-01-01"), pd.Timestamp("2004-12-31")),
        ("2005-2009", pd.Timestamp("2004-12-31"), pd.Timestamp("2005-01-01"), pd.Timestamp("2009-12-31")),
        ("2010-2014", pd.Timestamp("2009-12-31"), pd.Timestamp("2010-01-01"), pd.Timestamp("2014-12-31")),
        ("2015-2019", pd.Timestamp("2014-12-31"), pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")),
        ("2020-2022", pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
    ]
    gmom_forecasts: list[pd.Series] = []
    linreg_forecasts: list[pd.Series] = []
    regcombo_forecasts: list[pd.Series] = []
    coefficient_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    block_reports: list[dict[str, Any]] = []
    available_network_dates = network.index.get_level_values("date").unique()

    for block, train_end, test_start, test_end in schedules:
        train_dates = pd.DatetimeIndex(
            [
                date
                for date in available_network_dates[available_network_dates <= train_end]
                if date in next_date_map.index and pd.Timestamp(next_date_map.loc[date]) <= train_end
            ]
        )
        test_dates = available_network_dates[
            (available_network_dates >= test_start) & (available_network_dates <= test_end)
        ]
        x_gmom_train = network.loc[(train_dates, slice(None)), :]
        x_gmom_test = network.loc[(test_dates, slice(None)), :]
        x_linreg_train = u_panel.reindex(x_gmom_train.index)
        x_linreg_test = u_panel.reindex(x_gmom_test.index)
        x_regcombo_train = pd.concat(
            [
                x_linreg_train.add_prefix("individual__"),
                x_gmom_train.add_prefix("network__"),
            ],
            axis=1,
        )
        x_regcombo_test = pd.concat(
            [
                x_linreg_test.add_prefix("individual__"),
                x_gmom_test.add_prefix("network__"),
            ],
            axis=1,
        )
        gmom_model = fit_pooled_ols(x_gmom_train, target)
        linreg_model = fit_pooled_ols(x_linreg_train, target)
        regcombo_model = fit_pooled_ols(
            x_regcombo_train, target, expected_features=16
        )
        gmom_prediction = predict_pooled_ols(gmom_model, x_gmom_test)
        linreg_prediction = predict_pooled_ols(linreg_model, x_linreg_test)
        regcombo_prediction = predict_pooled_ols(regcombo_model, x_regcombo_test)
        gmom_forecasts.append(gmom_prediction)
        linreg_forecasts.append(linreg_prediction)
        regcombo_forecasts.append(regcombo_prediction)
        realized_train_max = max(pd.Timestamp(next_date_map.loc[date]) for date in train_dates)
        if not realized_train_max < test_dates.min():
            raise AssertionError(f"training labels overlap test block {block}")
        block_reports.append(
            {
                "block": block,
                "train_feature_start": train_dates.min().isoformat(),
                "train_feature_end": train_dates.max().isoformat(),
                "last_train_label_date": realized_train_max.isoformat(),
                "test_start": test_dates.min().isoformat(),
                "test_end": test_dates.max().isoformat(),
                "gmom_train_rows": int(x_gmom_train.join(target).dropna().shape[0]),
                "gmom_test_forecasts": int(gmom_prediction.notna().sum()),
            }
        )
        for model_name, model, columns in (
            ("GMOM", gmom_model, x_gmom_train.columns),
            ("LinReg", linreg_model, x_linreg_train.columns),
            ("RegCombo", regcombo_model, x_regcombo_train.columns),
        ):
            row = {"block": block, "model": model_name, "intercept": float(model.intercept_)}
            row.update({feature: float(value) for feature, value in zip(columns, model.coef_)})
            coefficient_rows.append(row)
            inference_rows.extend(
                _regression_inference_rows(block, model_name, model, list(columns))
            )

    gmom_forecast = pd.concat(gmom_forecasts).sort_index()
    linreg_forecast = pd.concat(linreg_forecasts).sort_index()
    regcombo_forecast = pd.concat(regcombo_forecasts).sort_index()
    test_index = gmom_forecast.index
    baseline_features = u_panel.reindex(test_index)
    macd_columns = [f"macd_{short}_{long}" for short, long in config.features.macd_windows]
    macd_position = baseline_features[macd_columns].apply(
        lambda column: column * np.exp(-(column * column) / 4.0) / 0.89
    ).mean(axis=1, skipna=False)
    linreg_direction = np.sign(linreg_forecast).where(linreg_forecast.notna())
    gmom_direction = np.sign(gmom_forecast).where(gmom_forecast.notna())
    positions = {
        "Long Only": pd.Series(1.0, index=test_index),
        "MACD": macd_position,
        "LinReg": linreg_direction,
        "GMOM": gmom_direction,
        "RegCombo": np.sign(regcombo_forecast).where(regcombo_forecast.notna()),
        "SignCombo": 0.5 * linreg_direction + 0.5 * gmom_direction,
    }
    next_return = inputs.returns.shift(-1).stack(future_stack=True).rename("next_return")
    next_return.index = next_return.index.set_names(["date", "asset"])
    annual_vol_long = annual_vol.stack(future_stack=True).rename("annual_volatility")
    annual_vol_long.index = annual_vol_long.index.set_names(["date", "asset"])
    raw_returns: dict[str, pd.Series] = {}
    signal_frames: list[pd.DataFrame] = []
    for strategy, position in positions.items():
        portfolio, asset_frame = _strategy_pnl_from_position(
            position,
            next_return,
            annual_vol_long,
            config.portfolio.annual_asset_vol_target,
        )
        raw_returns[strategy] = portfolio.rename(strategy)
        asset_frame["strategy"] = strategy
        signal_frames.append(asset_frame)
    raw_panel = pd.concat(raw_returns, axis=1).sort_index()
    scaled_panel = pd.concat(
        {
            strategy: causal_portfolio_vol_rescale(
                raw_panel[strategy],
                annual_target=config.portfolio.annual_asset_vol_target,
                span=config.portfolio.portfolio_vol_span,
                min_periods=config.portfolio.portfolio_vol_min_periods,
                trading_days=config.portfolio.trading_days,
                adjust=config.portfolio.portfolio_vol_ewm_adjust,
                leverage_cap=config.portfolio.portfolio_vol_leverage_cap,
            )
            for strategy in raw_panel.columns
        },
        axis=1,
    )
    raw_metrics = {strategy: performance_summary(raw_panel[strategy]) for strategy in raw_panel}
    scaled_metrics = {
        strategy: performance_summary(scaled_panel[strategy]) for strategy in scaled_panel
    }
    metrics_rows = [
        {"panel": panel, "strategy": strategy, **values}
        for panel, collection in (("raw", raw_metrics), ("portfolio_vol_scaled", scaled_metrics))
        for strategy, values in collection.items()
    ]
    metrics_table = pd.DataFrame(metrics_rows).set_index(["panel", "strategy"])
    report = {
        "status": "PASS_APPROXIMATE",
        "milestone": "five_block_oos_2000_2022_fixed_graph_parameters",
        "methodology_deviation": (
            "graph alpha/beta fixed at 0.1/0.1; paper states 10% validation grid search but not its score"
        ),
        "alpha": config.graph.prototype_alpha,
        "beta": config.graph.prototype_beta,
        "blocks": block_reports,
        "raw_metrics": raw_metrics,
        "portfolio_vol_scaled_metrics": scaled_metrics,
        "later_graph_diagnostics": {
            "dates_passed": int((later_dynamic.diagnostics["status"] == "PASS").sum()),
            "dates_skipped": int((later_dynamic.diagnostics["status"] == "SKIPPED").sum()),
            "cvx_solves": int(later_dynamic.diagnostics["cvx_solves"].fillna(0).sum()),
            "fast_solves": int(later_dynamic.diagnostics["fast_solves"].fillna(0).sum()),
            "scs_fallbacks": int(later_dynamic.diagnostics["scs_fallbacks"].fillna(0).sum()),
        },
        "paper_comparison": {
            "paper_raw_gmom_sharpe": 1.363,
            "paper_scaled_gmom_sharpe": 1.511,
            "paper_scaled_gmom_return": 0.222,
        },
        "next_stage": "resolve graph validation score and MOSEK availability for strict replication",
    }
    if save:
        feature_dir = config.paths.output_dir / "features"
        signal_dir = config.paths.output_dir / "signals"
        result_dir = config.paths.output_dir / "results"
        qa_dir = config.paths.output_dir / "qa"
        for directory in (feature_dir, signal_dir, result_dir, qa_dir):
            directory.mkdir(parents=True, exist_ok=True)
        network.to_parquet(feature_dir / "network_features_full_fixed_params.parquet")
        later_dynamic.diagnostics.to_csv(qa_dir / "full_fixed_graph_diagnostics_2005_2022.csv")
        pd.DataFrame(coefficient_rows).to_csv(result_dir / "full_fixed_coefficients.csv", index=False)
        pd.DataFrame(inference_rows).to_csv(
            result_dir / "full_fixed_regression_inference.csv", index=False
        )
        pd.concat(signal_frames).to_parquet(signal_dir / "full_fixed_asset_signals.parquet")
        raw_panel.to_parquet(result_dir / "full_fixed_raw_returns.parquet")
        scaled_panel.to_parquet(result_dir / "full_fixed_vol_scaled_returns.parquet")
        metrics_table.to_csv(result_dir / "full_fixed_metrics.csv")
        _write_json(qa_dir / "full_fixed_qa.json", report)
    return metrics_table, report


def _save_robustness_checkpoint(
    dynamic: DynamicNetworkFeatures,
    checkpoint_dir: Path,
) -> None:
    """Persist the expensive graph pass before plotting/backtesting begins."""
    if dynamic.lookback_panels is None or dynamic.class_ablation_panels is None:
        raise AssertionError("robustness checkpoint requires retained panels")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = checkpoint_dir / "selected_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    dynamic.panel.to_parquet(checkpoint_dir / "ensemble_panel.parquet")
    dynamic.diagnostics.to_parquet(checkpoint_dir / "diagnostics.parquet")
    lookback_files: dict[str, str] = {}
    for lookback, panel in dynamic.lookback_panels.items():
        name = f"lookback_{lookback}.parquet"
        panel.to_parquet(checkpoint_dir / name)
        lookback_files[str(lookback)] = name
    class_files: dict[str, str] = {}
    for label, panel in dynamic.class_ablation_panels.items():
        tag = label.lower().replace(" ", "_").replace("-", "_")
        name = f"class_{tag}.parquet"
        panel.to_parquet(checkpoint_dir / name)
        class_files[label] = name
    graph_files: dict[str, str] = {}
    for key, graph in (dynamic.selected_graphs or {}).items():
        name = f"{key}.parquet"
        graph.to_parquet(graph_dir / name)
        graph_files[key] = name
    _write_json(
        checkpoint_dir / "checkpoint_manifest.json",
        {
            "status": "PASS",
            "ensemble_panel": "ensemble_panel.parquet",
            "diagnostics": "diagnostics.parquet",
            "lookback_panels": lookback_files,
            "class_ablation_panels": class_files,
            "selected_graphs": graph_files,
        },
    )


def _load_robustness_checkpoint(checkpoint_dir: Path) -> DynamicNetworkFeatures:
    """Load a complete graph checkpoint written by `_save_robustness_checkpoint`."""
    manifest = json.loads(
        (checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "PASS":
        raise ValueError("robustness checkpoint is not complete")
    lookbacks = {
        int(key): pd.read_parquet(checkpoint_dir / name)
        for key, name in manifest["lookback_panels"].items()
    }
    classes = {
        key: pd.read_parquet(checkpoint_dir / name)
        for key, name in manifest["class_ablation_panels"].items()
    }
    graph_dir = checkpoint_dir / "selected_graphs"
    graphs = {
        key: pd.read_parquet(graph_dir / name)
        for key, name in manifest["selected_graphs"].items()
    }
    return DynamicNetworkFeatures(
        panel=pd.read_parquet(checkpoint_dir / manifest["ensemble_panel"]),
        diagnostics=pd.read_parquet(checkpoint_dir / manifest["diagnostics"]),
        lookback_panels=lookbacks,
        selected_graphs=graphs,
        class_ablation_panels=classes,
    )


def run_paper_robustness(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> dict[str, Any]:
    """Recompute retained graph artifacts for paper Tables 2--3/Figures 4--10."""
    u_panel = pd.read_parquet(config.paths.output_dir / "features" / "U_winsorized.parquet")
    daily_vol = pd.read_parquet(config.paths.output_dir / "features" / "daily_volatility.parquet")
    annual_vol = pd.read_parquet(config.paths.output_dir / "features" / "annualized_volatility.parquet")
    inputs = load_model_inputs(config.paths)
    dates = u_panel.index.get_level_values("date").unique()
    graph_dates = dates[(dates >= "1994-01-01") & (dates <= "2022-12-31")]
    asset_classes = paper_asset_classes(list(inputs.returns.columns))
    selected_dates = {
        pd.Timestamp("2006-12-22"),
        pd.Timestamp("2020-03-23"),
        pd.Timestamp("2022-04-05"),
    }
    checkpoint_dir = config.paths.output_dir / "robustness_checkpoint"
    checkpoint_manifest = checkpoint_dir / "checkpoint_manifest.json"
    if checkpoint_manifest.exists():
        dynamic = _load_robustness_checkpoint(checkpoint_dir)
    else:
        dynamic = build_dynamic_network_features_rolling(
            u_panel,
            pd.DatetimeIndex(graph_dates),
            config.graph,
            reference_interval=config.graph.production_reference_interval,
            retain_lookback_features=True,
            selected_graph_dates=selected_dates,
            asset_classes=asset_classes,
            retain_class_ablation_features=True,
        )
        if save:
            _save_robustness_checkpoint(dynamic, checkpoint_dir)
    if dynamic.lookback_panels is None or dynamic.class_ablation_panels is None:
        raise AssertionError("robustness graph pass did not retain requested panels")
    paper_dir = config.paths.replication_dir
    target_wide = build_next_day_scaled_target(inputs.returns, daily_vol)
    target = target_wide.stack(future_stack=True).rename("target")
    target.index = target.index.set_names(["date", "asset"])
    full_scaled = pd.read_parquet(
        config.paths.output_dir / "results" / "full_fixed_vol_scaled_returns.parquet"
    )
    full_signals = pd.read_parquet(
        config.paths.output_dir / "signals" / "full_fixed_asset_signals.parquet"
    )
    topology_outputs = save_graph_topology_outputs(dynamic, paper_dir)
    lookback_outputs = save_lookback_robustness_outputs(
        dynamic.lookback_panels,
        full_scaled["GMOM"],
        target,
        inputs.returns,
        annual_vol,
        config.portfolio,
        paper_dir,
    )
    class_outputs = save_asset_class_robustness_outputs(
        dynamic.class_ablation_panels,
        full_signals,
        full_scaled["GMOM"],
        target,
        inputs.returns,
        annual_vol,
        config.portfolio,
        paper_dir,
    )
    report = {
        "status": "PASS",
        "milestone": "paper_robustness_tables_2_3_figures_4_10",
        "graph_dates_requested": int(len(graph_dates)),
        "graph_dates_passed": int((dynamic.diagnostics["status"] == "PASS").sum()),
        "selected_graphs": sorted((dynamic.selected_graphs or {}).keys()),
        "lookback_panels": sorted(dynamic.lookback_panels),
        "class_ablation_panels": sorted(dynamic.class_ablation_panels),
        "topology_outputs": {key: str(value) for key, value in topology_outputs.items()},
        "lookback_outputs": {key: str(value) for key, value in lookback_outputs.items()},
        "class_outputs": {key: str(value) for key, value in class_outputs.items()},
    }
    if save:
        feature_dir = config.paths.output_dir / "features"
        qa_dir = paper_dir / "qa"
        feature_dir.mkdir(parents=True, exist_ok=True)
        qa_dir.mkdir(parents=True, exist_ok=True)
        dynamic.diagnostics.to_csv(qa_dir / "paper_graph_diagnostics.csv")
        for lookback, panel in dynamic.lookback_panels.items():
            panel.to_parquet(feature_dir / f"network_features_lookback_{lookback}.parquet")
        for name, panel in dynamic.class_ablation_panels.items():
            tag = name.lower().replace(" ", "_")
            panel.to_parquet(feature_dir / f"network_features_{tag}.parquet")
        _write_json(qa_dir / "robustness_manifest.json", report)
    return report


def run_graph_grid_searches(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> dict[str, Any]:
    """Run the paper's 10%-holdout 11x11 grid for all five blocks."""
    u_panel = pd.read_parquet(config.paths.output_dir / "features" / "U_winsorized.parquet")
    daily_vol = pd.read_parquet(config.paths.output_dir / "features" / "daily_volatility.parquet")
    annual_vol = pd.read_parquet(config.paths.output_dir / "features" / "annualized_volatility.parquet")
    inputs = load_model_inputs(config.paths)
    target_wide = build_next_day_scaled_target(inputs.returns, daily_vol)
    target = target_wide.stack(future_stack=True).rename("target")
    target.index = target.index.set_names(["date", "asset"])
    next_return = inputs.returns.shift(-1).stack(future_stack=True).rename("next_return")
    next_return.index = next_return.index.set_names(["date", "asset"])
    annual_vol_long = annual_vol.stack(future_stack=True).rename("annual_volatility")
    annual_vol_long.index = annual_vol_long.index.set_names(["date", "asset"])
    available_dates = u_panel.index.get_level_values("date").unique()
    next_date_map = pd.Series(inputs.returns.index[1:].to_numpy(), index=inputs.returns.index[:-1])
    train_ends = (
        ("2000-2004", pd.Timestamp("1999-12-31")),
        ("2005-2009", pd.Timestamp("2004-12-31")),
        ("2010-2014", pd.Timestamp("2009-12-31")),
        ("2015-2019", pd.Timestamp("2014-12-31")),
        ("2020-2022", pd.Timestamp("2019-12-31")),
    )
    selection: dict[str, Any] = {}
    qa_dir = config.paths.output_dir / "paper_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for block, train_end in train_ends:
        train_dates = pd.DatetimeIndex(
            [
                date
                for date in available_dates[available_dates <= train_end]
                if date in next_date_map.index
                and pd.Timestamp(next_date_map.loc[date]) <= train_end
            ]
        )
        result = grid_search_graph_parameters(
            u_panel,
            target,
            next_return,
            annual_vol_long,
            train_dates,
            config.graph,
            annual_target=config.portfolio.annual_asset_vol_target,
            trading_days=config.portfolio.trading_days,
        )
        selection[block] = {
            "alpha": result.alpha,
            "beta": result.beta,
            "validation_start": result.validation_dates.min().isoformat(),
            "validation_end": result.validation_dates.max().isoformat(),
            "validation_dates_evaluated": int(len(result.validation_dates)),
            "validation_fraction": config.graph.validation_fraction,
            "validation_metric": config.graph.validation_metric,
            "candidates_evaluated": int(len(result.scores)),
            "candidates_passed": int((result.scores["status"] == "PASS").sum()),
            "best_score": float(result.scores.iloc[0]["score"]),
        }
        if save:
            result.scores.to_csv(qa_dir / f"grid_search_{block}.csv", index=False)
    report = {
        "status": "PASS",
        "method": "latest 10% of each training period; complete 11x11 disclosed grid",
        "undisclosed_choice": (
            f"pooled asset-day forecast correlation on an internal {config.graph.validation_internal_train_fraction:.0%}/"
            f"{1-config.graph.validation_internal_train_fraction:.0%} split of up to "
            f"{config.graph.validation_max_dates} evenly spaced validation dates"
        ),
        "selection": selection,
    }
    if save:
        _write_json(qa_dir / "graph_grid_search_summary.json", report)
    return report


def run_grid_selected_backtest(
    config: ReplicationConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild every OOS block using its validation-selected alpha/beta."""
    summary_path = config.paths.output_dir / "paper_qa" / "graph_grid_search_summary.json"
    if not summary_path.exists():
        run_graph_grid_searches(config, save=True)
    selection = json.loads(summary_path.read_text(encoding="utf-8"))["selection"]
    u_panel = pd.read_parquet(config.paths.output_dir / "features" / "U_winsorized.parquet")
    daily_vol = pd.read_parquet(config.paths.output_dir / "features" / "daily_volatility.parquet")
    annual_vol = pd.read_parquet(config.paths.output_dir / "features" / "annualized_volatility.parquet")
    inputs = load_model_inputs(config.paths)
    target_wide = build_next_day_scaled_target(inputs.returns, daily_vol)
    target = target_wide.stack(future_stack=True).rename("target")
    target.index = target.index.set_names(["date", "asset"])
    next_date_map = pd.Series(inputs.returns.index[1:].to_numpy(), index=inputs.returns.index[:-1])
    schedules = [
        ("2000-2004", pd.Timestamp("1999-12-31"), pd.Timestamp("2000-01-01"), pd.Timestamp("2004-12-31")),
        ("2005-2009", pd.Timestamp("2004-12-31"), pd.Timestamp("2005-01-01"), pd.Timestamp("2009-12-31")),
        ("2010-2014", pd.Timestamp("2009-12-31"), pd.Timestamp("2010-01-01"), pd.Timestamp("2014-12-31")),
        ("2015-2019", pd.Timestamp("2014-12-31"), pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")),
        ("2020-2022", pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
    ]
    gmom_predictions: list[pd.Series] = []
    linreg_predictions: list[pd.Series] = []
    regcombo_predictions: list[pd.Series] = []
    inference_rows: list[dict[str, Any]] = []
    block_reports: list[dict[str, Any]] = []
    feature_dir = config.paths.output_dir / "features" / "grid_selected_blocks"
    qa_dir = config.paths.output_dir / "paper_qa" / "grid_selected_blocks"
    if save:
        feature_dir.mkdir(parents=True, exist_ok=True)
        qa_dir.mkdir(parents=True, exist_ok=True)
    all_dates = u_panel.index.get_level_values("date").unique()
    for block, train_end, test_start, test_end in schedules:
        alpha = float(selection[block]["alpha"])
        beta = float(selection[block]["beta"])
        block_graph = replace(
            config.graph,
            prototype_alpha=alpha,
            prototype_beta=beta,
        )
        requested = all_dates[(all_dates >= "1994-01-01") & (all_dates <= test_end)]
        dynamic = build_dynamic_network_features_rolling(
            u_panel,
            pd.DatetimeIndex(requested),
            block_graph,
            reference_interval=block_graph.production_reference_interval,
        )
        network = dynamic.panel
        network_dates = network.index.get_level_values("date").unique()
        train_dates = pd.DatetimeIndex(
            [
                date
                for date in network_dates[network_dates <= train_end]
                if date in next_date_map.index
                and pd.Timestamp(next_date_map.loc[date]) <= train_end
            ]
        )
        test_dates = network_dates[
            (network_dates >= test_start) & (network_dates <= test_end)
        ]
        x_gmom_train = network.loc[(train_dates, slice(None)), :]
        x_gmom_test = network.loc[(test_dates, slice(None)), :]
        x_linreg_train = u_panel.reindex(x_gmom_train.index)
        x_linreg_test = u_panel.reindex(x_gmom_test.index)
        x_combo_train = pd.concat(
            [x_linreg_train.add_prefix("individual__"), x_gmom_train.add_prefix("network__")],
            axis=1,
        )
        x_combo_test = pd.concat(
            [x_linreg_test.add_prefix("individual__"), x_gmom_test.add_prefix("network__")],
            axis=1,
        )
        gmom_model = fit_pooled_ols(x_gmom_train, target)
        linreg_model = fit_pooled_ols(x_linreg_train, target)
        combo_model = fit_pooled_ols(x_combo_train, target, expected_features=16)
        gmom_predictions.append(predict_pooled_ols(gmom_model, x_gmom_test))
        linreg_predictions.append(predict_pooled_ols(linreg_model, x_linreg_test))
        regcombo_predictions.append(predict_pooled_ols(combo_model, x_combo_test))
        for name, model, columns in (
            ("GMOM", gmom_model, x_gmom_train.columns),
            ("LinReg", linreg_model, x_linreg_train.columns),
            ("RegCombo", combo_model, x_combo_train.columns),
        ):
            inference_rows.extend(
                _regression_inference_rows(block, name, model, list(columns))
            )
        block_reports.append(
            {
                "block": block,
                "alpha": alpha,
                "beta": beta,
                "train_feature_end": train_dates.max().isoformat(),
                "last_train_label_date": max(
                    pd.Timestamp(next_date_map.loc[date]) for date in train_dates
                ).isoformat(),
                "test_start": test_dates.min().isoformat(),
                "test_end": test_dates.max().isoformat(),
                "network_rows": int(len(network)),
                "cvx_solves": int(dynamic.diagnostics["cvx_solves"].fillna(0).sum()),
                "fast_solves": int(dynamic.diagnostics["fast_solves"].fillna(0).sum()),
            }
        )
        if save:
            network.to_parquet(feature_dir / f"network_features_{block}.parquet")
            dynamic.diagnostics.to_csv(qa_dir / f"graph_diagnostics_{block}.csv")

    gmom_forecast = pd.concat(gmom_predictions).sort_index()
    linreg_forecast = pd.concat(linreg_predictions).sort_index()
    combo_forecast = pd.concat(regcombo_predictions).sort_index()
    test_index = gmom_forecast.index
    baseline = u_panel.reindex(test_index)
    macd_columns = [f"macd_{short}_{long}" for short, long in config.features.macd_windows]
    macd_position = baseline[macd_columns].apply(
        lambda column: column * np.exp(-(column * column) / 4.0) / 0.89
    ).mean(axis=1, skipna=False)
    gmom_direction = np.sign(gmom_forecast).where(gmom_forecast.notna())
    linreg_direction = np.sign(linreg_forecast).where(linreg_forecast.notna())
    positions = {
        "Long Only": pd.Series(1.0, index=test_index),
        "MACD": macd_position,
        "LinReg": linreg_direction,
        "GMOM": gmom_direction,
        "RegCombo": np.sign(combo_forecast).where(combo_forecast.notna()),
        "SignCombo": 0.5 * linreg_direction + 0.5 * gmom_direction,
    }
    next_return = inputs.returns.shift(-1).stack(future_stack=True).rename("next_return")
    next_return.index = next_return.index.set_names(["date", "asset"])
    annual_vol_long = annual_vol.stack(future_stack=True).rename("annual_volatility")
    annual_vol_long.index = annual_vol_long.index.set_names(["date", "asset"])
    raw_returns: dict[str, pd.Series] = {}
    signal_frames: list[pd.DataFrame] = []
    for strategy, position in positions.items():
        portfolio, frame = _strategy_pnl_from_position(
            position,
            next_return,
            annual_vol_long,
            config.portfolio.annual_asset_vol_target,
        )
        raw_returns[strategy] = portfolio.rename(strategy)
        frame["strategy"] = strategy
        signal_frames.append(frame)
    raw_panel = pd.concat(raw_returns, axis=1).sort_index()
    scaled_panel = pd.concat(
        {
            strategy: causal_portfolio_vol_rescale(
                raw_panel[strategy],
                annual_target=config.portfolio.annual_asset_vol_target,
                span=config.portfolio.portfolio_vol_span,
                min_periods=config.portfolio.portfolio_vol_min_periods,
                trading_days=config.portfolio.trading_days,
                adjust=config.portfolio.portfolio_vol_ewm_adjust,
                leverage_cap=config.portfolio.portfolio_vol_leverage_cap,
            )
            for strategy in raw_panel
        },
        axis=1,
    )
    metrics_rows = []
    for panel_name, panel in (("raw", raw_panel), ("portfolio_vol_scaled", scaled_panel)):
        for strategy in panel:
            metrics_rows.append(
                {"panel": panel_name, "strategy": strategy, **performance_summary(panel[strategy])}
            )
    metrics = pd.DataFrame(metrics_rows).set_index(["panel", "strategy"])
    report = {
        "status": "PASS",
        "milestone": "five_block_grid_selected_oos_2000_2022",
        "validation_selection": selection,
        "blocks": block_reports,
        "gmom_raw": performance_summary(raw_panel["GMOM"]),
        "gmom_scaled": performance_summary(scaled_panel["GMOM"]),
    }
    if save:
        result_dir = config.paths.output_dir / "results"
        signal_dir = config.paths.output_dir / "signals"
        result_dir.mkdir(parents=True, exist_ok=True)
        signal_dir.mkdir(parents=True, exist_ok=True)
        raw_panel.to_parquet(result_dir / "paper_grid_selected_raw_returns.parquet")
        scaled_panel.to_parquet(result_dir / "paper_grid_selected_scaled_returns.parquet")
        pd.concat(signal_frames).to_parquet(signal_dir / "paper_grid_selected_asset_signals.parquet")
        pd.DataFrame(inference_rows).to_csv(
            result_dir / "paper_grid_selected_regression_inference.csv", index=False
        )
        metrics.to_csv(result_dir / "paper_grid_selected_metrics.csv")
        _write_json(config.paths.output_dir / "paper_qa" / "grid_selected_backtest.json", report)
    return metrics, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Network Momentum replication runner")
    parser.add_argument(
        "--stage",
        choices=[
            "features",
            "graph-prototype",
            "model-prototype",
            "first-block",
            "full",
            "paper-main",
            "paper-robustness",
            "paper-grid",
            "paper-grid-backtest",
        ],
        default="features",
        help="Only the validated Stage-A feature milestone is enabled.",
    )
    parser.add_argument("--no-save", action="store_true", help="Validate without saving outputs")
    args = parser.parse_args()
    if args.stage == "full":
        _, report = run_full_backtest(save=not args.no_save)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.stage == "paper-main":
        result_dir = DEFAULT_CONFIG.paths.output_dir / "results"
        signal_dir = DEFAULT_CONFIG.paths.output_dir / "signals"
        outputs = save_main_paper_outputs(
            pd.read_parquet(result_dir / "full_fixed_raw_returns.parquet"),
            pd.read_parquet(result_dir / "full_fixed_vol_scaled_returns.parquet"),
            pd.read_parquet(signal_dir / "full_fixed_asset_signals.parquet"),
            DEFAULT_CONFIG.paths.replication_dir,
            trading_days=DEFAULT_CONFIG.portfolio.trading_days,
        )
        table4_numeric, table4_formatted = save_table4_regression_inference(
            pd.read_csv(result_dir / "full_fixed_regression_inference.csv"),
            DEFAULT_CONFIG.paths.replication_dir,
        )
        outputs["table_4_numeric"] = table4_numeric
        outputs["table_4_formatted"] = table4_formatted
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return
    if args.stage == "paper-robustness":
        report = run_paper_robustness(save=not args.no_save)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.stage == "paper-grid":
        report = run_graph_grid_searches(save=not args.no_save)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.stage == "paper-grid-backtest":
        _, report = run_grid_selected_backtest(save=not args.no_save)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.stage == "graph-prototype":
        _, report = run_stage_b_prototype(save=not args.no_save)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.stage == "model-prototype":
        _, report = run_stage_cd_prototype(save=not args.no_save)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.stage == "first-block":
        _, report = run_first_walkforward_block(save=not args.no_save)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    _, reports = run_stage_a(save=not args.no_save)
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
