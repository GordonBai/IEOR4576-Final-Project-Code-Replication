from __future__ import annotations

import numpy as np
import pandas as pd

from network_momentum.metrics import performance_summary
from network_momentum.paper_analysis import STRATEGY_ORDER, build_complete_metrics_table
from network_momentum.regression import fit_pooled_ols


def test_paper_metrics_include_duration_fraction_and_profit_loss_ratio() -> None:
    returns = pd.Series([0.01, -0.02, -0.01, 0.03, 0.01])
    summary = performance_summary(returns)
    assert 0.0 <= summary["maximum_drawdown_duration_fraction"] <= 1.0
    assert np.isclose(
        summary["average_profit_loss_ratio"],
        returns[returns > 0].mean() / abs(returns[returns < 0].mean()),
    )


def test_pooled_ols_exposes_table4_inference() -> None:
    rng = np.random.default_rng(3)
    index = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=20), ["A", "B", "C"]],
        names=["date", "asset"],
    )
    x = pd.DataFrame(rng.normal(size=(len(index), 8)), index=index)
    y = pd.Series(0.5 * x.iloc[:, 0] + rng.normal(scale=0.1, size=len(index)), index=index)
    model = fit_pooled_ols(x, y)
    assert len(model.standard_errors_) == 8
    assert len(model.t_statistics_) == 8
    assert len(model.p_values_) == 8
    assert model.nobs_ == len(index)


def test_complete_table1_requires_and_reports_six_strategies() -> None:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=300)
    raw = pd.DataFrame(
        rng.normal(0.0002, 0.005, (len(dates), len(STRATEGY_ORDER))),
        index=dates,
        columns=STRATEGY_ORDER,
    )
    scaled = raw * 1.5
    table = build_complete_metrics_table(raw, scaled)
    assert table.shape[0] == 12
    assert set(table.index.get_level_values("strategy")) == set(STRATEGY_ORDER)
    assert "average_profit_loss_ratio" in table.columns
