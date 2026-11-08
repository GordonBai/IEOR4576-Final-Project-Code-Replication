"""Public evaluation facade for metrics, tables and validation outputs."""

from .metrics import max_drawdown, performance_summary
from .paper_analysis import (
    build_complete_metrics_table,
    return_correlations,
    sign_agreement,
    turnover_and_cost_adjusted_sharpe,
)

__all__ = [
    "build_complete_metrics_table",
    "max_drawdown",
    "performance_summary",
    "return_correlations",
    "sign_agreement",
    "turnover_and_cost_adjusted_sharpe",
]
