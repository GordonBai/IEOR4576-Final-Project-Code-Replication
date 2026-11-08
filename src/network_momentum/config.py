"""Central configuration for the Network Momentum replication.

Any choice not fixed by the paper/handoff is exposed here instead of being
embedded inside a transformation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import (
    ARTIFACTS_DIR,
    CONFIG_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPLICATION_DIR,
)


@dataclass(frozen=True)
class PathConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROCESSED_DATA_DIR
    output_dir: Path = ARTIFACTS_DIR
    replication_dir: Path = REPLICATION_DIR
    returns_file: str = "paper64_returns_v3.parquet"
    availability_file: str = "paper64_availability_v3.parquet"
    prices_file: str = "paper64_prices_v3.parquet"
    model_data_file: str = "paper64_model_data_v3.parquet"

    @property
    def returns_path(self) -> Path:
        return self.data_dir / self.returns_file

    @property
    def availability_path(self) -> Path:
        return self.data_dir / self.availability_file

    @property
    def prices_path(self) -> Path:
        return self.data_dir / self.prices_file

    @property
    def model_data_path(self) -> Path:
        return self.data_dir / self.model_data_file


@dataclass(frozen=True)
class DataConfig:
    expected_assets: int = 64
    return_convention: str = "simple"
    require_sorted_unique_dates: bool = True
    require_availability_exact_match: bool = True
    require_finite_nonmissing_returns: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    sample_start: str = "2000-01-01"
    sample_end: str = "2022-12-31"
    random_seed: int = 7


@dataclass(frozen=True)
class FeatureConfig:
    # Paper-specified choices.
    vol_span: int = 60
    momentum_horizons: tuple[int, ...] = (1, 21, 63, 126, 252)
    macd_windows: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
    macd_price_std_window: int = 63
    macd_normalized_std_window: int = 252
    winsor_halflife: int = 252
    winsor_multiple: float = 5.0
    trading_days: int = 252

    # The appendix defines volatility with normalized exponential weights and
    # a population weighted variance.  In pandas this is adjust=True/bias=True.
    vol_ewm_adjust: bool = True
    vol_ewm_ignore_na: bool = True
    vol_min_periods: int = 20
    vol_bias: bool = True

    # MACD is a recursive EMA with alpha=1/J, so it intentionally has a
    # separate setting from the appendix volatility estimator.
    macd_ewm_adjust: bool = False
    macd_ewm_ignore_na: bool = True

    # The paper gives the winsor half-life and band width, but not pandas'
    # initialization/bias conventions.  Keep those assumptions explicit.
    winsor_ewm_adjust: bool = False
    winsor_ewm_ignore_na: bool = True
    winsor_bias: bool = False
    multi_period_return_method: str = "compound_observed"
    rolling_calendar: str = "asset_observations"
    rolling_ddof: int = 1
    macd_ewm_min_periods: int = 1
    macd_price_std_min_periods: int = 63
    macd_normalized_std_min_periods: int = 252
    winsor_min_periods: int = 20
    winsor_include_current: bool = True
    mask_features_to_return_availability: bool = True

    @property
    def feature_names(self) -> tuple[str, ...]:
        scaled = tuple(f"ret_{h}d_scaled" for h in self.momentum_horizons)
        macd = tuple(f"macd_{s}_{l}" for s, l in self.macd_windows)
        return scaled + macd


@dataclass(frozen=True)
class GraphConfig:
    lookbacks: tuple[int, ...] = (252, 504, 756, 1008, 1260)
    parameter_grid: tuple[float, ...] = (
        0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,
        0.1, 0.5, 1.0, 5.0, 10.0,
    )
    solver: str | None = None
    edge_tolerance: float = 1e-8
    # WRDS/LSEG markets have unequal holiday calendars. For graph history only,
    # carry the last already-known feature over short local-market closures.
    calendar_policy: str = "causal_ffill"
    feature_ffill_limit: int = 5
    require_current_unfilled_features: bool = True
    prototype_alpha: float = 0.1
    prototype_beta: float = 0.1
    prototype_solver: str = "CLARABEL"
    prototype_date: str = "2022-12-30"
    dynamic_prototype_dates: int = 60
    dynamic_prototype_train_fraction: float = 0.70
    validation_fraction: float = 0.10
    validation_metric: str = "forecast_correlation"
    # The paper does not state how densely dates inside the 10% validation
    # window are evaluated. Eight evenly spaced dates make the disclosed
    # 11x11 grid computationally auditable on a laptop.
    validation_max_dates: int = 8
    validation_internal_train_fraction: float = 0.60
    validation_min_days: int = 8
    production_reference_interval: int = 60


@dataclass(frozen=True)
class PortfolioConfig:
    annual_asset_vol_target: float = 0.15
    trading_days: int = 252
    portfolio_vol_span: int = 60
    portfolio_vol_min_periods: int = 20
    portfolio_vol_ewm_adjust: bool = False
    portfolio_vol_leverage_cap: float | None = None


@dataclass(frozen=True)
class ReplicationConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def load_replication_config(path: Path = CONFIG_DIR / "config.yaml") -> ReplicationConfig:
    """Load reproducible parameters while keeping all paths project-relative."""
    if not path.exists():
        return ReplicationConfig()
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    feature_values = dict(payload.get("features", {}))
    if "momentum_horizons" in feature_values:
        feature_values["momentum_horizons"] = tuple(feature_values["momentum_horizons"])
    if "macd_windows" in feature_values:
        feature_values["macd_windows"] = tuple(
            tuple(window) for window in feature_values["macd_windows"]
        )
    graph_values = dict(payload.get("graph", {}))
    if "lookbacks" in graph_values:
        graph_values["lookbacks"] = tuple(graph_values["lookbacks"])
    if "parameter_grid" in graph_values:
        graph_values["parameter_grid"] = tuple(graph_values["parameter_grid"])
    return ReplicationConfig(
        experiment=ExperimentConfig(**payload.get("experiment", {})),
        data=DataConfig(**payload.get("data", {})),
        features=FeatureConfig(**feature_values),
        graph=GraphConfig(**graph_values),
        portfolio=PortfolioConfig(**payload.get("portfolio", {})),
    )


DEFAULT_CONFIG = load_replication_config()


# These are intentionally visible and should be revisited if author code or a
# more precise paper specification becomes available.
UNDOCUMENTED_CHOICES = {
    "return_convention": "simple arithmetic returns from the locked data pipeline",
    "volatility_ewm": "paper appendix: normalized weights (adjust=True) and population variance (bias=True)",
    "macd_ewm": "recursive EMA (adjust=False) with alpha=1/J",
    "ewm_ignore_na": "True; missing local-market dates do not advance decay",
    "vol_min_periods": "20 valid observations",
    "rolling_calendar": "last N valid observations for each asset",
    "multi_period_return": "compound the last N simple returns for each asset",
    "rolling_ddof": "1 (sample standard deviation)",
    "winsor_timing": "bounds at t include the feature observed at t",
    "graph_calendar": "causal feature carry-forward for at most 5 union-calendar rows",
    "graph_validation_metric": "pooled validation asset-day forecast correlation (paper is silent)",
    "graph_solver": "CLARABEL/SCS plus validated warm-start L-BFGS; MOSEK is unavailable",
    "portfolio_vol_rescaling": "UNRESOLVED; not used in Stage A",
}
