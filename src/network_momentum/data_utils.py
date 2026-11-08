"""Load and validate the locked model inputs without rebuilding the data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig, PathConfig


@dataclass(frozen=True)
class ModelInputs:
    returns: pd.DataFrame
    availability: pd.DataFrame
    prices: pd.DataFrame


def _read_wide_panel(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {name} file: {path}")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex; got {type(frame.index).__name__}")
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    frame.columns = pd.Index(frame.columns.astype(str), name="asset")
    return frame


def load_model_inputs(paths: PathConfig) -> ModelInputs:
    """Load authoritative returns/availability and supporting prices.

    Returns are never recomputed here: the upstream pipeline's block-aware
    missingness is part of the locked input contract.
    """
    returns = _read_wide_panel(paths.returns_path, "returns").astype(float)
    availability = _read_wide_panel(paths.availability_path, "availability").astype(bool)
    prices = _read_wide_panel(paths.prices_path, "prices").astype(float)
    return ModelInputs(returns=returns, availability=availability, prices=prices)


def _finite_nonmissing(frame: pd.DataFrame) -> bool:
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    observed = values[~np.isnan(values)]
    return bool(np.isfinite(observed).all())


def validate_model_inputs(inputs: ModelInputs, config: DataConfig) -> dict[str, Any]:
    returns = inputs.returns
    availability = inputs.availability
    prices = inputs.prices

    same_return_availability_axes = (
        returns.index.equals(availability.index)
        and returns.columns.equals(availability.columns)
    )
    same_return_price_axes = (
        returns.index.equals(prices.index)
        and returns.columns.equals(prices.columns)
    )
    availability_matches = same_return_availability_axes and bool(
        availability.equals(returns.notna())
    )

    report: dict[str, Any] = {
        "authoritative_returns": str(config.return_convention),
        "shape": [int(returns.shape[0]), int(returns.shape[1])],
        "asset_count": int(returns.shape[1]),
        "date_start": returns.index.min().isoformat(),
        "date_end": returns.index.max().isoformat(),
        "date_index_type": type(returns.index).__name__,
        "dates_sorted": bool(returns.index.is_monotonic_increasing),
        "dates_unique": bool(returns.index.is_unique),
        "columns_unique": bool(returns.columns.is_unique),
        "finite_nonmissing_returns": _finite_nonmissing(returns),
        "returns_availability_axes_equal": bool(same_return_availability_axes),
        "availability_equals_returns_notna": bool(availability_matches),
        "returns_prices_axes_equal": bool(same_return_price_axes),
        "return_observations": int(returns.notna().sum().sum()),
        "price_observations": int(prices.notna().sum().sum()),
        "min_return_observations_per_asset": int(returns.notna().sum().min()),
        "max_return_observations_per_asset": int(returns.notna().sum().max()),
        "first_valid_return_by_asset": {
            str(asset): (date.isoformat() if date is not None else None)
            for asset, date in returns.apply(pd.Series.first_valid_index).items()
        },
        "last_valid_return_by_asset": {
            str(asset): (date.isoformat() if date is not None else None)
            for asset, date in returns.apply(pd.Series.last_valid_index).items()
        },
    }

    failures: list[str] = []
    if returns.shape[1] != config.expected_assets:
        failures.append(f"expected {config.expected_assets} assets, got {returns.shape[1]}")
    if config.require_sorted_unique_dates and not (
        report["dates_sorted"] and report["dates_unique"]
    ):
        failures.append("return dates are not sorted and unique")
    if not report["columns_unique"]:
        failures.append("return asset identifiers are not unique")
    if config.require_finite_nonmissing_returns and not report["finite_nonmissing_returns"]:
        failures.append("returns contain non-finite observed values")
    if config.require_availability_exact_match and not availability_matches:
        failures.append("availability does not exactly equal returns.notna()")
    if not same_return_price_axes:
        failures.append("supporting prices do not share the return panel axes")
    if failures:
        raise AssertionError("Input contract failed: " + "; ".join(failures))

    report["status"] = "PASS"
    return report


def audit_long_model_data(paths: PathConfig, inputs: ModelInputs) -> dict[str, Any]:
    """Verify that the long audit table reproduces the locked wide inputs."""
    if not paths.model_data_path.exists():
        return {"status": "SKIPPED", "reason": "long model-data file not found"}

    model = pd.read_parquet(
        paths.model_data_path,
        columns=["date_", "paper_ticker", "continuous_price", "daily_return"],
    )
    duplicate_keys = int(model.duplicated(["date_", "paper_ticker"]).sum())
    if duplicate_keys:
        raise AssertionError(f"model_data has {duplicate_keys} duplicate date/asset keys")

    long_returns = model.pivot(index="date_", columns="paper_ticker", values="daily_return")
    long_prices = model.pivot(index="date_", columns="paper_ticker", values="continuous_price")
    long_returns = long_returns.reindex(index=inputs.returns.index, columns=inputs.returns.columns).astype(float)
    long_prices = long_prices.reindex(index=inputs.prices.index, columns=inputs.prices.columns).astype(float)

    return_diff = (long_returns - inputs.returns).abs()
    price_diff = (long_prices - inputs.prices).abs()
    return_na_match = bool(long_returns.isna().equals(inputs.returns.isna()))
    price_na_match = bool(long_prices.isna().equals(inputs.prices.isna()))
    max_return_diff = float(return_diff.max().max()) if return_diff.notna().any().any() else 0.0
    max_price_diff = float(price_diff.max().max()) if price_diff.notna().any().any() else 0.0

    status = (
        "PASS"
        if return_na_match and price_na_match and max_return_diff <= 1e-14 and max_price_diff <= 1e-12
        else "FAIL"
    )
    if status == "FAIL":
        raise AssertionError(
            "Long model-data audit failed: "
            f"return_na={return_na_match}, price_na={price_na_match}, "
            f"max_return_diff={max_return_diff}, max_price_diff={max_price_diff}"
        )
    return {
        "status": status,
        "rows": int(len(model)),
        "duplicate_date_asset_keys": duplicate_keys,
        "return_missingness_equal": return_na_match,
        "price_missingness_equal": price_na_match,
        "max_abs_return_difference": max_return_diff,
        "max_abs_price_difference": max_price_diff,
    }
