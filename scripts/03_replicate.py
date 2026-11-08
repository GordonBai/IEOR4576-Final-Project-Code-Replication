"""Rebuild the main paper replication in the documented order."""

import argparse

from network_momentum.backtest import (
    run_first_walkforward_block,
    run_full_backtest,
    run_graph_grid_searches,
)
from network_momentum.config import DEFAULT_CONFIG
from network_momentum.paper_analysis import (
    save_main_paper_outputs,
    save_table4_regression_inference,
)
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()
    run_first_walkforward_block(save=True)
    run_full_backtest(save=True)
    artifacts = DEFAULT_CONFIG.paths.output_dir
    replication = DEFAULT_CONFIG.paths.replication_dir
    save_main_paper_outputs(
        pd.read_parquet(artifacts / "results" / "full_fixed_raw_returns.parquet"),
        pd.read_parquet(artifacts / "results" / "full_fixed_vol_scaled_returns.parquet"),
        pd.read_parquet(artifacts / "signals" / "full_fixed_asset_signals.parquet"),
        replication,
    )
    save_table4_regression_inference(
        pd.read_csv(artifacts / "results" / "full_fixed_regression_inference.csv"),
        replication,
    )
    if not args.skip_grid:
        run_graph_grid_searches(save=True)
    print("Main replication PASS")


if __name__ == "__main__":
    main()
