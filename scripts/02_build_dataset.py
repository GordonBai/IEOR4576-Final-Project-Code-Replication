"""Build and validate the model-ready feature dataset from locked processed data."""

from network_momentum.backtest import run_stage_a


def main() -> None:
    _, reports = run_stage_a(save=True)
    print({name: report["status"] for name, report in reports.items()})


if __name__ == "__main__":
    main()
