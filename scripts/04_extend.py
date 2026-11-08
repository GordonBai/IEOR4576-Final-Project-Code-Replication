"""Run the lightweight graph/model extension prototypes."""

from network_momentum.backtest import run_stage_b_prototype, run_stage_cd_prototype


def main() -> None:
    run_stage_b_prototype(save=True)
    run_stage_cd_prototype(save=True)
    print("Extension prototypes PASS")


if __name__ == "__main__":
    main()
