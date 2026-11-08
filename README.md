# Network Momentum across Asset Classes — replication

This repository provides an auditable replication of Pu, Roberts, Dong and
Zohren (2023). It covers the locked 64-market data interface, feature
construction, dynamic graph learning, network momentum, pooled regressions,
six trading strategies, and five out-of-sample blocks from 2000–2022.

The headline run is a documented fixed-parameter approximation with
`alpha=beta=0.1`. The original paper used MOSEK and does not disclose every
validation convention; this implementation uses CLARABEL with recorded SCS
fallbacks and exposes all such choices in `configs/config.yaml`.

## Reproduce the project

From the repository root:

```bash
make install
make data       # validate existing locked data; does not download it again
make features   # rebuild and validate Stage A / U_t
make replicate  # rebuild the main walk-forward results and paper outputs
make test
```

`make all` runs those four stages in order. The graph grid search is the slow
part of `make replicate`; use `python scripts/03_replicate.py --skip-grid` for
the already-specified fixed-parameter result only.

The scripts are the official reproducible entry points:

1. `scripts/01_get_data.py` checks the 58 preserved raw batches and the four
   authoritative processed inputs. It intentionally performs no download.
2. `scripts/02_build_dataset.py` reconstructs volatility, the eight features,
   causal winsorization and validated `U_t`.
3. `scripts/03_replicate.py` runs the walk-forward replication and saves the
   paper tables and figures.
4. `scripts/04_extend.py` runs lightweight graph/model prototypes separately
   from the reported replication.

## Repository map

```text
.
├── configs/config.yaml
├── data/
│   ├── raw/                 # untouched downloaded batches
│   ├── interim/             # cleaned contracts and continuous futures
│   └── processed/           # locked 64-market model inputs
├── src/network_momentum/    # importable implementation
├── scripts/                 # ordered, thin execution entry points
├── notebooks/               # inspection/exploration, never result authority
├── results/
│   ├── replication/
│   │   ├── tables/
│   │   ├── figures/
│   │   └── artifacts/       # intermediate parquet/JSON/CSV audit trail
│   └── extension/
├── tests/
└── report/
```

The importable package uses project-relative paths only. `data/` and
`results/` are generated/local and excluded from version control.

## Authoritative model inputs

- `data/processed/paper64_returns_v3.parquet`
- `data/processed/paper64_availability_v3.parquet`
- `data/processed/paper64_prices_v3.parquet`
- `data/processed/paper64_model_data_v3.parquet`

The model pipeline never rewrites these files and never silently fills missing
returns or prices. The original data-construction notebook is retained only as
historical provenance under `notebooks/archive/`; it is not part of the
reproduction command chain.

## Current headline result

| Strategy | Raw Sharpe | 15%-scaled Sharpe | 15%-scaled return |
|---|---:|---:|---:|
| Long Only | 0.936 | 0.983 | 15.23% |
| MACD | 0.789 | 0.872 | 13.70% |
| LinReg | 1.363 | 1.442 | 22.49% |
| GMOM | 1.430 | 1.499 | 23.23% |
| RegCombo | **1.517** | **1.583** | **24.64%** |
| SignCombo | 1.452 | 1.545 | 24.04% |

The paper reports GMOM raw Sharpe 1.363, scaled Sharpe 1.511 and scaled annual
return 22.2%. Closeness is useful validation, but does not imply identical
vendor data or undocumented conventions.

The full robustness pass is also complete: Table 2, Table 3, daily topology
(Figure 6), asset-class ablations (Figures 7–8) and lookback robustness
(Figure 10) are saved under `results/replication/`. The graph checkpoint makes
these results reloadable without repeating the multi-hour graph calculation.

For methodology, data flow, timing rules and artifact definitions, see
[`report/ARCHITECTURE.md`](report/ARCHITECTURE.md). For the exact
completion and remaining fidelity boundaries, see
[`report/REPLICATION_STATUS.md`](report/REPLICATION_STATUS.md).
