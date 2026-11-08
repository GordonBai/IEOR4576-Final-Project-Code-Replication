# Network Momentum replication architecture

## Design rule

Reported results come from the importable Python package and ordered scripts,
not from notebook state. Notebooks display QA, plots and saved results; they do
not contain the authoritative implementation.

```text
configs/config.yaml
        |
data/processed (locked 64-market inputs)
        |
        v
Module 0  input and availability QA
        |
Module 1  60-day EWM volatility
        |
Module 2  five scaled returns + three normalized MACDs
        |
Module 3  causal EW winsorization -> validated U_t
        |
five dynamic lookback graphs -> graph ensemble
        |
network features -> pooled OLS -> positions
        |
five OOS blocks -> portfolio returns and metrics
        |
results/replication/{tables,figures,artifacts}
```

## Directory responsibilities

```text
Final Project Code Replication/
├── README.md
├── pyproject.toml
├── Makefile
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                 # 58 untouched WRDS/LSEG batch files
│   ├── interim/             # contract cleaning, rolls and continuous futures
│   └── processed/           # final paper64 inputs; model-stage read-only
├── src/
│   └── network_momentum/
│       ├── paths.py         # paths derived from repository root
│       ├── config.py        # typed config plus YAML loading
│       ├── data.py          # public data facade
│       ├── data_utils.py    # loading and schema/lineage QA
│       ├── features.py      # volatility, momentum, MACD and winsorization
│       ├── graph_learning.py
│       ├── network_features.py
│       ├── models.py        # public graph/regression facade
│       ├── regression.py
│       ├── portfolio.py
│       ├── metrics.py
│       ├── evaluation.py    # public performance/validation facade
│       ├── plotting.py      # public plotting facade
│       ├── paper_analysis.py
│       ├── robustness.py
│       └── backtest.py      # orchestration and execution guards
├── scripts/                 # thin entry points, run 01 -> 04
├── notebooks/               # research display and saved-result inspection
├── results/
│   ├── replication/
│   │   ├── tables/
│   │   ├── figures/
│   │   └── artifacts/{features,graphs,signals,results,qa,...}
│   └── extension/{tables,figures}
├── tests/
└── report/
```

## Notebook versus package code

- `src/network_momentum/` contains every transformation used by a reported
  result. Functions accept explicit data/config and can be tested in isolation.
- `scripts/` determines execution order and is the reproducibility interface.
- `notebooks/01_replication_validation.ipynb` inspects Stage A and `U_t`.
- `notebooks/02_graph_backtest_validation.ipynb` inspects graph and OOS results
  without recomputing the expensive path.
- `notebooks/03_complete_paper_results.ipynb` displays saved paper tables and
  figures.
- `notebooks/archive/` is provenance only and may contain historical paths or
  download credentials. It is never called by `make all`.

## Input contract

| Role | Path | Rule |
|---|---|---|
| Returns | `data/processed/paper64_returns_v3.parquet` | authoritative; never recomputed by model stages |
| Availability | `data/processed/paper64_availability_v3.parquet` | must equal `returns.notna()` exactly |
| Prices | `data/processed/paper64_prices_v3.parquet` | MACD support; never forward-filled |
| Long audit source | `data/processed/paper64_model_data_v3.parquet` | must pivot back to locked panels |

Module 0 requires 64 assets, sorted unique dates, identical axes, finite
observed returns and exact availability. The long audit table is independently
pivoted and compared with the wide inputs.

## Stage A and auditable outputs

The daily volatility is a 60-span EWM standard deviation. Annual volatility is
daily volatility times `sqrt(252)`. For horizons `1, 21, 63, 126, 252`, scaled
returns divide the compounded horizon return by current daily volatility times
the square root of the horizon.

The three MACDs use `(8,24)`, `(16,48)` and `(32,96)`. Each EMA difference is
normalized by a 63-observation price standard deviation and then by its own
252-observation standard deviation. Each feature is clipped to its causal
252-day-half-life EWM mean plus/minus five EWM standard deviations.

Stage A writes to `results/replication/artifacts/`:

| Location | Contents |
|---|---|
| `features/daily_volatility.parquet` | daily sigma |
| `features/annualized_volatility.parquet` | annualized sigma |
| `features/vol_scaled_return_features.parquet` | five scaled-return features |
| `features/macd_features.parquet` | three normalized MACDs |
| `features/U_raw.parquet` | eight features before winsorization |
| `features/winsor_*_bounds.parquet` | causal clipping bounds |
| `features/U_winsorized.parquet` | validated feature tensor `U_t` |
| `qa/stage_a_manifest.json` | inputs, full config, assumptions and outputs |
| `qa/causality_qa.json` | full-versus-truncated history invariance test |

At any date:

```python
U_t = U_winsorized.xs(pd.Timestamp("2020-01-02"), level="date").dropna()
```

Rows are assets and columns are the eight features.

## Look-ahead rules

1. A feature stamped `t` may use observations through `t`, never after `t`.
2. No missing return or price is replaced with a future value or zero.
3. Winsor bounds are causal EWM statistics, not full-sample moments.
4. For graph history only, an already-known feature may be carried forward for
   at most five union-calendar rows; current trading features are never filled.
5. A signal at `t` is evaluated on the next trading return.
6. Hyperparameters and coefficients use training/validation history only.
7. Each training label's realization date must precede the first OOS test date.
8. The causality QA truncates the final 20% of history and requires all earlier
   values and missingness flags to remain unchanged.

## Configured choices not fully specified by the paper

All are in `configs/config.yaml` rather than embedded in method code:

- normalized EWM volatility weights (`adjust=True`) and population weighted
  variance (`bias=True`);
- recursive MACD EMA with `alpha=1/J`;
- 20 valid observations before volatility and winsorization begin;
- per-asset observation calendars for horizons and rolling standard deviations;
- winsor bounds include the known observation at `t`;
- pooled validation asset-day forecast correlation for the latest-10% grid;
- a strictly lagged 60-span portfolio EWM volatility estimator;
- CLARABEL with recorded SCS fallback in place of unavailable MOSEK.

## Walk-forward schedule and outputs

```text
train through 1999 -> test 2000–2004
train through 2004 -> test 2005–2009
train through 2009 -> test 2010–2014
train through 2014 -> test 2015–2019
train through 2019 -> test 2020–2022
```

Final paper-facing CSVs and PNGs are in
`results/replication/{tables,figures}`. Reusable intermediate artifacts remain
under `results/replication/artifacts/`, including coefficients, daily strategy
returns, signals, graph diagnostics, parameter-grid scores and QA manifests.

The completed deliverable includes the six-strategy headline results, Tables
1–4, Figures 1–10, grid search, selected-date graphs, full daily topology,
asset-class ablations and individual-lookback backtests. The expensive graph
pass is persisted under `results/replication/artifacts/robustness_checkpoint/`
so plotting and downstream regressions can be resumed without rebuilding the
full graph history. See `REPLICATION_STATUS.md` for remaining fidelity limits.
