# Network Momentum (Pu, Roberts, Dong & Zohren, 2023) — Replication Handoff

## 0. Purpose of this file

This file is a handoff for another ChatGPT / coding session.

**Goal:** faithfully implement the original methodology in:

> Xingyue Pu, Stephen Roberts, Xiaowen Dong, Stefan Zohren (2023),  
> *Network Momentum across Asset Classes*  
> arXiv: https://arxiv.org/abs/2308.11294

The user **already has the futures data**.  
Do **not** restart data acquisition or WRDS mapping unless the user explicitly asks.

The implementation should be **modular and testable**. Do not try to write the full paper replication in one giant notebook cell.

The long-term project is:

1. faithfully replicate the original paper;
2. test out-of-sample in 2023–2026;
3. develop a modern extension.

**For now, do only step 1.**

---

# 1. Core mechanism to reproduce

The implementation must follow this exact conceptual pipeline:

\[
\text{futures prices}
\rightarrow
\text{returns}
\rightarrow
\text{8 individual momentum features}
\rightarrow
\text{dynamic graph}
\rightarrow
\text{network momentum features}
\rightarrow
\text{OLS forecast}
\rightarrow
\text{long/short sign}
\rightarrow
\text{volatility-scaled futures exposure}
\rightarrow
\text{portfolio return}
\rightarrow
\text{Sharpe ratio}
\]

Use the notation:

- \(p_{i,t}\): continuous futures price of asset \(i\) at date \(t\)
- \(r_{i,t:t+1}\): next-day return
- \(\sigma_{i,t}\): volatility estimated using information available at \(t\)
- \(\mathbf u_{i,t}\in\mathbb R^8\): individual momentum features
- \(\mathbf A_t\): learned adjacency matrix
- \(\tilde{\mathbf A}_t\): normalized adjacency matrix
- \(\tilde{\mathbf u}_{i,t}\in\mathbb R^8\): network momentum features
- \(\hat y_{i,t}\): predicted next-day volatility-scaled return
- \(x_{i,t}\): trading direction
- \(r^{portfolio}_{t:t+1}\): realized strategy return

---

# 2. Important implementation rule: causal timing

Everything used to form the position \(x_{i,t}\) must be known by time \(t\).

The intended timeline is:

```text
history through t
    ↓
prices / returns through t
    ↓
momentum features U_t
    ↓
graph A_t using only historical features through t
    ↓
network features at t
    ↓
OLS prediction at t
    ↓
position x_t
    ↓
realized return from t to t+1
    ↓
P&L
```

The target:

\[
y_{i,t}^{target}
=
\frac{r_{i,t:t+1}}{\sigma_{i,t}}
\]

must **never** leak into any feature, graph, hyperparameter, or coefficient used to create \(x_{i,t}\).

Whenever there is uncertainty about alignment, explicitly print dates and inspect:

```text
feature date
signal date
position date
realized return date
```

Do not continue until alignment is verified.

---

# 3. Recommended project architecture

Do **not** put the whole replication into one notebook.

Recommended structure:

```text
network_momentum/
│
├── data/
│   └── existing_user_data.*        # already prepared; do not rebuild
│
├── src/
│   ├── config.py                   # horizons, graph lookbacks, grids, dates
│   ├── data_utils.py               # load + alignment only
│   ├── features.py                 # returns, vol, 8 momentum features, winsorization
│   ├── graph_learning.py           # V_t, graph optimization, ensemble, normalization
│   ├── network_features.py         # A_t @ U_t propagation
│   ├── regression.py               # pooled OLS and prediction
│   ├── portfolio.py                # sign, vol scaling, P&L, turnover
│   ├── metrics.py                  # Sharpe, return, vol, drawdown, etc.
│   └── backtest.py                 # walk-forward orchestration
│
├── notebooks/
│   └── 01_replication_validation.ipynb
│
├── outputs/
│   ├── features/
│   ├── graphs/
│   ├── signals/
│   └── results/
│
└── README.md
```

The notebook should be used for:

- running modules;
- inspecting intermediate outputs;
- plotting;
- comparing to paper numbers;
- debugging.

The actual logic should live in `.py` modules.

---

# 4. Module 0 — Validate the existing data interface

## Objective

Do not change the data yet. First identify what the user already has.

The coding chat should inspect:

- date column / DatetimeIndex;
- asset identifiers;
- price column(s);
- wide vs long format;
- first and last valid date for each market;
- missing observations;
- number of assets;
- whether the data are already continuous/back-adjusted;
- whether returns already exist.

Preferred internal representation:

```python
prices
# index = trading dates
# columns = assets
# values = continuous futures prices
```

Example:

```text
date         CL       GC       ES       TY       EUR ...
1999-01-04   ...
1999-01-05   ...
```

### Output checkpoint

Before any signal construction, print:

```python
prices.shape
prices.index.min()
prices.index.max()
prices.notna().sum().sort_values()
```

Also plot 3–5 representative markets.

Do not silently forward-fill missing futures prices across dates where a market did not trade.

---

# 5. Module 1 — Daily returns and volatility

## 5.1 Daily returns

Construct daily returns:

\[
r_{i,t}
\]

from the existing continuous price series.

The paper does not spell out the simple-vs-log return convention in the main text.

Therefore:

- isolate this calculation in one function;
- default to **simple percentage returns** unless the user's existing pipeline or authors' code establishes otherwise;
- do not bury the convention inside later functions.

Suggested function:

```python
compute_returns(prices) -> returns
```

---

## 5.2 60-day exponentially weighted volatility

The paper uses an exponentially weighted moving standard deviation of daily returns with **span = 60**:

\[
\alpha=\frac{2}{60+1}.
\]

This volatility estimate is used repeatedly.

Maintain two explicit versions if needed:

```python
daily_vol
annualized_vol = daily_vol * sqrt(252)
```

Do not mix them.

The momentum feature equation uses daily \(\sigma_{i,t}\).

The portfolio sizing equation refers to annualized realized volatility and a 15% annual target.

---

# 6. Module 2 — Construct the 8 individual momentum features

For each asset \(i\) and date \(t\), construct:

\[
\mathbf u_{i,t}
=
[u^{(1)}_{i,t},\ldots,u^{(8)}_{i,t}]^\top
\]

with exactly **8 features**.

---

## 6.1 Five volatility-scaled return features

Use horizons:

\[
\Delta\in\{1,21,63,126,252\}.
\]

For each horizon:

\[
u^{(\Delta)}_{i,t}
=
\frac{r_{i,t-\Delta:t}}
{\sigma_{i,t}\sqrt{\Delta}}.
\]

The five features correspond approximately to:

```text
1 day
1 month
3 months
6 months
1 year
```

Important:

- \(\sigma_{i,t}\) is the 60-day EWM daily volatility;
- only information through \(t\) is allowed.

The exact definition of the multi-period return \(r_{t-\Delta:t}\) should be isolated in a function so it can be replaced if the source convention is later confirmed.

---

## 6.2 Three normalized MACD features

Use:

\[
(S,L)\in
\{(8,24),(16,48),(32,96)\}.
\]

Define the exponentially weighted moving price:

\[
m(i,t,J)
\]

with smoothing:

\[
\alpha=\frac{1}{J}.
\]

Then:

\[
MACD(i,t,S,L)
=
m(i,t,S)-m(i,t,L).
\]

Normalize by the 63-day rolling standard deviation of prices:

\[
MACD_{norm}(i,t,S,L)
=
\frac{MACD(i,t,S,L)}
{\operatorname{std}(p_{i,t-63:t})}.
\]

Then normalize again by the 252-day rolling standard deviation of normalized MACD:

\[
y_{i,t}(S,L)
=
\frac{MACD_{norm}(i,t,S,L)}
{\operatorname{std}(MACD_{norm}(i,t-252:t,S,L))}.
\]

These three \(y_{i,t}(S,L)\) values are features 6–8.

Suggested implementation:

```python
compute_vol_scaled_return_features(...)
compute_macd_features(...)
build_feature_panel(...)
```

---

## 6.3 Feature matrix

At each date \(t\):

\[
\mathbf U_t
=
[\mathbf u_{1,t},\ldots,\mathbf u_{N_t,t}]^\top
\in\mathbb R^{N_t\times8}.
\]

Interpretation:

```text
rows    = assets
columns = 8 momentum features
```

---

# 7. Module 3 — Winsorization

The paper winsorizes the momentum features.

For each feature of each asset, compute an exponentially weighted:

- mean;
- standard deviation;

using **half-life = 252 days**.

Cap/floor the feature at:

\[
EWMA_t \pm 5\times EWMSTD_t.
\]

This must be causal.

Do not compute the mean/std using the full sample.

### Unit test

For selected dates, verify:

```python
lower_bound <= feature_winsorized <= upper_bound
```

and confirm the bounds use history only.

---

# 8. Module 4 — Dynamic graph learning

This is the central part of the paper.

At every date \(t\), build graph inputs from historical individual momentum features.

---

## 8.1 Graph lookback matrix

For a graph lookback \(\delta\), stack the past momentum matrices to create:

\[
\mathbf V_t
\in
\mathbb R^{N_t\times 8\delta}.
\]

Each row represents one asset.

Each row contains its 8 momentum features over the entire \(\delta\)-day lookback.

Use only assets that are consistently available over the relevant lookback window.

The paper uses five lookbacks:

\[
\delta
\in
\{252,504,756,1008,1260\}.
\]

These correspond to roughly 1–5 trading years.

---

## 8.2 Graph optimization

For each \(\delta\), estimate:

\[
\mathbf A_t^{(\delta)}
\in
\mathbb R^{N_t\times N_t}.
\]

The paper solves:

\[
\min_{\mathbf A_t}
\operatorname{tr}
\left(
\mathbf V_t^\top
(\mathbf D_t-\mathbf A_t)
\mathbf V_t
\right)
-
\alpha\mathbf 1^\top
\log(\mathbf A_t\mathbf1)
+
\beta\|\mathbf A_t\|_F^2
\]

where:

\[
D_{ii,t}
=
\sum_j A_{ij,t}.
\]

Constraints:

\[
A_{ij,t}\ge0,
\]

\[
\mathbf A_t=\mathbf A_t^\top,
\]

\[
A_{ii,t}=0.
\]

Interpretation:

- first term: connect assets whose historical momentum-feature paths are similar;
- log-degree term: prevent isolated nodes;
- Frobenius penalty: regularize edge magnitudes;
- graph is undirected;
- graph weights are non-negative;
- no self edges.

Implementation should use **CVXPY**.

The paper reports MOSEK through CVXPY. If MOSEK is unavailable, do not silently substitute a solver and call the result exact replication. Explicitly record the substitute solver and compare numerical behavior.

Suggested function:

```python
learn_graph(V, alpha, beta, solver=...) -> A
```

---

## 8.3 Hyperparameter grid

Tune:

\[
\alpha,\beta
\in
\{
0.0001,
0.0005,
0.001,
0.005,
0.01,
0.05,
0.1,
0.5,
1,
5,
10
\}.
\]

The paper uses the **most recent 10% of the in-sample period as validation data** for graph hyperparameter grid search.

Important:

The coding chat must verify the exact model-selection criterion from the paper/code before hard-coding it.

Do not invent a validation metric if it is not explicitly established.

---

## 8.4 Five-graph ensemble

Learn five graphs using the five lookbacks:

\[
252,504,756,1008,1260.
\]

Average them:

\[
\bar{\mathbf A}_t
=
\frac{1}{5}
\sum_{k=1}^{5}
\mathbf A_t^{(k)}.
\]

This is not optional for the faithful main specification.

---

## 8.5 Graph normalization

Define:

\[
\bar D_{ii,t}
=
\sum_j \bar A_{ij,t}.
\]

Then:

\[
\tilde{\mathbf A}_t
=
\bar{\mathbf D}_t^{-1/2}
\bar{\mathbf A}_t
\bar{\mathbf D}_t^{-1/2}.
\]

This normalized adjacency matrix is what is used to propagate momentum information.

Suggested functions:

```python
ensemble_graphs(graph_list) -> A_bar
normalize_graph(A_bar) -> A_tilde
```

---

## 8.6 Graph checks

Every graph should satisfy approximately:

```python
A.shape == (N, N)
A >= 0
A ≈ A.T
diag(A) == 0
```

Also inspect:

- degree distribution;
- edge density;
- number of isolated nodes;
- largest weights;
- cross-asset-class edges.

Do not proceed if graph optimization routinely returns invalid or numerically unstable matrices.

---

# 9. Module 5 — Network momentum features

For asset \(i\):

\[
\tilde{\mathbf u}_{i,t}
=
\sum_{j\in\mathcal N_t(i)}
\tilde A_{ij,t}\mathbf u_{j,t}.
\]

Matrix form:

\[
\tilde{\mathbf U}_t
=
\tilde{\mathbf A}_t
\mathbf U_t.
\]

This should produce:

\[
\tilde{\mathbf U}_t
\in
\mathbb R^{N_t\times8}.
\]

This is one of the most useful implementation checks because the entire graph propagation can be performed with a simple matrix multiplication after the graph is learned.

Important distinction:

```text
U_t        = each asset's own momentum features
A_tilde U_t = momentum information propagated from connected assets
```

Do not accidentally feed the original \(U_t\) into the GMOM regression. That would reproduce the individual-momentum LinReg benchmark instead.

---

# 10. Module 6 — Prediction target and pooled OLS

## 10.1 Target

The training target is:

\[
y^{target}_{i,t}
=
\frac{r_{i,t:t+1}}
{\sigma_{i,t}}.
\]

This is the **future one-day volatility-scaled return**.

It is allowed as a training label only after date \(t+1\) is part of the historical training sample.

It must never appear in the features at \(t\).

---

## 10.2 GMOM regression

Fit one pooled OLS model across all asset-date observations in the in-sample period:

\[
y^{target}_{i,t}
=
\tilde{\mathbf u}_{i,t}^\top\boldsymbol\beta+b+\epsilon_{i,t}.
\]

There are:

- 8 coefficients;
- 1 intercept;
- one shared model across all assets.

Do **not** fit 64 separate asset regressions.

Suggested interface:

```python
fit_gmom_ols(X_network, y_target) -> model
predict_gmom(model, X_network_t) -> y_hat_t
```

---

# 11. Module 7 — Five-year walk-forward training schedule

The paper retrains every five years.

Main schedule:

```text
Train: 1990–1999  → Test: 2000–2004
Train: 1990–2004  → Test: 2005–2009
Train: 1990–2009  → Test: 2010–2014
Train: 1990–2014  → Test: 2015–2019
Train: 1990–2019  → Test: 2020–2022
```

Training data are expanding-window.

For each recalibration:

1. use all historical preprocessed data up to the training cutoff;
2. reserve the most recent 10% of in-sample observations for graph hyperparameter validation;
3. tune graph hyperparameters;
4. fit OLS using historical data only;
5. freeze the trained configuration;
6. generate truly out-of-sample signals for the next test block.

Do not refit OLS every day if the objective is faithful replication of the paper's reported setup.

---

# 12. Module 8 — Convert forecast into futures direction

For each asset/date:

\[
\hat y_{i,t}
=
\tilde{\mathbf u}_{i,t}^\top\hat{\boldsymbol\beta}+\hat b.
\]

The trading signal is:

\[
x_{i,t}
=
\operatorname{sign}(\hat y_{i,t}).
\]

Thus:

```text
positive forecast -> +1 long
negative forecast -> -1 short
```

The magnitude of the OLS forecast does **not** directly determine position size in the main GMOM portfolio.

This is important.

A forecast of `+0.02` and `+2.0` both imply `x = +1`.

---

# 13. Module 9 — Asset-level volatility scaling and daily portfolio return

The paper defines:

\[
r^{portfolio}_{t:t+1}
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
x_{i,t}
\frac{\sigma_{tgt}}{\sigma_{i,t}}
r_{i,t:t+1}.
\]

Use:

\[
\sigma_{tgt}=0.15.
\]

For this equation, \(\sigma_{i,t}\) is the asset's **annualized** realized volatility derived from the 60-day-span EWM standard deviation.

Therefore implement explicitly:

```python
daily_vol = ...
annual_vol = daily_vol * np.sqrt(252)

risk_multiplier = 0.15 / annual_vol
```

Then:

```python
asset_strategy_return =
    position_sign * risk_multiplier * next_day_return

portfolio_return =
    mean(asset_strategy_returns across available assets)
```

Use the actual \(N_t\) of tradable/valid assets on date \(t\).

Do not replace missing strategy returns with zeros unless that is explicitly the intended exposure treatment.

---

# 14. Module 10 — Portfolio-level 15% volatility rescaling

Paper Table 1 reports two versions:

### Panel A
Raw strategy signal returns from the portfolio construction equation.

### Panel B
An **additional portfolio-level volatility scaling** to 15% annualized volatility.

This is distinct from the asset-level volatility scaling already inside:

\[
\frac{0.15}{\sigma_{i,t}}.
\]

Do not confuse these two layers.

The headline GMOM result around:

```text
annual return ≈ 22.2%
annual vol    ≈ 14.7%
Sharpe        ≈ 1.51
```

corresponds to the additionally portfolio-volatility-scaled result reported in Panel B.

The portfolio-level scaling itself must also be causal: use trailing historical portfolio volatility, not full-sample realized volatility.

If the exact portfolio-volatility estimator is not specified clearly enough in the paper, isolate it as a configurable component and document the assumption.

---

# 15. Module 11 — Evaluation metrics

At minimum compute:

```text
annualized mean return
annualized volatility
Sharpe ratio
downside deviation
maximum drawdown
maximum drawdown duration
Sortino ratio
Calmar ratio
hit rate
average profit / average loss
```

For daily returns:

\[
\text{Annual Return}
\approx
252\cdot \bar r
\]

\[
\text{Annual Vol}
=
\sqrt{252}\cdot std(r)
\]

\[
Sharpe
=
\frac{\text{Annual Return}}
{\text{Annual Vol}}.
\]

Keep arithmetic/compounding conventions explicit and use the same convention consistently when comparing to the paper.

---

# 16. Required benchmarks

Before claiming GMOM is correct, reproduce the simpler baselines.

## 16.1 Long Only

\[
x_{i,t}=1.
\]

Use the same asset-level volatility scaling.

---

## 16.2 Individual momentum linear regression (LinReg)

Use the original individual features:

\[
y_{i,t}
=
\mathbf u_{i,t}^\top\boldsymbol\beta+b.
\]

Everything else should be kept as comparable as possible.

This is a critical diagnostic:

```text
LinReg input = U_t
GMOM input   = A_tilde @ U_t
```

---

## 16.3 MACD baseline

Use the three normalized MACD features with:

\[
x_{i,t}
=
\frac13
\sum_{k=1}^3
\phi(y_{i,t}(S_k,L_k)),
\]

where:

\[
\phi(y)
=
\frac{y e^{-y^2/4}}{0.89}.
\]

This is a continuous position signal, unlike the sign of the regression forecast.

---

# 17. Transaction-cost extension after the core replication works

Do **not** start here.

Only after no-cost GMOM is validated, implement turnover:

\[
\zeta_{i,t}
=
\sigma_{tgt}
\left|
\frac{x_{i,t}}{\sigma_{i,t}}
-
\frac{x_{i,t-1}}{\sigma_{i,t-1}}
\right|.
\]

Cost-adjusted portfolio return:

\[
\tilde r^{portfolio}_{t:t+1}
=
\frac1{N_t}
\sum_i
\left[
x_{i,t}
\frac{\sigma_{tgt}}{\sigma_{i,t}}
r_{i,t:t+1}
-
c\zeta_{i,t}
\right].
\]

Paper pseudo-cost grid:

\[
c\in\{0,0.5,1,2,3,4,5\}\text{ bps}.
\]

---

# 18. Order of implementation

The next chat should work in this order and stop after each module for verification.

## Stage A — Data + features

1. Load existing prices.
2. Compute daily returns.
3. Compute 60-day EWM volatility.
4. Compute five volatility-scaled return features.
5. Compute three normalized MACD features.
6. Apply causal winsorization.
7. Build \(U_t\).

### Deliverable

A clean feature dataset with columns like:

```text
date
asset
ret_1d_scaled
ret_21d_scaled
ret_63d_scaled
ret_126d_scaled
ret_252d_scaled
macd_8_24
macd_16_48
macd_32_96
next_ret
next_ret_scaled
```

Do not move to graph learning until these are validated.

---

## Stage B — One-date graph prototype

Before running thousands of graph optimizations:

1. choose one historical date with sufficient data;
2. construct \(V_t\) for one \(\delta\);
3. solve one graph;
4. verify symmetry/non-negativity/no diagonal;
5. inspect edge weights.

Then test all five \(\delta\) values for the same date.

Then ensemble and normalize.

Only afterward automate through time.

---

## Stage C — Network feature construction

For the prototype date:

```python
U_tilde = A_tilde @ U_t
```

Verify dimensions and inspect one target asset manually:

```python
manual weighted sum
vs
matrix multiplication result
```

They should match.

---

## Stage D — OLS prototype

Construct a small historical training panel.

Fit pooled OLS:

```text
X = network features
y = next-day volatility-scaled return
```

Verify:

```python
X.shape[1] == 8
```

and that no future feature date enters the training sample.

---

## Stage E — One walk-forward block

Run only:

```text
Train 1990–1999
Test 2000–2004
```

(or the earliest feasible equivalent given the user's available history).

Inspect:

- OLS coefficients;
- signal distribution;
- long/short balance;
- asset-level risk multipliers;
- daily portfolio return;
- Sharpe.

Do not run the entire sample until the first block is correct.

---

## Stage F — Full original backtest

Run all five-year blocks.

Concatenate OOS returns only.

Then compare:

```text
Long Only
MACD
LinReg
GMOM
```

---

# 19. Minimum debugging assertions

Use assertions aggressively.

## Feature timing

For a sample row at date \(t\):

```python
assert feature_data_max_date <= t
assert target_return_date == next_trading_date(t)
```

## Graph

```python
assert np.all(A >= -tol)
assert np.allclose(A, A.T, atol=tol)
assert np.allclose(np.diag(A), 0, atol=tol)
```

## Network features

```python
assert U_tilde.shape == U_t.shape
```

## Model

```python
assert X_train.shape[1] == 8
assert len(beta) == 8
```

## Position

```python
assert set(np.unique(x[~np.isnan(x)])) <= {-1, 0, 1}
```

## Backtest separation

```python
assert train_end < test_start
```

No hyperparameter or coefficient may use information from the test block.

---

# 20. Biggest implementation mistakes to avoid

### Mistake 1 — Building one static network using the full sample

Wrong.

The network is dynamic and based on historical rolling information.

---

### Mistake 2 — Using future returns in graph learning

Wrong.

Graphs use historical momentum features only.

---

### Mistake 3 — Using raw correlations as the network

Wrong for faithful replication.

The paper uses a graph smoothness optimization problem.

---

### Mistake 4 — Feeding \(U_t\) into GMOM

Wrong.

GMOM regression input is:

\[
\tilde U_t=\tilde A_tU_t.
\]

---

### Mistake 5 — Using forecast magnitude as position size

Wrong for main GMOM.

Use:

\[
x_{i,t}=\operatorname{sign}(\hat y_{i,t}).
\]

---

### Mistake 6 — Dividing by volatility twice incorrectly

The regression target is volatility-scaled:

\[
r_{t:t+1}/\sigma_t.
\]

That is for **learning the direction relationship**.

Separately, realized futures exposure is scaled:

\[
0.15/\sigma_{i,t}.
\]

These are two different roles.

Do not remove the portfolio risk scaling just because the OLS label was volatility-scaled.

---

### Mistake 7 — Confusing daily and annualized volatility

For momentum features:

```text
daily EWM volatility
```

For the 15% portfolio exposure target:

```text
annualized EWM volatility
```

Keep variable names explicit.

---

### Mistake 8 — Full-sample winsorization

Wrong.

Winsorization statistics must be historical/causal.

---

### Mistake 9 — Daily refitting of OLS

Not the main paper setup.

The reported backtest recalibrates every five years.

---

### Mistake 10 — Comparing to the 1.51 Sharpe before portfolio-level vol scaling

The paper's headline 1.51 is the volatility-rescaled Panel B result.

First compare the raw-signal GMOM result, then the portfolio-scaled result.

---

# 21. Paper details that must NOT be guessed

If exact replication is the goal, flag these explicitly when encountered:

1. exact return convention if not already defined in the user's existing data pipeline;
2. precise behavior of EWM initialization / `adjust=True` vs `adjust=False`;
3. minimum periods for rolling statistics;
4. exact treatment of market holidays and unequal asset calendars;
5. numerical zero threshold used to define an edge;
6. exact graph hyperparameter validation selection criterion if not unambiguously stated;
7. exact portfolio-level volatility-rescaling estimator;
8. alternative solver behavior if MOSEK is unavailable.

Implement these behind configurable functions/parameters.

Do **not** hard-code an undocumented assumption deep inside the pipeline.

---

# 22. Suggested configuration

```python
MOMENTUM_HORIZONS = [1, 21, 63, 126, 252]

MACD_WINDOWS = [
    (8, 24),
    (16, 48),
    (32, 96),
]

VOL_SPAN = 60
WINSOR_HALFLIFE = 252
WINSOR_MULTIPLE = 5.0

GRAPH_LOOKBACKS = [
    252,
    504,
    756,
    1008,
    1260,
]

GRAPH_PARAM_GRID = [
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

ASSET_VOL_TARGET = 0.15
TRADING_DAYS = 252
```

---

# 23. Desired interaction style for the next ChatGPT

The user wants a **module-by-module replication**, not a giant final script.

For each module:

1. explain what is being implemented;
2. give the code;
3. tell the user exactly what to run locally in VS Code/Jupyter;
4. specify the expected shapes / diagnostic output;
5. wait for the user's actual output;
6. debug it;
7. only then move to the next module.

Do not skip directly from prices to the final Sharpe ratio.

The implementation should progressively prove:

```text
data correct
→ features correct
→ graph correct
→ network features correct
→ regression correct
→ timing correct
→ positions correct
→ P&L correct
→ performance correct
```

---

# 24. Definition of success

A faithful original-paper replication is successful only if:

- all features are constructed causally;
- the graph learning objective is implemented correctly;
- five lookback graphs are ensembled;
- graph normalization is correct;
- network features use \(\tilde A_tU_t\);
- one pooled 8-feature OLS is used;
- the five-year expanding walk-forward schedule is respected;
- position direction is the sign of the OLS prediction;
- asset exposures are scaled to a 15% annual volatility target;
- OOS returns are concatenated without leakage;
- portfolio-level rescaling is separately implemented;
- baseline strategies are reproduced;
- deviations from paper methodology are documented rather than hidden.

The important target is **methodological fidelity first**, not forcing the final Sharpe ratio to equal 1.51 exactly.

Different continuous-contract construction and data source can create differences in realized performance even when the methodology is implemented correctly.

---

# 25. First task for the next chat

Do **not** start with graph learning.

Start by asking the user to show the existing dataset schema or run a small inspection cell.

Then implement only:

> **Module 1 + Module 2: returns, volatility, and the eight individual momentum features.**

The first concrete milestone is a validated \(U_t\in\mathbb R^{N_t\times8}\).

Only after that is confirmed should the project move to graph learning.
