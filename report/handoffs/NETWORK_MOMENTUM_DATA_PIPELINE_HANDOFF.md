# Network Momentum Replication — Data-Pipeline Handoff

## 1. Project goal

Reproduce the Network Momentum strategy in Pu et al. (2023) on the paper's 64 futures markets. The data stage converts raw WRDS/LSEG contract data into a clean, auditable 64-market daily return panel. The next stage should follow the paper exactly before attempting extensions:

`returns -> volatility scaling -> momentum features -> network construction -> network momentum -> prediction -> positions -> portfolio returns -> Sharpe`

The data pipeline is considered finished and locked. Do not reopen cleaning unless a model-stage validation exposes a material error.

## 2. Local locations

- Project root: `/Users/komali/Documents/ChatGPT/Data Driven Method project`
- Data directory: `/Users/komali/Documents/ChatGPT/Data Driven Method project/Data`
- Primary model input: `Data/paper64_returns_v3.parquet`

All final outputs and audit tables are intended to live under `Data/`.

## 3. Source data

- Source: LSEG futures data accessed through WRDS.
- Core identifiers: LSEG contract family (`contrcode`) and individual futures contract (`futcode`).
- Core daily fields used by the pipeline include trading date and settlement price, together with contract/family metadata needed to order contracts and identify rolls.
- Raw individual-contract histories are first consolidated within LSEG families; paper-market construction occurs only after the family-level continuous series pass QA.

## 4. Paper universe and mapping approach

- Target universe: exactly 64 paper markets (`paper_ticker`).
- The paper universe and its metadata are authoritative: paper ticker, asset class, description, intended start year, and intended end year.
- Each paper ticker is mapped to an ordered chain of one or more LSEG families in `paper64_family_mapping_v3.csv`.
- The source contains 67 distinct LSEG families for the 64-market target. The count differs from 64 because some paper markets require historical predecessor/successor families, while some paper tickers intentionally use a shared LSEG-family proxy.
- Mapping is therefore not forced to be one-to-one:
  - predecessor/successor chains remain ordered and auditable;
  - shared-family proxies retain separate paper tickers, metadata, and output columns even when their underlying source family is the same;
  - no source family is silently substituted or merged merely to make the counts match.
- The final extraction is clipped to each paper market's intended start date and the actual available data end. Coverage shortfalls are recorded rather than filled with fabricated observations.

## 5. Continuous-futures methodology

### 5.1 Within each LSEG family

1. Order the usable individual contracts and create successive active-contract segments.
2. At a resolvable roll, use the common roll-boundary observation to calculate the multiplicative link

   `q_tau = P_new,tau / P_old,tau`.

3. Back-adjust historical segments multiplicatively so older history is expressed on the newest segment's price scale. For an observation in segment `j`:

   `continuous_price = settlement * adjustment_factor_j`.

4. Apply cumulative ratios from newest to oldest. The old segment must include its **own** `roll_ratio_to_next` before its adjustment factor is stored.
5. If a roll cannot be resolved reliably, do not invent a ratio. End the current `block_id`, begin a new block, and preserve the discontinuity as a gap for return purposes.
6. Build one family/date observation, retaining the active `futcode`, `block_id`, adjustment factor, and continuous price for auditability.

This produces `family_continuous_v3.parquet` before the 64 paper markets are assembled.

### 5.2 Across LSEG families for one paper market

ER, MD, ZB, and ZR use ordered historical predecessor -> successor chains.

- Keep the predecessor through its last available date.
- Start the successor strictly **after** the predecessor's last used date, so the paper series never has duplicate dates.
- If the successor also has a valid price on the predecessor's final date, use that overlap only as the stitch boundary:
  - `stitch_ratio = successor_price / predecessor_price` on the boundary date;
  - multiply the immediately connected earlier stitch block by this ratio;
  - mark the transition `connected_to_previous = True`.
- If no valid overlap exists, do not bridge the series. Start a new `stitch_block`, preserve the historical gap/disconnection, and later suppress the return across that boundary.
- Apply the same ordered-chain mechanics to any other multi-family mapping recorded in the mapping table; single-family/shared-proxy mappings require no cross-family stitch.
- Store every segment, source family, boundary, ratio, connection flag, and stitch block in `paper64_stitch_log_v3.csv`.

### 5.3 Block-aware daily returns

Arithmetic daily returns are calculated as

`r_i,t = continuous_price_i,t / continuous_price_i,t-1 - 1`,

but only within a valid return block. A new `return_block` starts at:

1. the first observation of a paper market;
2. a change in `stitch_block` (a disconnected cross-family transition or real gap); or
3. a change in family `block_id` while the `source_contrcode` is unchanged (an unresolved within-family roll).

The first return in every block is `NaN`. `pct_change(fill_method=None)` is used, and non-finite results are replaced with `NaN`. This prevents unresolved rolls and real gaps from appearing as false trading profits or losses.

## 6. Key fixes and why they mattered

### Multiplicative back-adjustment ordering

The original reverse loop stored a segment's factor before multiplying by that segment's own roll ratio. That left the old segment on the wrong scale. The corrected order is:

```python
if pd.notna(q):
    running *= float(q)
adjustment_factor = running
```

This was a mandatory fix because an off-by-one roll factor contaminates the continuous price level at every resolved transition and can generate artificial return jumps.

### Explicit continuity blocks

Unresolved rolls and non-overlapping family transitions are represented by `block_id`/`stitch_block`, not forced links. This matters because price discontinuities that are harmless in a level series become false, potentially extreme, returns if adjacent observations are compared.

### Boundary-safe family stitching

The predecessor is used through its last date and the successor starts afterward. A same-date successor observation may align the two histories but is not emitted twice. This removes duplicate paper-ticker/date rows while retaining a valid multiplicative bridge when one actually exists.

### Paper identity separated from source identity

`paper_ticker` remains the modeling identity and `source_contrcode` remains the data-lineage identity. This preserves the 64-market paper universe while making predecessor/successor histories and shared-family proxies transparent.

### No implicit filling

Returns are not forward-filled across missing days or blocks. `fill_method=None` and the availability matrix ensure that unavailable markets stay unavailable rather than acquiring zero or synthetic returns.

## 7. Final accepted rules

- Use settlement prices and multiplicative, not additive, back-adjustment.
- Use only verified overlap observations to estimate roll/stitch ratios.
- Never manufacture a link when overlap is absent or a roll is unresolved.
- Treat each unresolved transition as a new block and mask the boundary return.
- Keep the predecessor through its last observation; use the successor only after that date.
- Drop duplicate ticker/date observations deterministically and require one observation per market/date.
- Keep paper start dates as the desired coverage target, but accept later starts when LSEG history is genuinely unavailable.
- Retain source identifiers, block identifiers, mapping roles, and stitch metadata in the long-form data.
- Preserve separate paper tickers for intentional shared-family proxies.
- Use the return/availability matrices as-is at the model stage; do not blanket-fill missing values.

## 8. Materiality policy

The pipeline prioritizes errors that can materially alter model returns, covariance/network estimates, or reported performance.

- **Always fix:** mapping errors, duplicate market/date rows, missing continuous prices inside emitted rows, incorrect adjustment-factor logic, fabricated roll links, and returns calculated across known discontinuities.
- **Retain as gaps:** the 29 unresolved rolls. They are not silently interpolated or turned into returns.
- **Drop when clearly dirty and isolated:** a small number of unusable historical source points.
- **Do not keep chasing:** scattered anomalies representing roughly less than 1% of the relevant history when they do not alter the broad sample or material strategy results. Document them and proceed.
- This is a data-quality materiality rule, **not** a rule to delete legitimate daily returns merely because their absolute size exceeds 1%.

## 9. Final QA record

The exact family-level checkpoint preserved in the chat was:

| Check | Result |
|---|---:|
| Distinct LSEG families | 67 |
| Family-level observations | 512,906 |
| Duplicate family/date rows | 0 |
| Missing continuous prices | 0 |
| Unresolved rolls retained as gaps | 29 |

The final 64-market build was accepted only after these invariants passed:

| Check | Accepted result |
|---|---:|
| Paper markets | 64 |
| Wide price/return matrix columns | 64 |
| Duplicate paper-ticker/date rows | 0 |
| Missing `continuous_price` within emitted long rows | 0 |
| Non-finite returns after protection | 0 |

The chat's retained final summary confirms that the model-ready files were saved and the data stage was locked. The detailed terminal printout for total final long-form rows, total return blocks, total `NaN` returns, and maximum absolute daily return was not retained in the conversation snapshot available for this handoff, so those values are intentionally not guessed. They can be regenerated directly from `paper64_model_data_v3.parquet` if needed.

## 10. Final files under `Data/`

### Primary model-ready outputs

| File | Purpose |
|---|---|
| `paper64_model_data_v3.parquet` | Long format: paper ticker, date, continuous price, daily return, return/family/stitch blocks, source lineage, and paper metadata |
| `paper64_returns_v3.parquet` | Final daily arithmetic-return matrix; 64 paper-market columns; primary model input |
| `paper64_prices_v3.parquet` | Wide 64-market continuous-price matrix |
| `paper64_availability_v3.parquet` | Boolean market/date availability matrix derived from non-missing returns |
| `paper64_coverage_v3.csv` | First/last date, observations, stitch-block count, desired start, and start delay by paper market |
| `paper64_family_mapping_v3.csv` | Ordered paper ticker -> LSEG family mapping, including segment order/role and mapping type |
| `paper64_stitch_log_v3.csv` | Cross-family segment dates, stitch ratios, connection flags, and stitch blocks |

### Audit/intermediate outputs

| File | Purpose |
|---|---|
| `family_roll_segments_v3.parquet` | Corrected family roll segments and cumulative multiplicative adjustment factors |
| `family_continuous_v3.parquet` | QA-passed continuous histories for the 67 distinct LSEG families |
| `paper64_continuous_v3.parquet` | Pre-return long-form 64-market continuous-price data, if retained alongside the final model file |

## 11. Known and accepted limitations

- **DT begins in 1998:** earlier paper-target history is not available in the accepted LSEG mapping.
- **ZW begins in 1996:** earlier paper-target history is not available in the accepted LSEG mapping.
- **ER has a real historical gap:** it is preserved and the boundary return is masked; it is not interpolated.
- **29 unresolved rolls remain:** each is retained as a block break/gap, never as a fabricated return.
- Some paper tickers use a shared-family proxy. This can create identical underlying histories for those paper identities; it is an explicit mapping choice, not an accidental duplicate.
- Scattered immaterial source anomalies may have been dropped under the materiality policy above.

## 12. Starting point for the model-replication chat

Use `Data/paper64_returns_v3.parquet` as the authoritative input. First inspect its date index, column count/order, missingness, and availability mask; then implement each paper stage in isolation with intermediate QA and no look-ahead.

Recommended opening prompt:

> Continue my Network Momentum replication project. The data pipeline is finished and locked. The authoritative model-ready daily return matrix is `Data/paper64_returns_v3.parquet`, with availability in `Data/paper64_availability_v3.parquet` and full lineage in `Data/paper64_model_data_v3.parquet`. Reproduce Pu et al. (2023) progressively and exactly before making extensions: returns -> volatility scaling -> momentum features -> network construction -> network momentum -> prediction -> positions -> portfolio returns -> Sharpe. Preserve missingness and block-aware masking, prevent look-ahead at every stage, state each formula and timing convention, save auditable intermediate outputs under `Data/`, and validate each module before moving on. Do not reopen the data-cleaning pipeline unless model-stage QA identifies a material inconsistency.

## 13. Immediate first checks in the new chat

1. Load `paper64_returns_v3.parquet` and verify 64 columns, a sorted unique date index, finite non-missing values, and no silent filling.
2. Load the availability matrix and confirm it matches `returns.notna()` exactly.
3. Lock the paper's timing, volatility estimator, scaling target, momentum horizons, network estimation window, regularization/thresholding, forecast lag, and portfolio normalization before coding the strategy.
4. Build modular outputs so every transformation can be checked for look-ahead and alignment errors.

---

Data-stage status: **complete and locked**. Next stage: **paper-faithful model replication**.
