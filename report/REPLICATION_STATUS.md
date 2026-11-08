# Replication status

## Completed and validated

- Appendix-consistent 60-span EWM volatility, five volatility-scaled return
  features, three normalized MACDs and causal winsorization.
- Five dynamic graph lookbacks and graph ensemble over the full history.
- Five causal expanding OOS blocks covering 2000–2022.
- All six paper strategies: Long Only, MACD, LinReg, GMOM, RegCombo and
  SignCombo.
- Tables 1–4 and Figures 1–10, including transaction costs, regression
  inference, selected-date networks, daily topology, asset-class ablations and
  individual-lookback robustness.
- The disclosed latest-10% validation procedure and all 121 alpha/beta pairs
  for each of the five training periods. The undisclosed score is explicitly
  chosen as pooled validation asset-day forecast correlation.
- Solver, causality, availability and train/test timing QA.
- Selected-date graph validation for 2006-12-22, 2020-03-23 and 2022-04-05,
  including five individual lookback graphs, spectral clustering and t-SNE.
- A reloadable full-history robustness checkpoint containing the ensemble,
  five lookback panels, three class-ablation panels, diagnostics and selected
  graphs.

The robustness run requested 7,546 graph dates and produced 7,002 valid graph
dates; early dates without the full 1,260-observation history or sufficient
simultaneously available assets are intentionally skipped. The final
robustness manifest and all unit tests pass.

## Key completed robustness results

- Full GMOM: annual return 23.23%, annual volatility 15.49%, Sharpe 1.499.
- GMOM-Intra Sharpe: 1.356; GMOM-Inter Sharpe: 1.309.
- Single-lookback Sharpes range from 1.372 (`delta_504`) to 1.499
  (`delta_252`), while the five-lookback ensemble Sharpe is 1.499.
- Table 2 is `results/replication/tables/table_2_asset_class_performance.csv`.
- Table 3 is `results/replication/tables/table_3_lookback_performance.csv`.
- Figures 6–8/10 are in `results/replication/figures/`.

## Remaining fidelity limits

This is a complete, causal replication of the paper's demanded methodology and
reported result families, but not a claim of byte-for-byte author-code
identity. The remaining differences are documented rather than hidden:

1. CLARABEL/SCS and validated warm-start L-BFGS replace the paper's MOSEK
   solver because MOSEK is unavailable in the current environment.
2. WRDS/LSEG data and its unequal local-market calendar replace the authors'
   original vendor panel.
3. The paper does not disclose the graph validation score or every
   portfolio-volatility convention; the configured choices remain explicit in
   `configs/config.yaml` and QA manifests.
4. Headline results use the validated fixed `alpha=beta=0.1` baseline. Grid
   selections and scores are saved separately and are not silently mixed into
   that baseline.
