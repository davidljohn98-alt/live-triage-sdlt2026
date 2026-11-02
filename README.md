# Artefact: Live Triage of Cross-Chain Flows during Bridge Exploits

David L. John and Vallipuram Muthukkumarasamy, School of Information and Communication Technology, Griffith University. SDLT 2026.

Artefact accompanying the SDLT 2026 paper. It contains the
labelled flow panels, the complete evaluation pipeline, the pre-specified
parameter grid, and the result files behind the paper's tables and figures.

## Layout

```
panels/     one CSV pair (train/test) per exploit, plus the no-exploit control
code/       the evaluation pipeline
code/analysis/  scripts for the permutation, heuristic, spread, and lag analyses
results/    the JSON/CSV outputs backing each table and figure
figures/    figure sources (SVG overview; learning-curve PDF)
```

## Data

`panels/` holds one panel per exploit: `<case>_{train,test}.csv`, one row per
reconstructed cross-chain flow, with the `incident_linked` label column
(1 = incident-linked, 0 = benign) and the feature columns described in the
paper (Section 4). The seven exploit panels are ChainSwap, Poly Network,
Multichain, Qubit, Meter Passport, Harmony Horizon, and Nomad; the
`socket_2024_quiet_control` pair is the 15-day no-exploit window of Section 5.6.
The train/test split files record the original within-exploit chronological
partition; the streaming study concatenates each pair and replays the whole
panel in observation-time order (Eq. 1). All data are public on-chain
observations; attacker labels derive from published attribution (paper
appendix).

## Pre-specified parameters (fixed in advance of every replay)

- Fixed alarms: tau in {0.5, 0.9}
- Extreme-value threshold (SPOT): calibration stretch max(150, floor(0.15 n));
  peak level = calibration 0.98 quantile; method-of-moments generalised Pareto
  tail; risks q in {1e-2, 1e-3}; alerting flows excluded from tail updates;
  fallback alert level at the calibration 0.999 quantile until ten excesses exist; minimum stream length 500
- NPDT: window w = max(200, floor(0.3 n)), step w/2, level grid
  z in {2, 2.5, 3, 4, 5, 7, 10}
- Online rank threshold: percentile among previously arrived flows, first
  arrival rank 0; levels top 1% and top 0.1%
- Models: GB, RF, ET, LR, MLP, IF, GNN (simplified graph convolution);
  fixed hyperparameters in `code/train_models_per_case.py` /
  `code/train_score_graph_models.py`; random seed 42
- Training pools library flows in chronological library order; tree-ensemble
  fitting is sensitive to training-row order under a fixed seed (paper Appendix B.5),
  so the ordering is part of the reproducibility contract

## Reproduction map (paper table/figure -> generator -> result file)

| Paper item | Generator | Result file |
|---|---|---|
| Table 2 (panels) | `panels/` row counts | -- |
| Section 5.2 values (GB score extremes) | `code/stream_chrono.py` | `results/scale_grid.json`, `results/score_distributions.json` |
| Table 13 (stream-calibrated thresholds) + appendix grids | `code/stream_chrono.py`, `code/analysis/jobB.py` | `results/chrono_results.json`, `results/rank_grid_all_models.json` |
| Table 8 (oracle) | derived from the two grid files | same |
| Table 3 (heuristics) | `code/analysis/quick_fixes.py` | `results/heuristic_baselines.json` |
| 28-configuration aggregate (Sec. 5.3 prose) | `code/rank_score.py`, `code/analysis/round2_compute.py` | `results/rank_score_results.json`, grid files |
| Table 9 (permuted orderings) | `code/analysis/jobB.py`, `code/analysis/round2_compute.py` | `results/permuted_learning_curve.json` (+ script stdout for audited-cell alert budgets) |
| Table 14 (negative control) | `code/negative_control.py`, `code/analysis/jobB.py`, `code/analysis/round7_compute.py` | `results/negative_control.json`, `results/gnn_negctl.json` |
| Table 10 (latency) | `code/latency_bench.py` | `results/latency_results.json` |
| Figure 2 | `code/plot_learning_curve.py` | `results/learning_curve.json` |
| Hour-granularity lag; pre-alert arrival fraction; ordering spreads; below-minimum-length EVT diagnostic | `code/analysis/quick_fixes.py`, `code/analysis/round2_compute.py` | script stdout (values quoted in §5) |
| Multi-seed variance (Appendix B.5) | `code/analysis/multiseed.py` | `results/multiseed_nomad.json` |
| Value-splitting evasion probe; GNN control row; exact lag arithmetic (§5.8, §5.6, §5.3) | `code/analysis/round7_compute.py` | script stdout |
| Post-onset propagation baseline (Sec. 5.7); matched-budget appendix table | `code/analysis/round8_compute.py` | `results/matched_budget.json` + stdout |
| Per-alert attribution (Appendix B.3) | `code/alert_attribution.py` | `results/alert_attribution.json` |

## Camera-ready analyses (November 2026)

`code/analysis/camera_ready/` holds the analyses added in response to review; their outputs are in `results/camera_ready/`. Set `GEN` in `common.py` to the code root. Each script takes the folder holding `common.py` as its first argument.

| Paper item | Script | Output |
|---|---|---|
| Reproduction of the policy; per-fold timings | `step1_repro_timing.py`, `step13_timing.py` | `step1.json`, `step13.json` |
| No-exploit window, one record per event; reseeds and orderings | `step2_control.py`, `step6_control_robust.py`, `step9_gb_orders.py` | `step2_control.json`, `step6.json`, `step9.json` |
| Summary table intervals; precision-recall curves; recall at fixed budgets; address concentration | `step3_budget_pr.py`, `make_figs.py` | `step3.json`, `pr_curves.json` |
| Attacker-controlled features | `step4_evasion.py`, `step7_evasion2.py`, `step10_mimic.py` | `step4.json`, `step7.json`, `step10.json` |
| False seeds and the fan-out guard | `step5_falseseed.py`, `step8_guard.py` | `step5.json`, `step8.json` |
| Library-address overlap, zeroed address history | `step11_checks.py`, `step12_overlap.py` | `step11.json`, `step12.json` |
| Saved re-check: two-record scoring, risk levels, tail fitting, attacker list, heuristic timing, label checks | `step14_recheck.py` | `step14.json` |
| Public message counts, read once on 2 November 2026 | none (recorded by hand) | `step14_messages.json` |
| Exploit onset and first public report per fold | none (read at source) | `sources_timeline.csv` |


## Running

Python 3 with numpy, pandas, scikit-learn, matplotlib, shap. The pipeline
scripts resolve paths relative to their own location and expect the panels at
`data/observed_panels_v2/` relative to the code root; either recreate that
layout from `panels/` or adjust the `P` constant at the top of each script.
Scripts under `code/analysis/` reference `<ARTEFACT_ROOT>` -- replace it with
the code root path. Training is deterministic given the seed and the library
ordering. `figures/streaming_triage_overview.svg` renders to PDF with any
SVG renderer at 788x430.

## Ethics

Alerts express similarity to past-exploit activity and establish no wrongdoing.
The intended use is human-in-the-loop triage; see the paper's Scope,
Limitations, and Ethical Considerations. Known white-hat and recovery addresses
are excluded from labelling.
