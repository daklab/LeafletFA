# Remaining Work — LeafletFA Reproducibility

## LeafletFA repo — clean, migration complete

- [x] Add AnnData format tests to `tests/test_basic.py`
- [x] Add AnnData format description to README and docs
- [x] Run quickstart notebook end-to-end — `cell_ontology_class` is the column name
- [x] Pin deps and promote scanpy/umap-learn to core — done in `pyproject.toml`
- [x] Fix COO sparse FutureWarning / cluster count indexing bug
- [x] Migrate package from `src/BetaDirichletFactor` to `leafletfa/`
- [x] Scaffold MkDocs Material docs site

### Remaining MkDocs pages

- [ ] **Fill `outputs/splicing-programs.md`** — what psi means, how to interpret loadings
- [ ] **Fill `outputs/cell-activities.md`** — assign_post / X_PHI, how to plot
- [ ] **Fill `outputs/differential.md`** — link to `differential_splicing` module
- [ ] **Fill `ecosystem.md`** — scanpy / scvi-tools / muon interop
- [ ] **Deploy docs to GitHub Pages** — add `.github/workflows/docs.yml`

### Other

- [ ] **Implement `analyze_all_junctions_groups`** — high-level wrapper that loops over all junctions for a group-vs-group comparison (equivalent of `analyze_all_factors_psi` but for predefined cell groups). Should call `compute_differential_splicing_groups` per junction, collect into a DataFrame, then call `compute_junctions_significance_groups`. Also fix the docs in `outputs/differential.md` which references a `test_differential_activity` function that does not exist.
- [ ] **Efficiency: vectorize group DS over junctions** — currently loops per junction; could instead draw S posterior samples of `(φ, ψ)` once, compute all imputed PSI values as a single matrix multiply `φ @ ψ` → `[S, C, J]`, then compute group means and ΔPSI across all J junctions in one shot rather than one junction at a time.
- [ ] **PyPI release** — `python -m build && twine upload`
- [ ] **Docstrings on key public methods** — `from_anndata`, `train`, `get_all_variables` need
  Google-style Args/Returns blocks so mkdocstrings renders them properly

---

## leafletfa-paper repo — not started

> **Purpose:** Frozen reproducibility record. NOT a user-facing tool.

- [ ] **Rename repo on GitHub:** Settings → Rename → `leafletfa-paper`
- [ ] **Add figure index to README** — table mapping every paper figure to its script/notebook
- [ ] **Update analysis script imports** — replace HPC `sys.path` hacks with `from leafletfa import LeafletFA`

## Next atlas round — config changes (decided 2026-07-15, apply AFTER array 18866425 finishes)

DO NOT edit run_training.py / config while 18866425 has pending tasks — editable install => mixed-code array.

Evidence (from run_14 / run_3 wandb + sim):
- pass_avg_elbo_per_cell converges by PASS 2-3, then flat-to-slightly-rising (7062 @ pass2 -> 7963 @ pass20).
  20 passes is overkill; the extra passes add sampling noise + mild sequential drift.
- FIRST chunk (150 ep) does NOT converge for RANDOM init (still at 1.68e9 @ epoch 125, floor ~7e8).
  Later chunks (60 ep, warm-started) do roughly settle. So passes and per-chunk epochs are coupled.
- psi_mac_pass_to_pass plateaus at 0.10 (globals stable); cross-arm (waypoint vs random) agreement 0.95
  (programs are data-driven, not init-driven).

Changes:
1. num_passes: 20 -> 8   (test 5/8/20 on the SIM first — fast, has ground truth. Keep 8 as default.)
2. SELECT THE BEST-ELBO PASS as the final model, not the last pass (last-pass is NOT elbo-best; pass 2
   was better for run_14). run_training currently keeps the last pass.
3. lr STAYS 0.05. Do NOT lower it: the model starts on a symmetric saddle and lr is the escape velocity
   (sim: random lr0.005 -> psi_r 0.45 vs lr0.05 -> 0.86). Lower lr re-introduces the saddle failure.
   The drift is a passes+selection issue, not an lr issue.
4. Track phi delta-from-init (currently only psi is tracked, and only for the waypoint arm -> random
   shows nan). Add phi_mac_vs_init + fix the random-arm reference.

Cheap sim test to run before the next atlas round (GPU, ~30 min, does not touch the atlas):
   num_passes in {3,5,8,20} x lr in {0.05} for waypoint+chunked, measure final psi_r AND
   psi_mac_pass_to_pass drift. Locks the pass count with ground truth.

## bb_conc is never carried across chunks (found 2026-07-16)

SYMPTOM: trained bb_conc is bit-identical across waypoints/random and junction/global-prior arms
(same seed => same value to 4 dp) while psi genuinely differs between those arms. It varies only
with (K, seed): K=10 seed0=0.337, seed1=2.053, seed2=1.373.

CAUSE: run_training.py captures global_bb_conc (line ~486) but NEVER FEEDS IT BACK into the next
chunk. grep: psi/pi have 4 feedback sites, bb_conc has 0. So on each of ~240 chunks bb_conc is
re-initialised from its seed-determined prior draw and gets ~60 epochs on 8,192 cells. The saved
value is whatever the LAST chunk landed on -- it never accumulates over the 92,894 cells.

IS LOW bb_conc ITSELF WRONG?  NO -- probably real biology. Beta-Binomial per-cell PSI ~
Beta(rho*c, (1-rho)*c); c < 1 gives BOTH shape params < 1 => U-shaped => PSI pushed to 0/1 =>
BIMODAL, which is the known behaviour of splicing (the simulation deliberately uses Beta(0.5,0.5)
for this reason). The atlas is genuinely more overdispersed than the simulation, which recovered
~20 only because it was GENERATED at conc=20. So do not "fix" low conc -- fix the fact that it is
estimated from one chunk instead of the full dataset.

CONSEQUENCES:
1. bb_conc CANNOT be used as the convergence diagnostic claimed in METHODS 11.4b (it is seed
   noise, not a data-determined quantity). Correct that claim.
2. The held-out E-step inherits it; at conc=0.337 the likelihood is nearly flat so phi leans on
   the prior. This is the condition the old forced-Binomial workaround existed for -- but forcing
   Binomial is still wrong (it is not the trained model). The fix is to estimate conc properly.
3. METHODS 7b says the Gamma(2,0.1) prior "fixes the cause" of tiny bb_conc. NOT SUPPORTED by
   these models (median 3.04, four below 1). Soften that claim.

FIX (next round, with num_passes 8 + best-ELBO-pass selection):
  carry bb_conc chunk-to-chunk exactly as psi/pi/alpha_pi are carried.

## Imputation coverage axis: RESAMPLED_X_PERCENT = % reads REMOVED, not retained (found 2026-07-19)

The resampled test files are labelled by DOWNSAMPLING fraction, NOT coverage retained:
    10_PERCENT  -> 730M reads (MOST coverage, 90% retained)
    50_PERCENT  -> 405M
    90_PERCENT  ->  81M
   100_PERCENT  -> 0.8M  (near-empty, ~0.1%)
The _original truth layer is identical across all files (verified).

CONSEQUENCE: the earlier "imputation collapses at 100% coverage" was NOT a bug -- 100_PERCENT is the
MOST downsampled file, so imputation is correctly worst there. The 04e code is fine (mb=0.80 at
50%-removed is a healthy point). For Figure 3.1 the x-axis must be plotted as COVERAGE RETAINED
(= reads, or 100 - removed%), so the curve reads left-to-right as depth decreasing.

Also (per Karin): for testing we dropped the 10k-HVJ restriction and evaluate ALL junctions
(pass n_hvj huge). Decide for the array whether all-junctions is affordable across 48 models x 7
levels, or keep HVJ for the array and show all-junctions on the released model only.

OPEN QUESTION from the HVJ smoke test (K=10): mb_ct (cell-type-stratified mean) global_pearson was
~0.91 at 50%-removed while LeafletFA lf_bb was ~0.66 -- i.e. the mean baseline BEAT the model. Watch
this on the all-junctions run and across K: if LeafletFA does not beat mb_ct, the programs are
redundant with cell-type-average splicing for imputation. K=10 may just be too weak.

## Imputation finding (run_0, K=10, ALL junctions, 300G) — 2026-07-19

Coverage direction now correct (10%-removed = most reads):
  global Pearson   lf_bb: 0.747 (10%rm) -> 0.570 (90%rm)   mb_ct: 0.960 -> 0.861
  calibration lf_bb slope: 0.99 (high cov) -> 0.87 (low) -- WELL CALIBRATED, not compressed.

HEADLINE: mb_ct (cell-type-stratified mean) BEATS LeafletFA (lf_bb) on imputation at every coverage
(0.96 vs 0.75, 0.86 vs 0.57). Holds on HVJ and all-junctions.

BUT mb_ct is SUPERVISED -- it is handed the 39 cell-type labels. LeafletFA uses none. So this is not
a fair imputation bar. Reframe Figure 3.1:
  - do NOT use mb_ct as the pass/fail bar (it is supervised).
  - fair bars = naive (global mean) and mb (coverage-weighted mean, unsupervised). Beating those is
    the honest claim. lf_bb=0.75 vs naive~0 vs mb (check) -- lf_bb clearly beats naive.
  - show mb_ct as an ADVANTAGED reference ("even a label-conditioned mean only does X better").
  - LeafletFA's selling point is NOT imputation accuracy; it is the calibrated, LABEL-FREE latent
    representation. Lead with calibration + coverage-robustness, not "beats mb_ct".
CAVEAT: K=10 only. K=50 may narrow the gap -- rerun on finalists before finalizing framing.

MEMORY: all-junctions needs ~200-300G/run (5 dense 40k x 80k arrays). NOT feasible for the 48x7
array under the account quota. Plan: HVJ for any sweep; all-junctions only on released finalists.
Better: block-process junctions in 04e (score 20k at a time) to run all-junctions at ~50G.
