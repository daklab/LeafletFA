# Handoff — 2026-08-16 (late)

## Settled today

**Gna13 was a program-only call; A65 measures how often that happens.** Posterior TV 0.33-0.41
in every seed, cells 0.06-0.09; direction real atlas-wide (endothelial vs mesenchymal ~0.15) but
amplified 3x because psi_10 (mesenchymal) sits below every mesenchymal type's observed usage. Of the
3,490 LeafletFA-only gated calls, 2,191 have observed pooled TV >= 0.15 and 1,299 do not. Fig 3e
shows the split (hatched). Example replaced by Mbd3 ATSE1 (brain endo vs oligo, 238/414 cells,
observed TV 0.35, LC q 0.20). Examples must pass observed TV >= 0.25 AND >= 50 cells/arm.

**The posterior test's reach is set by phi distance.** TV_e <= 0.5 ||dphi||_1 (psi sums to 1);
empirically the contrast's 99th-percentile TV = 0.34 ||dphi||_1 - 0.04 (r 0.95). 29 of 70
contrasts have no posterior call at any ATSE (LeafCutter makes 8,345 there). Fig 3 supp
`FIG3_supp_program_sharing` + `FIG3_supp_effect_sizes_by_contrast`. In Methods.

**Atlas K sweep for the posterior test done** (A67 on A51 K10/K30 x 3 seeds): calls 6,360 / 7,041 /
6,666 at K 10/20/30; LC agreement 46/50/50%; K20-K30 Jaccard 0.71, TV rank Spearman 0.97; K10-K20
0.52 / 0.93. LRT and LeafCutter arms do not depend on K. `LEAFCUTTER_CMP_DELTAONLY/ksweep/summary_by_K.tsv`.

**Figure 2 redesigned (v2)**: setup (true-phi UMAP; a DS and a null ATSE with per-program truth +
observed violins) / recovery (learned-phi UMAP silhouette 0.82 vs NMF 0.41; AUROC/AUPRC of DS-ATSE
detection from learned psi vs NMF, mean +- SD over 3 seeds) / test (LRT, posterior from effect 0.25,
LeafCutter: calibration + power) / imputation panel i pending A3. `build_fig2_v2.py`, caption v2.
LeafCutter WAS run on the sims (08/06: 6 v 6 random pseudobulks per state pair).

**Imputation, what is now known** (A66 v2 on ALL 39,812 held-out cells, HVJ set):
- all-entry r is inflated by between-junction level: junction-mean baseline scores 0.86, prior 0.87
- within-junction (ranking cells): prior 0.086, junction-mean 0.000 by construction; posterior mean
  0.995/0.948 at 0/10% removed on all entries but 0.043 on entries with NO reads left at 10%
- so imputation of a cell's rank within a junction from nothing is near the floor for every
  label-free method; the model's honest imputation claims are (i) the junction level, (ii) graceful
  degradation with the cell's own reads, (iii) calibrated per-cell uncertainty
- **posterior sd of the imputed value tracks across-seed spread**: Fgfr2 event, Spearman 0.74
  over cells / 0.76 over cell types, ratio 1.2 (slightly conservative). Worth a panel.
- Fgfr2 IIIb/IIIc (`FIG2_supp_Fgfr2_IIIb_IIIc`, 3 seeds): epithelial recovered (0.90-0.97 vs
  raw ~1.0), mesenchymal direction-only and seed-dependent (0.3-0.8 vs raw 0.0; 5-59 covered
  cells) because mesenchymal types also load SP1/SP4 whose psi_IIIb is unconstrained (cells that
  do not express Fgfr2). Mammary basal clusters with mesenchymal in all seeds (myoepithelial);
  bladder epithelial only in seed 0.

**SP naming**: `figures/_common/programs.py` -- SP1 = largest pi, per fit. Shipped fit: fit-4 -> SP1
(myeloid), 2 -> SP2, 1 -> SP3, 7 -> SP4 (endothelial), 3 -> SP5, 8 -> SP6 (lymphoid), 6 -> SP7,
5 -> SP8, 10 -> SP9 (mesenchymal). NOT yet propagated to: Fbln2 decomposition, Fig 3 heatmaps,
aging, any plot that says "program N".

## Running (check first)

    a3_s0/1/2   IMPUTATION_DM/results/a3_heldout70_seed{0,1,2}.csv   200G GPU, ~3-4 h each
    a66all      IMPUTATION_DM/a66_all_cells/level_{000,010,050,090}.npz  then A66b NMF on all cells
    K3/K10 sims DM_SHIPPED_SIM_K3, DM_SHIPPED_SIM_K10  (18 fits, ~13 done at last check)

## To do -- Figure 2

1. panel i from A3 (3 seeds, mean +- SD): within-junction and within-cell r vs % reads removed,
   posterior mean vs NMF (mean-impute, 0-impute), on all entries and on lost entries; DROP mb_ct.
2. sim K sweep: score K3/K10 fits (A45 LRT, A62 posterior delta 0.15, sim_benchmark metrics incl.
   AUROC/silhouette) -> supp panel: performance vs model K with SD over seeds.
3. supp figure from the old FIG2_final panels (conc recovery, psi vs depth, phi by regime, delta
   trade-off, atlas reach, conc-2 DS) with K and seed SD stated.
4. A66 v2 plot: all cells, rows = posterior mean / NMF / junction-mean baseline, cols = 0/10/50/90%.
   Needs A66b rerun on all cells x HVJ (dense 39,812 x 31k; mean-impute).
5. imputation-uncertainty panel (posterior sd vs across-seed sd), maybe generalised beyond Fgfr2.
6. caption v2: fill i, K sentence, seed sentence.

## To do -- Figure 3

1. propagate SP naming (Fbln2 decomposition, heatmaps).
2. move a/d/f to supp per Karin; main = b, c, e (split bar), g (Apbb2 + Mbd3), + program-sharing?
3. supp panel: posterior calls vs K (table above) + per-seed counts.
4. caption: "program-level, amplified" wording for the hatched bar (not "wrong").
5. methods XX for A3.

## Later
- aging program characterisation on the shipped fit; allen_brain vs TMS separately; Figure 1.

## After Fig 2/3 are cleared: per-dataset training (requested 2026-08-16)

Train LeafletFA on TMS alone and Allen Brain alone, SAME junction set, so cross-dataset batch effects
cannot leak into programs. Imputation repeated on the TMS 70/30 split ONLY; the AB-only model feeds
downstream analyses (conservation with human brain), no AB split needed. Files exist --
see memory `reference_per_dataset_split_files.md`:
  MODEL_INPUT/20260526/train_70_30_TMS_ONLY_*.h5mu, test_30_70_TMS_ONLY_*.h5mu,
  resampled/RESAMPLED_{10..100}_PERCENT_test_30_70_TMS_ONLY_*.h5mu (7),
  AB_ONLY_*.h5mu (full, unsplit -- train on all of it).
Recipe: TMS: A44 with --holdout-ids from the TMS split, shipped config, then A3 on the TMS_ONLY resampled
dir. AB: A44 on AB_ONLY, no holdout.

## Package status (asked 2026-08-17): is the DM/minibatch model in the LeafletFA source?

Yes for the model, no for the packaging:
- IN the package: `leafletfa/minibatch.py` (`LeafletFAMinibatch`: likelihood="dirichletmultinomial",
  psi_parameterization="simplex", conc_per_atse, conc_min, event_id, likelihood_weight; NaN rollback;
  psi_mean/sample_psi normalised within ATSE; freeze_globals with per-ATSE conc), `leafletfa/
  dirichlet_multinomial.py` (dm_log_prob, segment_softmax, denominator_consistency), tests
  (test_dirichlet_multinomial.py 33/33).
- NOT yet: defaults are still betabinomial/beta; `leafletfa/__init__.py` exports only the legacy
  `LeafletFA` (model.py); README still says "Beta-Dirichlet factor model"; the E-step for new cells
  (`estep_minibatch.py`) and the training driver with the shipped recipe (A44: schedule, conc
  floor, delta=K, checkpoints) live in the ANALYSIS repo, not the package.
- To do before release: flip defaults (or add `LeafletFA.dm()` constructor), export
  LeafletFAMinibatch + infer_phi in __init__, move estep_minibatch into the package as
  `leafletfa.estep`, add a `leafletfa train` entry point wrapping A44's recipe, rewrite README.

## Human AB training launched 2026-08-17
`HUMAN_SPLICING_FOUNDATION/LEAFLETFA_MODELS/DM_shipped_2026-08-17/` K10/20/30 x seeds 0-2 (jobs
20688497-9 K20, 20691475-80 K10/K30), shipped recipe, full 43,442 cells. For mouse-AB vs human-AB
conservation. RULE (Karin): every model we train gets K 10/20/30 x 3 seeds -- TMS-only, mouse-AB-only,
any 70/30 refit -- always the full 9-fit grid.

## Full-grid training launched 2026-08-17 (K 10/20/30 x seeds 0-2 each)

    ATLAS_DM/runs_TMS70/      TMS 70% only (54,694 cells; holdout = AB + TMS test)   9 fits  -> A3 on TMS_ONLY resampled
    ATLAS_DM/runs_ABonly/     mouse Allen Brain only (54,571 cells; holdout = TMS)   9 fits  -> conservation vs human
    ATLAS_DM/runs_heldout70/  atlas 70/30 refit, adds K10 + K30 x 3 to the existing K20 x 3   6 fits  -> imputation K-robustness
    HUMAN .../DM_shipped_2026-08-17/   human AB full, K10/20/30 x 3                    9 fits
Holdout index files: ATLAS_DM/holdout_idx_for_TMS70_model.txt, holdout_idx_for_ABonly_model.txt
(verified: TMS train+test = all 78,135 TMS cells, no overlap). Each fit ~3-5 h GPU, 300 GB.
When done: A57 index sanity, A54 scores, A61 cross-seed/K replication on each grid; pick K20 s0
unless the scores say otherwise; then package cleanup (see above) once the queue is empty.

## 70/30 seed 0 merges endothelial with mesenchymal (found 2026-08-17)
||dphi||_1 endo-fibro 0.08 (vs 0.97-1.27 in atlas s0/s1 and 70/30 s1), silhouette 0.009 -- weakest
partition of the six K20 fits although psi matches the atlas 16/20. Known phi-encoding degeneracy
across seeds. A66 cell-type-level metric now queued for 70/30 seeds 1,2 (a66_all_cells_seed{1,2});
report mean and range, flag seed 0. A3 already spans 3 seeds.

## Landed 2026-08-17 (late)
- A66 cell-type-level within-junction r (seed 0, 90% removed): prior phi.psi 0.386 median over 27,107
  junctions; posterior mean 0.830; junction mean undefined. -> Fig 2 imputation text + supp.
- A68: 7.5M entries, Spearman(post_sd, cross_sd) 0.878; 90% CI coverage of other seeds 0.871
  (0.83-0.91 across sd bins). -> supp panel "imputation uncertainty is calibrated" (build from
  IMPUTATION_DM/a68_uncertainty.tsv.gz).
- sim K3/K10 scoring arrays COMPLETED -> aggregate DM_SHIPPED_SIM_K{3,10}/ds/*.tsv into the K panel.
- A66 seeds 1 (running) and 2 (done) + seed-0 phi run (running) -> SP train-vs-heldout heatmap
  (build_sp_train_vs_heldout.py) and mean/range of the cell-type-level r.
- A66b (NMF all cells) resubmitted on CPU with LEAFLET_ALLOW_CPU=1.
- A3 seeds 1-2 finishing; then rebuild Fig 2 (panel i mean +- SD).

## 2026-08-18 morning
- All grids done except atlas 70/30 K10/K30 (4 running). Four K30 fits hit the 41-NaN-restart cap and
  stopped early (TMS s0 11.4k, s2 13.2k; ABonly s0 4.7k; human s0 11.9k steps) -> moved to
  *_INCOMPLETE_cap41 and relaunched with --max-nan-restarts 150. Atlas K30 needed 3-13 restarts.
- A44's silhouette_celltype was MISALIGNED under --holdout-ids (labels in atlas order vs training-only
  phi) -> every held-out fit showed -0.3; fixed in A44 (maps through ho_idx). Ignore the stored values.
- A66 cell-type-level within-junction r: prior 0.39 (all 3 seeds, 0 and 90% removed), posterior mean
  0.88 / 0.83. A68 panel built (FIG2_supp_imputation_uncertainty). SP train-vs-heldout heatmap built.
  A66b (NMF, with cell-type accumulators) rerunning on CPU. Sim K panel: LRT rerun for K3/K10 (A45
  matching bug at K != 5 fixed); K5 posterior rows now from delta 0.15 sweep.

## 2026-08-18 afternoon
- K30 at lr 0.025 does NOT converge on the smaller datasets (TMS-only, AB-only, human): NaN every ~30
  steps, hit the 41 and then 150 restart caps at 4-13k steps. Relaunched K30 x 3 seeds for all three
  at lr 0.015 (--max-nan-restarts 60) into <grid>/K30_lr0.015/ (jobs *K30s*L). Incomplete dirs kept as
  *_INCOMPLETE_cap41 / _INCOMPLETE_lr0.025. Methods: state that K=30 on <60k-cell datasets used lr 0.015.
- A61 on all grids (ATLAS_DM/a61_replication_all_grids.tsv): cross-seed at K20 replicating at r>=0.9:
  atlas 11/20, 70/30 12/20, TMS-only 13/20, AB-only 14/20, human 12/20; K10 6-8/10; K10->K20 6-9/10.
  K30 rows provisional (included incomplete fits) -> rerun A61 after lr0.015 fits land (also fix
  arm_fits to skip *_INCOMPLETE* dirs).
- SP-by-cell-type heatmaps for TMS-only / AB-only / human K20 x 3 seeds: figures/FIGURE4_programs/plots.
- Sim K sweep done and in results draft; NMF cell-type-level r 0.41.
- Still running: hoK30s0/s2 (atlas 70/30), tmsK30s2r (lr 0.025), 9 x K30 lr0.015 queued.
- Package cleanup: still waiting for the GPU queue to drain.

## 2026-08-18 evening
- Atlas 70/30 grid complete (K10/20/30 x 3). A57 on its K20 s0: 17/17 pass.
- K30 lr0.015: TMS x3 done (16.2k steps, ~19 restarts each); AB s0 and human s0 done; AB s1/s2 and
  human s1/s2 running. Grids now expose K30 via symlinks -> K30_lr0.015/ (old lr0.025 dirs kept as
  *_lr0.025_complete_INCOMPLETEGRID / *_INCOMPLETE_*).
- A61 across all grids (a61_replication_all_grids.tsv), programs replicating across seeds at r>=0.9:
    K10  atlas 6, 70/30 8, TMS 8, AB 8, human 6   (of 10)
    K20  atlas 11, 70/30 12, TMS 13, AB 14, human 12   (of 20)
    K30  atlas 17, 70/30 18, TMS 20, AB 18*, human 21*  (of 30; * = mixed lr until s1/s2 land)
  cross-K K10->K20 6-9/10, K20->K30 11-15/20. Same effective-K picture on every dataset.
- After the last 4 fits: rerun A61 once more, SP heatmaps at K30, then package cleanup.

## 2026-08-19
- All 36 training fits COMPLETE (all grids K10/20/30 x 3 seeds; K30 on small datasets at lr 0.015,
  4-22 restarts, full 16.2k steps). Final A61 (a61_replication_all_grids.tsv): r>=0.9 across seeds
  K10 6-8/10, K20 11-14/20, K30 17-20/30 on every dataset; K10->K20 6-9/10, K20->K30 11-15/20.
- K30 SP heatmaps: figures/FIGURE4_programs/plots/SP_by_celltype_{TMSonly,ABonly,humanAB}_K30.
- PACKAGE CLEANUP DONE (v0.2.0): LeafletFAMinibatch.dm(), leafletfa/estep.py, leafletfa/train.py
  (python -m leafletfa.train), exports in __init__, README rewritten. 33 DM tests + phi_init pass;
  smoke fit ok. Defaults NOT flipped (would break constructor calls without event_id) -- .dm() is
  the front door.
- Next: TMS-only A3 imputation (models ready); Fig 4 aging characterisation on shipped fit; Fig 5
  mouse-AB vs human-AB conservation (both grids ready).

## 2026-08-19 (day): Fig 4/5 drafted, TMS imputation running
- TMS-only imputation: A3 x 3 seeds running on IMPUTATION_TMS (TMS_ONLY resampled, runs_TMS70 K20).
- Fig 4 aging on shipped fit (A69 + FIGURE4_programs/FIG4_aging_draft): aging program = the one at
  the mouse-level p floor (0.016) that RISES in 94-99% of cell types in all 3 seeds (seed0 prog6=SP4,
  seed1 prog2, seed2 prog9; psi r 0.95-0.97). Top genes Sos1, Frmd4b, Ppfibp2, Lipe, Fbln2, Cald1...
  Panels: variance partition / violins+mouse dots / delta dot heatmap / top junctions.
  2025 archaeology in the two Explore reports (results embedded in RESULTS skeleton); RBP mediation excluded.
- Fig 5: scripted junction map (pipelines/10_cross_species/00_, CROSS_SPECIES_2026/): 20,583 pairs,
  49% exact, 5,804 events with >=2 matched junctions. Program conservation (01_): junction-mean
  baseline r=0.64; on deviation-psi the SPECIFIC programs (mouse SP14-20) are conserved (r 0.57-0.67
  all 9 seed pairs), broad programs are not (0.13-0.34). To do: name conserved pairs by cell type,
  transfer-learning arm, per-event conserved/diverged classes, sequence filter.
- Methods: aging section rewritten (per-mouse test, ANOVA, content); conservation section updated
  (scripted 2026 map + deviation metric + baseline). Results skeletons: RESULTS_DRAFT_Fig4_Fig5_bullets.md.

## 2026-08-19 evening: TMS-only imputation numbers (seeds 1-2; seed 0 finishing)
- IMPUTATION_TMS/results/a3_tms70_seed{1,2}.csv. All-covered within-junction r, posterior mean:
  0.92/0.91/0.88/0.80/0.68/0.32/0.09 at 10..100% removed (atlas 70/30: 0.85/0.83/0.77/0.66/0.50/
  0.22/0.07); junction mean 0.95 vs 0.89 -> TMS junctions more polarised, gain partly the dataset.
  Prior 0.054; NMF-mean 0.12; lost-entry floor unchanged. TMS-only >= atlas everywhere.
