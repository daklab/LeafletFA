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
cannot leak into programs; repeat imputation within each dataset (70/30 inside each). Files exist --
see memory `reference_per_dataset_split_files.md`:
  MODEL_INPUT/20260526/train_70_30_TMS_ONLY_*.h5mu, test_30_70_TMS_ONLY_*.h5mu,
  resampled/RESAMPLED_{10..100}_PERCENT_test_30_70_TMS_ONLY_*.h5mu (7),
  AB_ONLY_*.h5mu (NOT split: make the AB 70/30 + resampled set first).
Recipe: A44 with --holdout-ids from the TMS split, shipped config; A3 on the TMS_ONLY resampled dir.
