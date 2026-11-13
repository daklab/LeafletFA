# Simulation inventory (facts only; 2026-11-13; no new runs)

Compiled from the repos, run records, the old manuscript (`~/LeafletFA_overleaf_old/MANUSCRIPT.tex`) and the Dec 2025 preprint text. `V` = `/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation`. Production recipe = DM simplex, K = 20, 16,200 steps × 8,192 cells, lr 0.025 → 0.1×, float64 φ, sparse likelihood, γ_e ≥ 2, δ = K (`METHODS_LEAFLETFA_CURRENT.md` §2–4).

## Answers first
- **Cell-type ψ recovered against a known truth, as a function of reads or cells: no study does this.** Every study scores program ψ, φ, or differential-splicing ranking.
- The closest is a ψ-vs-depth table in study 2 (`DM_FRESH/Q5_recovery_by_coverage.tsv`, ψ r by read-depth bin, 0.3–14 reads). It uses Aug 4 fits on an older recipe, and the script that wrote it was not found (UNSOURCED).
- **Only check 4 (study 1) used the production model and preset.** It has no figure.
- **Studies about programs: all five.** The note does not claim programs.
- **Cited in the current manuscript:** one simulation number (`main.tex:356`, `\SimProgramsRecoveredEasySci` = "0 of 7", check 4).

## Studies

| # | Study | How simulated | Model / preset | Measured | Comparators | Headline | Cell-type ψ vs reads/cells | Programs |
|---|---|---|---|---|---|---|---|---|
| 1 | **Check 4** (validation) | from the fitted seed-0 TMS K20 model (truth = its φ, ψ); real per-cell ATSE totals; y ~ DM(N_e, γ_e φψ). Depth A = TMS; depth B = totals thinned Binomial(N, 0.011) | **production** (run_record: K 20, 16,200 × 8,192, lr 0.025, γ floor 2, float64 φ, sparse, δ = K; commit 9e6a029, branch `leafletfa-validation`); GPU 12.9 min (A), 6.6 min (B) | program ψ (matched r, centered cosine), per-cell φ r, kNN on fitted vs true φ | true φ | A: ψ r **0.979**, centered cos 0.884, **6/7** programs, φ r 0.982; B: ψ r 0.745, **0/7** programs | no | yes |
| 2 | **Figure 2 v2 heart cassette exons** (Genome Biology, Aug 2026) | planted signal on real heart coverage: 6,986 cells, K = 5 programs, 4,268 cassette ATSEs, 30 % differential, effects 0.1–1.0; DM emission, conc 50 (also a conc 2 arm). Generator also does Binomial / Beta-binomial (`--bb-conc`) | **not production**: K 5 (sweep 3, 10), 3,000 × 2,048, lr 0.05 → 0.5×, float32 φ, old driver (`A36_dm_calibration.py`) | program recovery (silhouette, ARI, ψ r), DS ranking (AUROC/AUPRC), DS calibration, γ recovery, ψ r by depth (Q5), misspecified K | NMF (0- and mean-impute), LeafCutter (7 vs 8 pseudobulks) | silhouette **0.82** vs NMF 0.41; at effect 0.5 ψ r 0.989, ARI 0.998, AUROC 0.990 (NMF 0.77); LeafCutter FDP 2.2× nominal at effect 1.0. Caveat in caption: 33 reads/ATSE median vs 16 in the atlas | no (Q5 is ψ r by depth on programs, old fits) | yes |
| 3 | **Old Binomial simulation** (preprint, old manuscript) | TMS brain real ATSE totals, labeled types, 3-junction ATSEs, ψ ~ Beta(0.5, 0.5) per group, Binomial emission | old Beta/Binomial LeafletFA (K 2, 17), no run record | silhouette, Spearman Δ, precision, PR curve | NMF | AP **0.94**; silhouette 0.82 at K = 17 | no | yes |
| 4 | **K overspecification / minibatch** | Beta-binomial, true K = 10 | Beta-binomial, K 10/20/30, 5,000 steps, ~315 s/fit | true programs recovered, pruned K | — | recovered 0.998 at K = 20 | no | yes |
| 5 | **Junction-prior / SVI diagnostics** | Binomial heart simulation (`sim_heart_K5_binom_eff1.0`) | full batch vs chunked, old | ψ r vs truth | — | 0.738 full batch vs 0.646 chunked | no | yes |

**Sources:**
1. `scripts/validation/VALIDATION_PREREG.md:92-110`; `c4_simulate.py`, `c4_recover.py`; `$V/check4/fit_sim{A,B}/run_record.json`; `$V/tsv/c4_recovery{,_programs}.tsv`; `VALIDATION_REPORT.md:102-112`.
2. `Leaflet-analysis/GenomeBiology_Submission2026/pipelines/09_simulation/00_simulate_splicing.py:24-49,321,466-479`; `FIGURE2_simulation_imputation/CAPTION_FIG2_v2.md`; outputs under `Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/{DM_SHIPPED_SIM,_K3,_K10,_CONC2,DM_FRESH,LC_SIM_DM}`; `data/dm_vs_nmf.tsv`; fit cost 24.6 min CPU (job 19663671).
3. `~/LeafletFA_overleaf_old/MANUSCRIPT.tex:497-499, 586-590`; preprint text; `~/LeafletSC/LeafletSC/simulation/simulate_counts.py:121` (commit c7862cc, 2024).
4. `experiments/minibatch/sim_recover10_Kmodel*.csv`.
5. `METHODS.md:712, 1083-1093`; `experiments/junction_prior_shrinks_programs.py`.

**Not simulations:** old Fig. 2B/C (70/30 masking vs NMF on real data) and Fig. 2i (held-out φ, Pearson 0.998; 0.991 with 90 % of reads removed). Both are real data with no ground truth.

## Match to the production model, and cost to rerun
- **1:** matches. Nothing to rerun.
- **2:** differs in K (5 vs 20), steps × minibatch (3,000 × 2,048 vs 16,200 × 8,192), lr schedule, float32 φ and the driver.
  - Rerun estimate: ≤ 13 min GPU per fit, by analogy with check 4 (54,694 cells, 12.9 min). This is an extrapolation, not measured.
  - The study has about 40 fits (21 at K = 5, 18 in the K sweep, plus the conc-2 arm): about 9 GPU-h at most. GPU runs need Karin's approval.
- **3–5:** old Beta/Binomial model. They would need re-implementation on the DM model, not just a rerun; cost UNSOURCED.
- **A cell-type-ψ-vs-depth study does not exist.** It would be new work, not a rerun.

## Figure panels reusable as is
- **None match the production recipe.** Check 4 has no figure.
- From study 2, in `GenomeBiology_Submission2026/figures/FIGURE2_simulation_imputation/plots/` (sizes from the PDF MediaBox; K = 5, Aug recipe):

  | File | Size (mm) | Note |
  |---|---|---|
  | `FIG2_v2.pdf` | 213.0 × 205.1 | wider than 183 mm |
  | `FIG2_supp_from_v1.pdf` | 179.5 × 133.8 | includes ψ vs depth |
  | `FIG2_supp_sim_K_sweep.pdf` | 179.6 × 61.2 | |

## Simulation results the current manuscript cites (`~/LeafletFA_overleaf`, c17d35b)
- **Live:** `main.tex:356` uses `\SimProgramsRecoveredEasySci` ("0 of 7"; check 4, depth B).
- **Comments only:** `main.tex` ~371 (excluded rare-program sentence); ~417 (old "Simulating Junction Usage Ratios", marked superseded by check 4).
- **Defined, unused:** `numbers.tex:16-20` `\SimPsiR` 0.979, `\SimCenteredCos` 0.884, `\SimProgramsRecovered` "6 of 7", `\SimPhiR` 0.982.
- **Supplement:** none.
- `VALIDATION_PREREG.md:360`: the Genome Biology simulation vs NMF was dropped from the Bioinformatics paper (Karin 2026-11-01).
