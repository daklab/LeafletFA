# LeafletFA validation — pre-registration

Written 2026-10-24, before any validation check ran.
- **Scope:** measurement and diagnosis only. No change to model defaults without Karin's approval.
- **Code:** LeafletFA branch `leafletfa-validation` (from main 7446210 + 92e3834).
- **Compute:** CPU first. GPU only within the remaining approved budget (~3 GPU-h after seeds 1–2), b6k allowed.

Fits used unless stated: the new SpliceVI-baseline fits
`LeafletFA_splicevi_baseline/{TMS,Allen}/full_K20_s{0,1,2}_f64/` (DM simplex, K 20, the shipped recipe,
float64 φ, sparse DM). TRAIN/TEST = SpliceVI's 70/30 split (TMS 54,694 / 23,441; Allen 38,199 / 16,372).
Cell-type label = tissue | broad cell type.

## Check 0 — which model is the paper's model (read-only)
Report the likelihood, ψ parameterisation, concentration, φ prior, K and recipe described by:
- the manuscript text;
- the fits behind each main figure (scripts, run records, commits);
- current main.

**Pass:** all three describe the same model, or every difference is listed with whether it can change a
result. Any mismatch between the text and the figures' model is a required manuscript change, not a fail of
the model.

## Check 1 — held-out likelihood (the open problem)

**1a.** Source of the −3.1e7 gain: script, data, null definition (done; status report §10). Pass =
documented.

**1b. Scoring protocol (fixed before running):**
- **Score:** full DM log-pmf, including the multinomial coefficient, summed over observed (cell, event)
  pairs. Reported as **log-likelihood per observed read** = total log-lik / total reads in the scored
  entries.
- **Fair φ — within-cell event split:** for each TEST cell, its observed ATSEs are split at random 50/50
  (seed 20261025) into halves A and B.
  - φ is inferred by the frozen-global E-step on A only, and B is scored; then the reverse.
  - Score = sum of both halves. No read is ever scored by a φ that saw it.
  - The naive same-reads score (φ inferred on all reads, all reads scored) is reported beside it,
    labelled optimistic.
- **Model:** frozen globals (ψ, π, δ = K, fitted per-ATSE concentration) of the K20 seed-0 fit.
  E-step 3,000 steps, batch 8,192, lr 0.1, γ 0.1, float64 φ.
- **Nulls, every parameter fitted on TRAIN only:**
  - **(i)** one population ψ per junction: TRAIN read-weighted usage within each ATSE.
  - **(ii)** one ψ per junction per cell type: TRAIN read-weighted usage per label. Labels with < 20 TRAIN
    cells or no reads at an event fall back to (i).
  - **(iii)** the same model with K = 1, fitted with the shipped recipe on TRAIN.
  - **Concentration:** for (i) and (ii), each ATSE's concentration is fitted by maximum likelihood on TRAIN
    given that null's ψ (1-D, log-grid over [0.1, 1e4] then refined). It is floored at 2 like the model's.
    (iii) uses its own fitted concentration.
- **Scoring verification before reading any comparison:**
  - (a) Sparse scorer equals the dense `dm_log_prob` + multinomial coefficient on 500 cells (|Δ| < 1e-3 per
    cell).
  - (b) On data simulated from the fitted TMS model (check 4 simulator, TMS depth, 2,000 cells), the TRUE
    parameters score above nulls (i) and (iii).
  - (c) Swapping cells' labels at random makes null (ii) collapse toward (i).
  - If (a), (b) or (c) fails, the scorer is fixed before any number is interpreted.
- **Uncertainty:** 95 % CI of each gain (model − null, per read) by 1,000 bootstrap resamples over mice.

**Pass (per dataset):**
- **Primary:** model per-read log-lik above nulls (i) and (iii) on the event-split score, with the CI lower
  bound > 0.
- **Secondary:** model vs (ii). Reported as a gain or loss with CI. (ii) is a supervised reference that uses
  the labels; being below it does not fail the model, but it is reported.
- **STOP RULE (Karin):** if, after (a)–(c) pass, the model is below ANY null (i), (ii) or (iii) on the
  event-split score (CI upper bound < 0), stop and report before checks 2–5.

**1c.** If the model loses to a null, attribute the loss with these diagnostics:
- **Overfitting:** TRAIN-vs-TEST per-read gap, with TRAIN scored the same way (event split, E-step).
- **Corner prior:** the same score with the prior variant from check 2.
- **E-step convergence:** loss change < 0.1 % over the last 10 % of steps; and 3,000 vs 6,000 steps.
- **Scoring bug:** (a)–(c) above.

## Check 2 — K and prior (TMS)
Fits on TMS TRAIN, the shipped recipe, seed 0:
- K = 5, 10, 30, plus K = 20 from the baseline;
- K = 20 with the **prior floored at 1**: α_k = max(δ π_k, 1), an opt-in flag; defaults unchanged.

Each is scored as in check 1 (event-split per-read log-lik vs null (i)), with kNN (as §170: k 15,
mouse-grouped 4-fold) and live programs (argmax ≥ 1 % of cells).

**Pass:** K = 20 within 1 % of the best per-read gain over null (i) among the K values, OR kNN within 0.02 of
the best. Otherwise report which K is better. The prior variant is descriptive; adopting it needs Karin.

## Check 3 — seed stability (seeds 0–2, TMS and Allen)
- kNN real / content-only / exact average: mean and range.
- Program replication: Hungarian matching of ψ rows by Pearson r for each seed pair. Report the median r of
  matched live programs (live in both) and the number of live programs replicating at r > 0.8.

**Pass:**
- kNN real range ≤ 0.03;
- median matched-live r ≥ 0.8 for every seed pair;
- ≥ 4 programs replicate at r > 0.8 in all three pairs (TMS), ≥ 3 (Allen).

## Check 4 — simulation recovery (TMS)
**Simulator:**
- Truth = the seed-0 TMS fit: ψ, per-ATSE concentration, and the TRAIN cells' φ.
- Each TRAIN cell keeps its observed ATSE totals N_e. The counts per event are drawn
  y ~ DirMult(N_e, conc_e · (φ_c ψ)_e).

Two depths:
- (A) TMS depth;
- (B) EasySci-like: N_e' ~ Binomial(N_e, 0.011), events with N_e' = 0 dropped.

Each is refitted with the shipped recipe, K 20, seed 0 (GPU).

**Recovery measures:**
- ψ: Hungarian-matched Pearson r of the ψ rows of truth-live programs.
- φ: per-cell Pearson r between true and fitted φ after matching; plus kNN on fitted φ vs true φ.

**Pass:**
- (A) median matched ψ r ≥ 0.9 and median per-cell φ r ≥ 0.7.
- (B) reported descriptively, with no pass threshold (expected degraded; this sets the EasySci expectation).

## Check 5 — biological sanity (TMS) and figure reproducibility
**5a.** For each program replicating across seeds (check 3), report:
- the cell types using it (mean φ by label, top 5);
- its top 20 ATSEs by contrast (max − min ψ across programs) and their genes.

Pre-registered known events. A hit = the event is in the program's top 1 % by contrast, in a program whose
top cell types match the expectation:
- *Ptprc* (lymphoid);
- *Fgfr2* IIIb/IIIc (epithelial vs mesenchymal);
- a muscle exon in *Tpm1*, *Tpm2*, *Mef2c*, *Ldb3* or *Ttn* (muscle / cardiomyocyte);
- a neuronal exon in *Clta*, *Nrxn1*, *Ptbp2* or *Kif1bp* (neurons; TMS brain cells where present).

**Pass:** ≥ 2 of the 4 hit.

**5b.** List the manuscript's key quantitative claims (figure, number, script). For each, record whether it
reproduces from the current code and saved fits, with no retraining:
- reproduced: same number within the stated precision;
- not reproduced;
- not checkable (output or input missing).

Retraining for figures is out of scope. **Pass:** every main-figure claim is reproduced or the reason is
explained.

## Reporting
- A TSV per check under `tsv/`; plots under `plots/`.
- `VALIDATION_REPORT.md`: each check, pass or fail against this file, and the manuscript changes required.
- Deviations from this pre-registration are listed as such, with the reason.

## Amendment 1 (2026-10-24, before check 2 ran) — check 2b: is K = 20 a fair Figure 1 setting?

**Question:** does raising K (e.g. to SpliceVI's latent size of 30) give LeafletFA more usable
information, or only more unused programs? This supports the SpliceVI Methods sentence: "LeafletFA was run
with its recommended K = 20. Results were similar at K = 10 and 30 (Supplementary Table SX)."

**Fits:** the TMS fits of check 2 (K = 5, 10, 20 [baseline seed 0], 30; seed 0; shipped recipe).
- K 30 uses lr 0.025, the shipped rate: TMS TRAIN has 54,694 cells, and the README's lr 0.015 note
  applies only below ~60k cells. Deviation: we follow the TMS70 practice, which used 0.015 for K30. A K30
  fit with any non-finite restart is rerun at 0.015 and both are reported.

**Projections per K:** real, content-only and exact-average TEST inputs, with the same E-step as the
baseline (frozen globals, 3,000 steps, float64 φ).

**Readouts per K:**
- kNN on centred log φ: real, content-only, exact average, and the content-only drop (real − content-only).
  The §170 protocol: k 15, standardised, mouse-grouped 4-fold, label = tissue | broad type.
- Live programs: argmax of ≥ 1 % of TEST cells.
- Effective dimensionality of the embedding: participation ratio (Σλ)² / Σλ² of the covariance
  eigenvalues of centred log φ, real TEST input.
- Event-split held-out log-likelihood per read vs null (i), as in check 2.

**Pre-registered rules:**
- **"Results similar at K = 10 and 30" holds** iff, for both K = 10 and K = 30:
  - |kNN_real(K) − kNN_real(20)| ≤ 0.03; and
  - |drop(K) − drop(20)| ≤ 0.03.
  If it does not hold, the Methods sentence is not used; the table is reported as it is.
- **K = 30 replaces K = 20 in Figure 1** if kNN_real(30) − kNN_real(20) > max(0.03, the seed range of
  kNN_real at K = 20 from check 3).
  The baseline should be LeafletFA's strongest reasonable version; the switch needs Karin's sign-off.
- **"Higher K adds unused programs, not information" is supported** at K = 30 iff all three hold:
  - live programs rise by ≤ 3 over K = 20;
  - the participation ratio rises by < 20 %;
  - kNN_real is not better by > 0.03.

**Scope:**
- TMS only within the current GPU budget.
- An Allen K 10 / 30 pair would cost ~2.5 GPU-h and needs Karin's approval. Allen's 4 live programs at K 20
  make it the dataset where this matters most.
- Planted-signal AUROC per K can be scored by the SpliceVI agent's §170 readout from the same exports, if
  wanted.

## Amendment 2 (2026-10-25, before check 3 ran) — replication on centred ψ

**Reason:** the 27 Sept A70 analysis (Leaflet-analysis 7971974) showed raw ψ similarity is ~0.98 for
every program, because programs share most of each junction's usage. Pearson r on raw ψ rows (as
pre-registered in check 3) is therefore inflated and cannot separate reproducible from non-reproducible
programs.

**Change:**
- **Primary metric** is the A70 metric. Each seed's ψ is centred by its π-weighted mean composition, and
  programs are Hungarian-matched on cosine of the centred rows.
- **A program replicates** if it is live (argmax ≥ 1 % of cells) and its matched centred cosine is
  ≥ 0.8 in BOTH other seeds (seed 0 as reference, as in A70).
- **Raw-ψ Pearson r** is still reported, as a secondary metric.

**Pass criteria (replacing check 3's):**
- kNN real range ≤ 0.03 (unchanged);
- ≥ 4 programs replicating (TMS) and ≥ 3 (Allen) under the centred rule;
- report the median centred cosine of matched live programs per seed pair.

Check 5a's "replicating programs" use this definition.

## Amendment 3 (2026-10-26, before any computation) — check 6: program stability up to resolution

**Question:** are programs that fail the one-to-one rule recovered in other seeds as splits or merges of
programs, or as programs with the same meaning?

**Fits:** `LeafletFA_splicevi_baseline/{TMS,Allen}/full_K20_s{0,1,2}_f64` (K 20, seeds 0–2). Training
cells only.
- **Live program:** argmax of ≥ 1 % of the fit's cells (amendment 2).
- **Centered ψ:** each fit's ψ minus its π-weighted mean composition (A70).
- Every ordered seed pair (i → j) is analyzed: 6 per dataset.

**1. Many-to-one (primary).** For each live program a of seed i, its centered ψ vector c_a (all junctions)
is regressed on the centered ψ of the live programs of seed j by non-negative least squares, with no
intercept.
- R² = 1 − ‖c_a − Xw‖² / ‖c_a‖² (c_a is already a deviation vector, so zero is its reference).
- "Programs needed" = the smallest s ∈ {1, 2, 3} for which the best subset of s seed-j programs
  (exhaustive search) reaches R² ≥ 0.8. The R² with all live seed-j programs is also reported.
- Note: for s = 1, R² ≥ 0.8 means centered cosine ≥ 0.894, stricter than the one-to-one rule (≥ 0.8).

**Classification of each live seed-i program, per ordered pair (first match wins):**
- **one-to-one:** its Hungarian-matched seed-j program (centered cosine) is live and the cosine is ≥ 0.8;
- **reproduced up to splitting/merging:** otherwise, R² ≥ 0.8 with ≤ 3 seed-j programs. Labeled a *split*
  (one seed-i program spread over 2–3 seed-j programs) when s ≥ 2; s = 1 cannot occur here, since that
  implies cosine ≥ 0.894;
- **not reproduced:** otherwise.
- Merges are the same relation seen from the other side: a seed-j program needing 2–3 seed-i programs.
  They are listed per pair with their top cell types.

**2. Meaning-level match (supporting, descriptive).**
- **(a)** Cell-type activity profile: mean φ per tissue–cell type (types with ≥ 20 training cells).
  Pearson r of program a's profile with (i) the best single seed-j program (maximum r) and (ii) the
  summed loadings of the many-to-one set (Σ_{b∈S} φ_b).
- **(b)** Top-50 marker junctions, ranked by the departure of its ψ from the π-weighted mean of the other
  programs of its fit.
  - For a set S, the junctions are ranked by |Σ_b w_b c_b| from the NNLS fit.
  - Report the Jaccard overlap with the best single program (highest centered cosine) and with the set.

**3. Reporting.** Per dataset and ordered seed pair: counts of one-to-one, up to splitting, and not
reproduced; the list of splits and merges with the top 3 cell types of each member.

**Reading (Karin):**
- **If at least half of the live programs are reproduced (one-to-one or up to splitting) in every ordered
  seed pair, in both TMS and Allen,** the paper may say "programs are reproducible up to resolution".
- **Otherwise the current wording stays:** program-level statements only for one-to-one reproduced
  programs.
- The meaning-level results (2) are descriptive and do not change the reading.

## Amendment 4 (2026-10-30, before any fit or computation) — check 8: consensus programs (cNMF-style)

**Question:** do programs averaged over many seeds (consensus programs, as in cNMF; Kotliar et al. 2019)
recur reliably and keep a stable junction identity, where single-seed programs do not?

**Fits.** TMS, K = 20, production recipe (`run_leafletfa_baseline.py` defaults: 16,200 steps, batch 8,192,
lr 0.025 decaying to 0.1×, concentration floor 2, float64 φ, sparse likelihood), training cells only
(54,694). 15 seeds in total:
- seeds 0–2 reused from `LeafletFA_splicevi_baseline/TMS/full_K20_s{0,1,2}_f64`;
- seeds 3–14 new, fit only (no projections), written to `c8_consensus/fits/full_K20_s{N}_f64`.
- GPU (b6k allowed), about 0.25 GPU-h each; stop and ask before 6 GPU-h in total.
- A fit that ends with non-finite φ is excluded. The recurrence denominator is the number of fits that
  completed; if fewer than 12 complete, stop and report.

**Pooling and clustering.**
- **Live programs** per fit: argmax of ≥ 1 % of training cells (amendment 2).
- **Centered ψ** per fit: ψ − π·ψ (π-weighted mean composition), all 79,768 junctions (A70).
- Pool the centered ψ of all live programs of all fits. Agglomerative clustering, **average linkage,
  cosine distance (1 − cosine)**, cut at **0.2**: members of a cluster have mean pairwise centered cosine
  ≥ 0.8, the same threshold as the one-to-one rule.
- **Seed recurrence** of a cluster = distinct fits contributing ≥ 1 member / number of fits.
- **Keep clusters with recurrence ≥ 0.6** (≥ 9 of 15).
- Sensitivity (descriptive; does not change the reading): cuts at 0.1 and 0.3; complete linkage at 0.2.

**Consensus program.**
- ψ = mean of the members' raw ψ (stays on each ATSE's simplex). A fit with > 1 member in the cluster is
  averaged first, so each fit has equal weight.
- π = for each fit, the summed π of its members (0 if absent), averaged over fits, renormalized over the
  consensus programs.
- Per-ATSE concentration = median over the fits' learned concentrations (all ≥ 2).

**E-step.** Consensus ψ, π and concentration frozen; δ = number of consensus programs; production E-step
(3,000 steps, batch 8,192, lr 0.1 decaying to 0.1×) on the 54,694 training cells (`--reference-fit`).

**Report, per consensus program.**
- seed recurrence, number of members, mean pairwise centered cosine;
- cell-type usage profile: mean consensus φ per tissue–cell type (types with ≥ 20 cells), top 5 types;
  fraction of cells whose argmax it is;
- top 10 junctions, ranked by marker score (check 6 definition: departure of its ψ from the π-weighted
  mean of the other consensus programs);
- **within-cluster top-50 Jaccard**: mean pairwise Jaccard between the top-50 marker-junction sets of
  members from different fits, each member's markers computed in its own fit (check 6 definition).

**Reading (Karin).** "Consensus restores interpretable programs" if **at least 8 consensus programs**
recur in ≥ 60 % of fits **and** have within-cluster top-50 Jaccard ≥ 0.5. Otherwise the current wording
stays.

**Pseudobulk check (descriptive; does not change the reading).**
- **Strongest 3** consensus programs: among those with recurrence ≥ 0.6, highest within-cluster Jaccard
  (ties: higher recurrence).
- **Top ATSEs:** for each, rank ATSEs by total-variation distance of its composition from the
  π-weighted mean of the other consensus programs, 0.5 Σ_j∈e |ψ_kj − ψ̄_j|; the top 10 ATSEs that are
  testable (below) are tested. Tested junction = the one with the largest |ψ_kj − ψ̄_j| in the ATSE.
- **Users / non-users:** tissue–cell types (≥ 20 training cells) with mean consensus φ_k ≥ 0.2 /
  ≤ 0.02; if fewer than 3 users, the top 3 types by mean φ_k.
- **Pseudobulk samples:** (tissue–cell type × mouse) groups, summed reads, kept at an ATSE if ≥ 20 reads
  there. An ATSE is testable if ≥ 3 user and ≥ 3 non-user samples are kept.
- **Test:** two-sided Mann–Whitney U on the tested junction's pseudobulk usage, users vs non-users;
  Benjamini–Hochberg over all tested ATSEs of the 3 programs. **Concordant** = sign(median users −
  median non-users) equals sign(ψ_kj − ψ̄_j). **Confirmed** = concordant and q < 0.05.
- **Null:** 200 random testable ATSEs per program (seeded), a random junction in each, same test; report
  the fraction with P < 0.05.
- Caveat stated in advance: samples from the same mouse are not independent across cell types.

Code: `scripts/validation/c8_consensus.py`. Results: `tsv/c8_*`, section "Check 8" in VALIDATION_REPORT.md.

## Amendment 5 (2026-11-01, before any computation) — check 9: is differential splicing reproducible and calibrated?

**Question:** can LeafletFA's differential-splicing test be a main figure (Bioinformatics Fig. 3)? CPU only, no new
training. Fits: the 15 TMS K20 fits of check 8 (seeds 0–2 from `LeafletFA_splicevi_baseline/TMS/`, seeds 3–14 from
`c8_consensus/fits/`), training cells only (54,694). Posterior parameters (q(ψ) and q(φ) loc and scale) are read from each
fit's `checkpoint.pt` (step 16,199, the final step), since `fit.npz` stores only point estimates.

**The shipped test** (A51 as used for GB Fig. 3, `analysis_matched`):
- Δ_j = Δφ̄ · ψ_:,j per posterior draw, 200 draws of ψ and of the group-mean φ difference;
- TV = ½ Σ_{j∈e} |Δ_j| per draw; P(TV > δ) with **δ = 0.15**; Bayesian FDR (running mean of 1 − P); call at **q ≤ 0.05**;
- coverage gate: an ATSE is tested only if **≥ 25 cells in each arm** have ≥ 1 read on it (A67).

**9a. Reproducibility across seeds.**
- **Contrasts:** from GB Fig. 3b's 70 TMS tissue–cell-type contrasts
  (`LEAFCUTTER_CMP_DELTAONLY/analysis_group/a50_contrast_comparison.tsv`, column `lfa_cal` = calls at matched FDP), keep
  those whose two arms each have ≥ 50 training cells in the TMS-only split; rank by `lfa_cal`; take the contrasts at the
  minimum, 25th percentile, median, 75th percentile and maximum (nearest rank). 5 contrasts.
- Run the shipped test with each of the 15 fits (same draws seed, 0).
- **Report:**
  - pairwise Jaccard of the gated call sets (105 fit pairs × 5 contrasts), median and range per contrast;
  - the fraction of seed-0 calls called in ≥ 12 of 15 fits;
  - for tested ATSEs, the cross-seed correlation of per-cell ψ (ρ_cj = φ_c · ψ_:,j) over the contrast's cells and the
    tested ATSEs' junctions: Pearson r per fit pair, median.
- **Pass:** median pairwise Jaccard (over all fit pairs and contrasts) **≥ 0.7**. The other two numbers are reported, not
  thresholded.

**9b. Calibration on a real-data null.**
- **Cell types:** the 5 TMS tissue–cell types with the most mice having ≥ 10 training cells (ties: more cells); a type
  needs ≥ 8 such mice. Only those mice's cells are used.
- **Splits:** within each mouse, the type's cells are shuffled and split in half (odd cell to arm A), so both arms have
  the same mice. 20 splits per type (seeds 0–19). Every call is false.
- **LeafletFA:** the shipped test (seed-0 fit). Report per split: ATSEs tested (gate as above), calls; **false-call rate =
  calls / tested**; per type: mean over splits and the fraction of splits with ≥ 1 call. The test is an FDR procedure, so
  on an all-null set the target is no calls; nominal is 0.05.
- **LeafCutter** on the same splits: pseudobulk = one sample per (mouse, arm), summed junction counts; `leafcutter-ds`
  as in GB (`08_leafcutter_ds/05_permutation_null_leafcutter.py`: min 4 samples per group and per intron), no covariates
  (both arms have the same mice). Report the fraction of tested clusters with p < 0.05 (nominal 0.05) and with BH
  q < 0.05 (target 0).
- **Pass (LeafletFA):** mean false-call rate **≤ 0.05 in each of the 5 types**. LeafCutter's numbers are reported, not
  thresholded.

**Reading (Karin's rule):** if 9a and 9b both pass, differential splicing may be main Fig. 3 and simulation recovery moves
to the supplement. Otherwise DS stays out of the main text.

**Not run:** the GB simulation vs NMF (dropped from the Bioinformatics paper; Karin, 2026-11-01).

Code: `scripts/validation/c9_ds_seeds.py` (9a), `c9_ds_null.py` (9b). Results: `tsv/c9_*`, section "Check 9" in
VALIDATION_REPORT.md.

### Amendment 5a (2026-11-01, before computation) — deviation from amendment 5: 9a repeated on the GB Fig. 3c setting

**Reason.** Amendment 5 chose its 5 contrasts by GB Fig. 3b **LRT** counts; four of them had 0 gated **posterior** calls in
GB Fig. 3c, so 9a tested the posterior test where it was already known to be silent. It also ran the test with settings
that differ from GB Fig. 3c: TMS-only fits instead of the pooled-atlas fits; q computed after the coverage gate (over the
gated ATSEs only) instead of over all 24,475 ATSEs before gating.

**Repeat (9a′).**
- **Fits:** the pooled-atlas K20 fits the test was designed for (`ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed{0,1,2}_b8192`),
  the 3 seeds available.
- **Test and settings exactly as GB Fig. 3c:** the existing per-seed A51 outputs (`LEAFCUTTER_CMP_DELTAONLY/a51_delta0.15`,
  `a51_seed1_delta0.15`, `a51_seed2_delta0.15`; δ = 0.15, 200 draws, ≥ 50 cells per arm, Bayesian FDR over all ATSEs),
  then the A67 gate (≥ 25 covered cells per arm, from `analysis_matched/posterior_consensus_gated.tsv.gz`). No re-run of
  the test: these are the shipped call sets.
- **Contrasts:** the 5 with the most gated consensus calls in GB Fig. 3c.
- **Report:** per contrast, each seed's gated call set (q ≤ 0.05 and gate) and the pairwise Jaccard (3 pairs); the
  fraction of seed-0 calls called in both other seeds; Pearson r of per-cell ρ = φ·ψ between seeds over all cells of the
  two arms and the junctions of the called events (union over seeds).
- **Pass:** median pairwise Jaccard ≥ 0.7 (as in amendment 5).
- The amendment 5 result (9a on TMS-only fits) stands as recorded. If 9a′ passes, DS is reproducible on the atlas fits in
  the GB setting, not on the TMS-only fits.

Code: `scripts/validation/c9_ds_atlas.py`. Results: `tsv/c9a2_*`.

## Amendment 6 (2026-11-01, before any computation) — check 10: the cell-type-resolved splicing catalog, and benchmarks

**Question:** is LeafletFA's cell-type-resolved junction usage (ψ per ATSE × cell type, with an interval) more accurate
than raw pseudobulk, calibrated, and are its cell-type-specific calls reproducible? This is the paper's resource
(Karin, 2026-11-01). CPU only; existing K20 fits; TMS-only and Allen-only 70/30 splits.

**Cell types.**
- TMS: `tissue_cell_type` (144; types with ≥ 10 TRAIN cells are used).
- Allen: **subclass** = `cell_ontology_class` with its numeric cluster prefix removed (`168_L2/3 IT CTX` → `L2/3 IT CTX`);
  types with ≥ 10 TRAIN cells. (The model-ready files have no separate subclass column; this is the Allen cluster's
  subclass name.)
- Fits: TMS seeds 0–14 (check 8), Allen seeds 0–2. Primary estimates use seed 0; seeds are used only for 10.2c.

**Estimator (LeafletFA cell-type ψ).** For cell type t and junction j: the mean over t's TRAIN cells of the per-cell
posterior-mean usage ρ̃_cj = (γ_e ρ_cj + y_cj) / (γ_e + N_ce), with ρ = φψ (cells with no reads at e contribute their
prior ρ_cj). **Interval:** 2.5–97.5 % over 200 joint draws from q(ψ) and q(φ) (read from the fit checkpoint), each draw
giving the type mean of ρ̃; γ_e at its fitted value.
**Comparator (raw pseudobulk).** Σ_c y_cj / Σ_c N_ce over t's TRAIN cells; where t has 0 TRAIN reads at e, the TRAIN
population ψ (all cells).

### 10.1 Estimation accuracy (against TEST pseudobulk)
- **Units:** (ATSE e, type t) with ≥ 20 TEST reads at e and ≥ 3 TEST cells covering e, in types with ≥ 10 TRAIN and ≥ 5
  TEST cells.
- **Error:** total variation between the estimate and the TEST pseudobulk composition, ½ Σ_{j∈e} |ψ̂_tj − ψ^TEST_tj|.
- **Bins:** type size by TRAIN cells (10–49, 50–199, 200–999, ≥ 1,000) and TRAIN reads at (e, t) (0, 1–19, 20–99,
  ≥ 100). Report mean error per method per bin, and the paired difference (LeafletFA − raw) with a 95 % bootstrap CI
  over cell types (1,000 resamples).
- **Pass (Karin's rule):** overall mean error of LeafletFA ≤ raw pseudobulk, **and** LeafletFA better (CI of the paired
  difference entirely below 0) in the smallest type-size bin **or** the lowest-read bin with ≥ 1 read (1–19). Separately
  reported: the 0-read bin (raw falls back to the population ψ there).

### 10.2 Cell-type-specific splicing calls
- **Leave-one-type-out reference ψ_−t:** the same estimator over all TRAIN cells not in t (same draws).
- **δ from the data:** split each type's TRAIN cells in half within each mouse (seed 0) and compute TV between the two
  halves' LeafletFA estimates for every (e, t) unit of 10.1's coverage rule applied to TRAIN; **δ = the 99th percentile,
  rounded up to the next 0.05** (one δ per dataset, reported).
- **Call:** (e, t) is specific if the point TV between ψ_t and ψ_−t exceeds δ **and** for at least one junction j of e
  the 95 % intervals of ψ_tj and ψ_−t,j do not overlap. Only units with ≥ 20 TRAIN reads and ≥ 3 covering TRAIN cells
  in t are callable (no calls from the prior alone).
- **10.2a Replication across mice:** mice split at random into halves A and B (seed 0; Allen donors likewise). Calls
  made with half-A cells (same fit; estimator restricted to A's cells); confirmed in half B if (i) the same rule holds
  on B's cells, and, model-free, (ii) B's raw pseudobulk TV between t and the rest exceeds δ/2 with the same sign at the
  call's most different junction. Report the fraction confirmed under (i) and (ii).
- **10.2b Null:** the same confirmation after permuting cell-type labels among B's cells within each mouse (20
  permutations); report the null confirmation fraction. Pass: confirmed fraction (ii) ≥ 0.5 and ≥ 5× the null.
- **10.2c Across seeds:** call sets per seed (all TRAIN cells), pairwise Jaccard (TMS 105 pairs, Allen 3). Reported;
  pass: median Jaccard ≥ 0.7 (same bar as check 9).

### 10.3 Interval calibration
- Units of 10.1 with ≥ 100 TEST reads and ≥ 10 covering TEST cells: the fraction of junctions whose TEST pseudobulk ψ
  falls inside the LeafletFA 95 % interval, by TEST-read bin (100–499, ≥ 500). TEST pseudobulk is itself noisy, so
  coverage is reported, not thresholded; nominal 0.95.

### 10.4 Known biology (each stated before looking)
- ***Fgfr2* IIIb/IIIc (TMS):** every epithelial type (keyword list as in `build_fgfr2_example.py`) has a higher IIIb
  usage estimate than every mesenchymal type; and the catalog flags IIIb as specific in ≥ 1 epithelial type.
- ***Ptprc* (TMS):** at least one *Ptprc* ATSE is flagged specific in a lymphoid type (B or T cells), with the direction
  confirmed in TEST pseudobulk.
- ***Nrxn1–3* SS4 (Allen):** for each of *Nrxn1*, *Nrxn2*, *Nrxn3*, the ATSE containing the SS4 alternative exon
  (identified by annotation before looking at usage); at least two neuronal subclasses differ by more than δ with
  non-overlapping intervals. Reported per gene.

### 10.5 Export
- The catalog table: event id, gene, junctions (coordinates), annotation status; per (event, type): ψ per junction,
  interval, TRAIN cells, TRAIN cells covering, TRAIN reads, specific flag, ψ_−t and TV. One table per dataset, in the
  Zenodo export script; the portal is built from it.

### 10.6 Benchmarks (descriptive; no claims beyond the measured numbers)
- **Hardware:** recorded per run (CPU model and cores, GPU model).
- **LeafletFA:** fit wall-clock and peak host/GPU memory at TMS size (from the production run records and logs) and at
  atlas size (pooled 132,706 cells, one K20 fit, sparse float64, GPU, if Karin approves ~0.5 GPU-h; else the existing
  dense atlas fit's log, labeled as such); E-step for new cells (TMS TEST, existing records).
- **scQuint** on the same TMS TRAIN data (CPU or GPU as its defaults; settings from the SpliceVI §170 run).
- **LeafletSC** (original Leaflet) on TMS TRAIN, if it still installs and runs; otherwise reported as not runnable.
- **LeafCutter** producing the same cell type × event catalog: one-vs-rest pseudobulk run per cell type; total wall
  clock and peak memory.

**Reading.** If 10.1 and 10.2 (a/b) pass, the paper is an Original Paper with the catalog as its resource; otherwise
an Application Note. 10.2c, 10.3, 10.4 and 10.6 are reported, not gating.

Code: `scripts/validation/c10_catalog.py`, `c10_benchmark.py`. Results: `tsv/c10_*`, section "Check 10" in
VALIDATION_REPORT.md.

### Amendment 6, note 1 (2026-11-01, before any check 10 computation)
- **Allen cell types (Karin: use the SpliceVI §111 subclass labels).** SpliceVI STAGE5 §111 (`audit_2026_09/stage5_s111_score.py`
  line 21) defines subclass as `cell_ontology_class` with the leading cluster number removed (`168_L2/3 IT CTX` →
  `L2/3 IT CTX`), read from the model-ready obs, not joined from separate Allen metadata. That is the definition above:
  380 clusters → 70 subclasses, 68 with ≥ 10 TRAIN cells. Karin's instruction also said "do not strip cluster numbers";
  the two conflict, so **the Allen part of check 10 waits for Karin's confirmation**; TMS proceeds.
- **Atlas-size LeafletFA benchmark fit approved** (Karin, 2026-11-01): ≤ 0.5 GPU-h, b6k, pooled 132,706 cells, K20,
  production recipe.
- **Atlas benchmark scope (before submitting):** a full 16,200-step fit at 132,706 cells is estimated at ~1.1–1.3 GPU-h
  (from 0.05 s/step TMS, 0.14 s/step Allen), above the 0.5 GPU-h approved. So the benchmark runs **2,000 steps** on a
  b6k card (`--fit-steps 2000 --no-md5`, otherwise the production recipe), recording s/step and peak host/GPU memory;
  the full-fit time is reported as s/step × 16,200, labeled as extrapolated. A full fit only with further approval.

### Amendment 5b (2026-11-01, before computation) — check 9a′ post hoc, descriptive
Labeled post hoc in the report; the pre-registered 9a′ Jaccard result (0.548, fail) stands. Same 5 contrasts and 3 atlas
seeds, shipped A51 outputs, gated ATSEs (A67 gate):
- Spearman correlation between seeds of per-event effect (A51 `tv`) and of posterior probability (`prob_de`), over all
  gated ATSEs of each contrast (3 seed pairs × 5 contrasts).
- Relaxed replication: of the events called in seed A (q ≤ 0.05), the fraction with q < 0.2 in seed B **and** the same
  sign of Δψ at seed A's lead junction (largest |Δψ_j|, Δψ_j = (φ̄_1 − φ̄_2)·ψ_j from each seed's point estimates).
- Sign agreement of Δψ at the lead junction for events called in both seeds of a pair.
- The August consensus rule (called in ≥ 2 of 3 seeds): not re-testable for stability with only 3 seeds, so reported as
  the fraction of each seed's calls that are in the consensus set, and the consensus set's size per contrast.
Code: `scripts/validation/c9_ds_atlas_posthoc.py`; `tsv/c9a3_*`.

### Amendment 6, extension 1 (2026-11-01, before any check 10 computation) — pairwise cell-type comparisons (10.7)
- **Contrasts:** the 70 GB Fig. 3b/3c TMS tissue–cell-type pairs (`a50_contrast_comparison.tsv`) with both types having
  ≥ 10 TRAIN cells in the TMS-only split.
- **Rule (as 10.2):** an (event, pair) is called if the point TV between the two types' LeafletFA cell-type ψ exceeds
  δ (10.2's data-derived δ) **and** at ≥ 1 junction the two 95 % intervals do not overlap; callable only with ≥ 20 TRAIN
  reads and ≥ 3 covering cells in **each** type.
- **Validation (as 10.2a/b):** calls on mouse half A; confirmed in half B (i) by the same rule on B's cells and,
  model-free, (ii) by B's raw pseudobulk TV between the two types > δ/2 with the same sign at the lead junction; null: 20
  within-mouse permutations of the two types' labels in B. **Pass:** confirmed fraction (ii) ≥ 0.5 and ≥ 5× the null,
  pooled over the 70 pairs (per-pair values reported).
- **If it passes,** the portal offers pairwise comparison computed client-side from the stored ψ and intervals with the
  same rule; if not, the portal shows ψ and intervals side by side without a call.

### Amendment 6, note 2 (2026-11-01, before any check 10 result was read, including the smoke test)
- **Allen cell types (Karin):** option (a), the SpliceVI §111 subclass labels (cluster number removed; 70 subclasses,
  68 with ≥ 10 TRAIN cells). The Allen part proceeds.
- **Atlas benchmark (Karin):** the full 16,200-step atlas fit is approved (~1.3 GPU-h, b6k); the 2,000-step timing run is
  kept as well.
- **What the 200 draws sample (stated from `c10_catalog.py` as written, before results):** each draw is one joint draw
  of φ for all TRAIN cells from q(φ) and of every program's ψ_k from q(ψ) (the fit's variational posterior); y, N and
  γ_e are fixed. Given that draw, each cell's posterior-mean usage ρ̃_cj = (γ_e ρ_cj + y_cj)/(γ_e + N_ce) is computed
  deterministically and averaged over the type's cells. There is no per-cell Dirichlet-multinomial sampling of the
  cell's own composition and no bootstrap over cells. The cell-type interval is the 2.5–97.5 % quantile of that type
  mean across the 200 draws; it reflects model-parameter uncertainty only, not cell sampling. The leave-one-type-out
  reference is formed the same way from the same draws.
- **10.2 and 10.7 rule amended (Karin):** "intervals do not overlap" is replaced by **P(TV > δ) ≥ 0.95**, where TV is
  computed per draw between the cell type and the rest (10.2) or between the two types (10.7), from the same draws, and
  P is the fraction of the 200 draws with TV > δ. The amended call is: point TV > δ **and** P(TV > δ) ≥ 0.95 **and** the
  read/cell condition (unchanged). δ stays the 99th
  percentile of within-type split-half TV (computed first, from point estimates), and mouse-half replication with the
  permuted null is unchanged. In the null permutations the same rule is applied.
- The smoke-test output (job 22597339, 4 draws) is not read before this note is committed; it is used only to check that
  the code runs.

### Amendment 6, note 3 (2026-11-01, before any check 10 output was read, including the smoke test)
- **Mouse bootstrap added to every draw (Karin).** In each of the 200 draws, together with the joint q(φ), q(ψ) draw, each
  cell type's mice are resampled with replacement (within type; the "other cells" group likewise); the type's ψ is the
  mean of ρ̃ over the cells of the resampled mice, each cell weighted by how often its mouse was drawn. The
  leave-one-type-out reference uses the same draw's weights for all other cells.
- **Applies to:** cell-type ψ intervals; P(TV > δ) in 10.2 and 10.7 (per-draw TV from these combined draws); 10.3
  coverage. Point estimates are unchanged (fit's φ and ψ, all cells, no resampling).
- **Coverage reported both ways:** parameter-only intervals (q(φ), q(ψ) draws, no bootstrap, as in note 2) and
  parameter + mouse-bootstrap intervals, side by side, so the difference is visible.
- **Unchanged:** δ (99th percentile of within-type split-half TV, point estimates, computed first), the read/cell
  condition, mouse-half replication and its permuted null (whose draws also use the combined resampling, within the half).
- A type observed in one mouse has no between-mouse variation in this bootstrap; such types are flagged in the output.
- The smoke test (job 22597357, 4 draws) will be cancelled and rerun on this code; its earlier output is not read.
- **Note 3 addendum (2026-11-01, before any check 10 output was read):** a cell type is **eligible for specificity calls**
  (10.2 and 10.7, all stages: full data, mouse halves, null permutations, per-seed call sets) only if its TRAIN cells
  come from **≥ 2 mice** and it has **≥ 1 mouse in each of the two mouse halves** (A and B) used for 10.2a. Ineligible
  types still get ψ estimates and intervals in the catalog (flagged), but no calls; a pair in 10.7 needs both types
  eligible. Reported: the number of ineligible types and their TRAIN cells, per dataset.
- **Note 1 update, benchmark scope (Karin, 2026-11-01; check 10 rules unchanged):** 10.6 benchmarks LeafletFA against
  **scQuint and the LeafCutter cell type × event catalog only**. LeafletSC (the original Leaflet) is dropped. LeafletFA
  fit (TMS and atlas size) and E-step timings as before.
- **Note 1, naming (Karin, 2026-11-01; no change to any computation):** the Allen labels of check 10 (cluster label without
  its number, 70 groups, as in SpliceVI §111) are called **"cell groups (cluster names)"** in all text, not subclasses.
  The official Allen `subclass_label` (42 subclasses; each cluster maps to one) is added as an extra column of the catalog
  export and the portal (`c10_allen_subclass.py`), after the run.

### Amendment 6, note 4 (2026-11-02, bug fix; before any 10.4 Ptprc-direction or Nrxn result exists)
- **Bug:** `c10_catalog.py known_biology()` did not implement two parts of 10.4 as stated: the *Ptprc* direction
  confirmation in TEST pseudobulk, and the *Nrxn1–3* SS4 check (Allen) was never coded. Also, its lymphoid keyword list
  included NK cells; the stated rule is "a lymphoid type (B or T cells)". Karin approved coding exactly the stated rules as
  a documented bug fix (2026-11-02). No other rule changes. Script: `scripts/validation/c10_known_biology_fix.py`, which
  reads the saved seed-0 check 10 outputs (`c10/<ds>_s0_estimates.npz`) and the TEST counts; no refit, no new draws.
- ***Ptprc* (TMS), as stated:** a (lymphoid type, *Ptprc* ATSE) flagged specific, lymphoid = type name contains "B CELL"
  or "T CELL" (NK excluded per the stated rule). Direction confirmed in TEST if, at the call's lead junction (largest
  |ψ_t − ψ_−t| in the event, as in 10.2a), TEST pseudobulk ψ of the type minus TEST pseudobulk ψ of all other TEST cells
  has the same sign. Pass: ≥ 1 flagged lymphoid (type, ATSE) confirmed. TEST reads at the event reported per call.
- **SS4 exons, from annotation only (identified 2026-11-02 before any usage was read):** the single 30-nt exon of each
  gene in GENCODE vM19 (`gencode.vM19/genes/genes.gtf`): *Nrxn1* chr17:90,992,115–90,992,144 (−), *Nrxn2*
  chr19:6,448,255–6,448,284 (+), *Nrxn3* chr12:88,850,279–88,850,308 (+). The SS4 ATSE = the catalog ATSE containing a
  junction that ends at the exon's start − 1 or starts at its end + 1 (an inclusion junction), or that spans the exon
  (skipping junction). If none or more than one ATSE qualifies, reported as such.
- ***Nrxn1–3* (Allen), as stated:** neuronal cell groups = groups whose TRAIN cells' clusters are majority
  `class_label` Glutamatergic or GABAergic (Allen metadata); a pair of neuronal groups "differs" if the point TV between
  their SS4-ATSE ψ exceeds δ (0.10) **and** at ≥ 1 junction their 95 % intervals (the check 10 primary intervals,
  parameter + mouse bootstrap) do not overlap. Pass per gene: ≥ 1 such pair (i.e. at least two groups differ). Reported
  per gene: number of qualifying pairs; also, descriptively, the count restricted to groups with ≥ 20 TRAIN reads and ≥ 3
  covering cells at the ATSE (the read/cell condition of 10.2, not part of the 10.4 rule).

### Amendment 6, note 5 (2026-11-03, bug fix; before any Nrxn3 usage was read)
- **Bug:** note 4's SS4 ATSE rule said a junction "starts at its end + 1". Catalog junction ids use start = the upstream exon's
  last base and end = the downstream exon's first base − 1 (e.g. *Nrxn2* `chr19_6448284_6450527_+` for an exon ending at
  6,448,284), so that clause could never match. *Nrxn1* and *Nrxn2* were still found through their other junctions; *Nrxn3* was
  reported "0 ATSEs qualify; not testable", which was wrong: `chr12_88850308_88853100_+` and `chr12_88850308_89187021_+` start at
  its SS4 exon. **Corrected rule:** a junction that ends at the exon's start − 1, or starts at the exon's end, or spans the exon.
  Exons and all other parts of note 4 are unchanged. *Ptprc* is unaffected and is not rerun.
- **Limit stated before the result:** the *Nrxn3* catalog has no junction ending at the exon's start − 1 and none spanning the exon, so
  its ATSE (if one qualifies) compares the exon's downstream splice choices, not inclusion against skipping. Reported as such.

### Amendment 6, note 6 (2026-11-11, recorded correction of an annotation error; written before any SS4 row below was computed)
- **Error:** note 4 identified "the SS4 alternative exon" as "the single 30-nt exon of each gene in GENCODE vM19". SS4 is a
  90-nt exon (30 residues) in the sixth LNS domain. The 30-nt exons of note 4 are other sites:
  - *Nrxn1* chr17:90,992,115–90,992,144 (`atse_2`) is one of three small early exons of Nrxn1α (18 / 30 / 12 nt, exons 2–4 of the
    α transcripts; residues ~259–274 of UniProt Q9CS84, after EGF-like 1). Consistent with SS#1 (Treutlein et al. 2014:
    "Nrxn1α exons 3a and 4 (SS#1)"); the SS number is not verified.
  - *Nrxn2* chr19:6,448,255–284 and *Nrxn3* chr12:88,850,279–308: the 30-nt exons chosen by the same rule. Not SS4 (SS4 exons
    below); their site is not identified here.
  - Found 2026-11-11 while drawing the Application Note figure: the SpliceVI Fig 2 SS4 exon (below) translates to residues
    1248–1277 of Q9CS84, inside the 30-residue variable segment 1247–1276 in Laminin G-like 6; the note 4 exon does not.
- **Treatment (Karin 2026-11-11):** this is a fix of the exon annotation, not a change of rule after results.
  - The note 4 / note 5 rows stay in the report with their verdicts unchanged, relabeled as "early Nrxn site (note 4 30-nt
    exon; not SS4)".
  - New 10.4 rows are added for the true SS4, by the same 10.4 rule (neuronal groups; a pair differs if point TV between their
    ψ at the ATSE > δ = 0.10 **and** their check 10 primary 95 % intervals do not overlap at ≥ 1 junction; pass per gene:
    ≥ 1 such pair; the read/cell-restricted count reported descriptively).
- **SS4 exons and ATSEs (SpliceVI STAGE5 §111 event list, fixed 2026-10-06, before any LeafletFA value of these events was
  read; GENCODE vM19):**
  - *Nrxn1*: chr17:90,163,792–90,163,881 (90 nt), `ENSMUSG00000024109.18_atse_7`; inclusion `chr17_90162433_90163791_-`,
    skipping `chr17_90162433_90208322_-`.
  - *Nrxn2*: chr19:6,513,716–6,513,805 (90 nt), `ENSMUSG00000033768.16_atse_6`; inclusion `chr19_6509949_6513715_+`,
    skipping `chr19_6509949_6516922_+`.
  - *Nrxn3*: chr12:90,199,147–90,199,236 (90 nt), `ENSMUSG00000066392.11_atse_7`; inclusion `chr12_90169091_90199146_+`,
    skipping `chr12_90169091_90204510_+`.
  - The SS4 ATSE is the §111 ATSE (it holds both junctions). Cross-check: the note 5 junction rule (ends at the exon's
    start − 1, starts at its end, or spans it) is also run and any additional ATSE it finds is reported, not scored.
- **Code:** `c10_known_biology_fix.py nrxn_ss4` (sbatch), reading the saved seed-0 check 10 outputs; no refit, no new draws.
  Output `tsv/c10_Allen_s0_nrxn_pairs_ss4.tsv`, key `nrxn_ss4` in `tsv/c10_known_biology_fix.json` (the note 4 / note 5
  `nrxn` key is kept unchanged).
- Direction (descriptive, not part of the rule): Nguyen et al. 2016 (mouse) — SS4 skipped in excitatory pyramidal neurons,
  included in PV and SST interneurons.

### Record, LeafCutter descriptive control for 10.2 (Karin 2026-11-05; written before any LeafCutter run of this control)
Descriptive. The pre-registered verdict (10.2 fails, Application Note) stands and this cannot change it. (ii) only; (i) is not run.
- **Types:** those eligible in check 10 with ≥ 4 mice in half A and ≥ 4 in half B (TMS 100, Allen 41; 141). The mouse column is `mouse.id` (Allen: `external_donor_name_label`). Halves: `mouse_halves` (10.2a, seed 0).
- **Run:** per type, half-A TRAIN cells. One pseudobulk sample per mouse for the type (≥ 1 cell) and one per mouse for all other cells, summed junction counts; ATSEs as clusters, so events match LeafletFA's. `leafcutter-ds`, baseline = rest, `--min_samples_per_intron 4 --min_samples_per_group 4 --num_threads 4`, other settings default, as in check 9b.
- **Call rule (fixed here):** status Success and `p.adjust` < 0.05 and ½ Σ |deltapsi| over the ATSE's tested introns > δ (0.10, as 10.2), and ≥ 20 half-A reads and ≥ 3 half-A cells of the type at the ATSE (as 10.2). Lead junction: the largest |deltapsi|; sign from it.
- **Confirmation (ii):** half B raw pseudobulk TV between the type and the rest > δ/2 with the same sign at the lead junction, same function as 10.2.
- **Nulls:** the same calls, labels permuted among half B's cells (a) within each mouse, seeds 1000 to 1019, as 10.2b, and (b) ignoring mouse, seeds 2000 to 2004, as `c10_null_diag.py`.
- **Report:** per atlas, LeafCutter calls, real, null (a), null (b), real minus null (a), with type-bootstrap intervals (2,000), beside LeafletFA restricted to the same types, including LeafletFA's null (b) on those types.
- **Cost:** ceiling 200 CPU-h. A watcher cancels the array at the ceiling and the partial state is reported. One smoke task per atlas first, to set memory (peak + 30 %) and check that outputs parse; only status and cost are read from it.
- **Fixed and not tuned after seeing results:** call rule, mouse minimum, δ.

## Amendment 7 (approved by Karin 2026-11-06, 30 CPU-h cap; appended unchanged before any code ran)

# Amendment 7 (DRAFT, 2026-11-02; not approved, not run) — predictive and recalibrated cell-type intervals

Status: draft for Karin's approval. Nothing below has been computed. Once approved, this text is appended unchanged to
`VALIDATION_PREREG.md` before any code runs. The check 10 results stand: 10.3 is reported as is, and no 10.1, 10.2 or 10.7
result or verdict changes.

**Question.** The check 10 intervals covered TEST pseudobulk ψ in 14–35 % of junctions against a nominal 95 %. Do they
reach nominal coverage when the interval also includes (a) the sampling noise of the pseudobulk being compared and (b) a
model-error term estimated on held-out mice?

**What the interval covers.** Both (a) and (a)+(b) are *predictive* intervals for the pseudobulk ψ of new cells of the type,
at a given read depth. They are not intervals for the true cell-type ψ. The paper text and the catalog columns will say so.

**Fixed inputs (unchanged from check 10).**
- Seed-0 fits, TRAIN estimator and point estimates as in check 10.
- The 200 combined draws, regenerated with the check 10 seeds: q(φ), q(ψ) plus the mouse bootstrap of note 3. As a
  variant, the parameter-only draws of note 2 are also used.
- Units: the 10.3 units, i.e. (type, junction) with ≥ 100 TEST reads and ≥ 10 covering TEST cells at the ATSE. TEST read
  bins are 100–499 and ≥ 500.
- Mouse halves A/B: the 10.2a split (seed 0; Allen donors likewise), applied to TEST cells by their `mouse.id`.
- In each half, a unit is used only if it meets the 10.3 read and cell rule within that half's TEST cells.

## (a) Pseudobulk sampling noise added to the parameter interval
- For each unit, draw s ∈ 1..200 and replicate r ∈ 1..5:
  - Take the draw's type ψ_t^(s) (the type mean of ρ̃ in that draw).
  - For every TEST cell c of the type in the comparison set, simulate its junction counts at that ATSE from
    **DM(N_ce, γ_e ψ_t^(s))**, using the cell's observed TEST read total N_ce and the fitted γ_e.
  - Sum over cells and form the simulated pseudobulk ψ.
- Interval (a): 2.5–97.5 % of the 1,000 simulated values.
- Sensitivity variant, reported beside it: a **binomial** at the pooled read count, which ignores between-cell
  overdispersion.
- No parameters are estimated from TEST, so (a) is evaluated on all TEST cells and also within each mouse half.

## (b) Model-error term, estimated on one mouse half and evaluated on the other
- For each unit in the calibration half, take the mean m and variance v of its (a) predictive distribution.
- Model-error variance: τ²_{d,b}, one per dataset d and TEST read bin b. It is the smallest τ² on the grid
  {0, 1e-5 · 2^i, i = 0..20} for which m ± 1.96 √(v + τ²) covers the calibration half's TEST pseudobulk ψ in ≥ 95 % of
  that bin's junctions.
- Interval (a)+(b): m ± 1.96 √(v + τ²_{d,b}), clipped to [0, 1], applied to the **other** half's units.
- Both directions are run: calibrate on A and evaluate on B, then calibrate on B and evaluate on A.

## Reported
- Coverage by dataset × TEST read bin for:
  - the check 10 interval (as in 10.3)
  - (a) DM
  - (a) binomial
  - (a)+(b), in each direction
- The same table for the parameter-only draws.
- Median interval width for each method, and τ²_{d,b} with √τ².
- Coverage of (a)+(b) by number of mice of the type (1–2, 3–5, ≥ 6) and by TRAIN reads (< 1,000, ≥ 1,000), descriptive.

## Pass (written before any run)
- **Threshold:** coverage within 0.05 of nominal, i.e. in [0.90, 1.00], at ≥ 100 TEST reads.
- **(a)+(b) passes** if this holds in **both** read bins (100–499, ≥ 500), **both** datasets and **both** directions,
  using the primary (combined) draws and the DM version of (a).
- (a) alone is checked against the same threshold and reported as pass or fail. It is not a gate.

## Readings (stated now)
- **(a) alone passes:** the under-coverage is explained by comparing a parameter interval to a noisy pseudobulk. The
  catalog ships the (a) interval, labeled as predictive at the stated read depth.
- **(a)+(b) passes but (a) fails:** the catalog ships the (a)+(b) interval. The text reports τ² as the model-error term
  and says it was estimated on held-out mice.
- **Both fail:** the catalog ships the check 10 interval, labeled "model-based, not calibrated", with the measured
  coverage. The paper makes no interval-coverage claim.
- In every case the paper does not claim calibrated intervals for the true cell-type ψ (Karin, 2026-11-02).

## Limits stated in advance
- TEST and TRAIN share mice (per-cell split). The mouse halves separate calibration from evaluation for (b) only.
  Both halves' estimates still use all TRAIN cells.
- Junctions of one ATSE and units of one type are not independent, so coverage is reported without a CI.

**Cost and code.**
- CPU only, from the fit checkpoints and TEST counts; no refit. About 1 h per dataset (same draw cost as check 10,
  without the null permutations).
- Code: `scripts/validation/c12_intervals.py` (to be written after approval). Results: `tsv/c12_*` and section
  "Check 12" in VALIDATION_REPORT.md.

### Amendment 7, note 1 (Karin 2026-11-06; written before any check 12 coverage was read)
- **Option 1 adopted:** one replicate per draw (R = 1) for both the DM and the binomial version of (a). Each unit's (a)
  predictive distribution has 200 simulated values (one per draw) instead of 1,000; interval (a) is their 2.5–97.5 %.
- Everything else in amendment 7 is unchanged: units, mouse halves, τ² grid, both directions, pass rule, readings.
- Projected cost about 17 CPU-h (under the 30 CPU-h cap). Results by Tue 10 Nov 2026 or check 12 stays out.
- **Order (Karin):** check 13 (below) runs first. If check 13's pooled interval comes within 0.05 of nominal at ≥ 100
  TEST reads, Karin is told before check 12 is started.
- The smoke runs (jobs 22780809, 22780810) only timed the code; no coverage from them was read.

## Amendment 8 (Karin 2026-11-06; written before any computation) — check 13: a group-pooled cell-type estimate

**Question:** does pooling a cell type's reads behind one program prior remove the shrinkage of large between-type
differences found in the browser-gap diagnosis, without losing check 10.1's accuracy, and does its Beta interval cover
held-out data? Read-only: saved seed-0 fits, TMS and Allen; no refit; CPU only.

**Three estimates of ψ_tj** (type t, junction j in ATSE e), all from TRAIN cells of t:
- **current:** the check 10 point estimate, mean over t's cells of ρ̃_cj = (γ_e ρ_cj + y_cj) / (γ_e + N_ce), ρ = φψ.
- **covered-cells-only:** the same mean over only t's cells with N_ce > 0. If t has no covered TRAIN cell at e, the
  program prior ρ̄_tj (below) is used, which equals current there.
- **group-pooled:** ψ_tj = (γ_e ρ̄_tj + Σ y_cj) / (γ_e + Σ N_ce), with ρ̄_tj = mean over t's TRAIN cells of (φ_c ψ)_j,
  Σ over t's TRAIN cells, γ_e the fitted concentration (`bb_conc_vec`). **Interval:** 2.5–97.5 % of
  Beta(γ_e ρ̄_tj + Σ y_cj, γ_e (1 − ρ̄_tj) + Σ N_ce − Σ y_cj).

**Readouts.**
1. **Browser-gap table** (same code path, pairs and filters as `c10_browser_gap_rank.py`: the browser example pair plus
   20 random eligible pairs, seed 7; ATSEs with ≥ 20 TRAIN reads and ≥ 3 covering TRAIN cells in both types; lead
   junction = argmax |Δψ| of the estimate being ranked). Each estimate ranks its own list. Bins: top 100, 101–1,000, all.
   Per bin, on rows with ≥ 20 TEST reads and ≥ 3 covering TEST cells in both types: median |Δψ − Δψ_TEST|, median
   |Δψ| − |Δψ_TEST|, sign agreement with TEST; also median |Δψ − Δψ_TRAIN raw| on all rows. Raw TRAIN ranked by itself
   is reported beside them.
2. **10.1-style accuracy:** the 10.1 units and error (TV vs TEST pseudobulk) and bins (type size; TRAIN reads 0, 1–19,
   20–99, ≥ 100). Mean error per estimate and raw; paired differences (estimate − raw) and (pooled − current) with
   95 % bootstrap CIs over cell types (1,000 resamples, seed 0, as in 10.1).
3. **Interval coverage of the pooled Beta interval:** the 10.3 units (≥ 100 TEST reads, ≥ 10 covering TEST cells),
   per junction, fraction of TEST pseudobulk ψ inside the interval, by TEST read bin (100–499, ≥ 500); also by TRAIN read
   bin (descriptive). Median interval width. Nominal 0.95. The check 10 coverage (10.3) is quoted beside it.

**Reading (fixed now, Karin).**
- **Pooled improves on current** if both hold:
  (a) its top-100 error vs TEST (median over the 20 random pairs of the per-pair median |Δψ − Δψ_TEST|, each estimate
  ranked by itself) is lower than current's **in both atlases**; and
  (b) its overall 10.1 accuracy is no worse than current's: the 95 % CI of the overall paired difference
  (pooled − current) is not entirely above 0, in both atlases. *(This is my reading of "no worse than current, within
  the 10.1 interval"; stated here before any result.)*
- The example pair is reported but is not part of the reading (it was chosen, the 20 are random).
- Covered-cells-only is reported on the same readouts; no pass rule (it is the browser alternative Karin asked to test).
- **Coverage is descriptive.** If the pooled interval is within 0.05 of nominal ([0.90, 1.00]) in both TEST read bins
  ≥ 100 in both atlases, Karin is told before check 12 starts.
- If pooled improves on current, the browser and the LeafletFA atlas switch to the pooled estimate only after Karin
  confirms; nothing is switched by this check.

**Limits stated in advance.**
- γ_e was fitted as per-cell overdispersion; here it is reused as the weight of the type's prior. That is a modeling
  choice, not a fitted pooling weight.
- The Beta interval treats a type's reads as draws from one composition. It ignores variation between cells and mice, so
  it can be narrow at high depth.
- TEST and TRAIN share mice (per-cell split), as in check 10.

**Cost and code.** About 10–25 min per atlas on 2 CPUs (the browser-gap jobs took 10 and 22 min, peak 8 and 20 GB).
Code: `scripts/validation/c13_pooled.py`. Results: `$V/c13_pooled/`, section "Check 13" in VALIDATION_REPORT.md.

### Amendment 7, note 2 and amendment 8, note 1 (Karin 2026-11-06; before any check 12 or check 13 output was read)
- **Check 12 no longer waits for check 13 (Karin).** Reason: the same pooling idea failed in SpliceVI (§196), because summing
  reads across cells ignores between-cell variation and makes the group posterior too narrow. Check 12 (option 1) is
  started now, independently of check 13; note 1's "check 13 first, tell Karin before check 12" is withdrawn.
- **Check 13 coverage is descriptive only** and has no consequence for check 12 or for any interval shipped. The
  point-estimate comparison and its reading are unchanged.
- Check 13 jobs (22814975, 22814976) were already running when this was written; none of their output had been read.

## Amendment 9 (Karin 2026-11-13; written and committed before any code ran) — check 14: simple-shrinkage comparators for 10.1

**Step 0 (done before writing this).** No earlier check computed a simple-shrinkage comparator. Check 10 compares
LeafletFA with raw TRAIN pseudobulk only (population ψ fallback at 0 reads); check 13 (amendment 8) shrinks toward the
program prior ρ̄ = mean of φψ, which is a LeafletFA quantity, not simple shrinkage; the browser-gap scripts decompose
LeafletFA's own shrinkage. So check 14 runs.

**Question:** does LeafletFA's gain over raw in check 10.1 exceed what simple shrinkage gives?

**Data and estimates.** Read-only; the 10.1 units, TRAIN and TEST counts of the seed-0 data split (per-cell 70/30); no
refit; CPU only. For type t, junction j in ATSE e (m_e junctions), with y_tj, n_te the TRAIN pseudobulk junction and
ATSE reads of t:
- **LeafletFA:** the saved check 10 point estimate (its 10.1 per-unit error is read from `c10/<ds>_s0_accuracy_units.tsv.gz`).
- **Comparator A (pseudocount):** ψ_tj = (y_tj + 1/m_e) / (n_te + 1): one read spread evenly over the junctions of e.
  At 0 TRAIN reads this is uniform 1/m_e (no population fallback; A is applied as defined).
- **Comparator B (shrinkage to the across-type mean):** ψ_tj = (y_tj + c · p_j) / (n_te + c), with p_j the pooled
  TRAIN ψ over all TRAIN cells (all cell types, the same population ψ used as raw's fallback). At 0 reads it equals p_j.
- **Choice of c (TRAIN only; TEST never used to choose anything):** one c per atlas. Inner split of TRAIN cells, 70/30
  within each cell type × mouse (seed 1; rounding gives the inner-test part ⌊0.3 n⌋ of each stratum). On the inner split,
  p_j and pseudobulks come from the inner-train part; units follow the 10.1 rule applied to the inner-test part (types
  with ≥ 10 inner-train and ≥ 5 inner-test cells; ≥ 20 inner-test reads and ≥ 3 covered inner-test cells at e); error =
  TV vs the inner-test pseudobulk; c = the grid value with the lowest mean error over units. Grid, fixed now:
  {0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}. The chosen c is then applied to the full TRAIN pseudobulk and
  full-TRAIN p_j. The inner-split error curve is reported. If the minimum is at a grid end, that is reported as is.
- **Check on alignment:** raw's error is recomputed from the counts and must match the saved 10.1 `err_raw` per unit
  (max |difference| < 1e-6), else the run stops.

**Readouts.** Same units, same error (TV vs TEST pseudobulk), same bins (TRAIN cells per type: 10–49, 50–199, 200–999,
≥ 1,000; TRAIN reads at the event: 0, 1–19, 20–99, ≥ 100) and the same bootstrap as 10.1 (cell types resampled,
1,000 resamples, seed 0; ratio of summed differences to summed units). Reported, overall and per bin, both atlases: mean
error of LeafletFA, raw, A, B; paired differences LeafletFA − A and LeafletFA − B with 95 % CI; LeafletFA − raw
quoted from 10.1 for reference.

**Reading (fixed now, Karin).**
- **"The model adds beyond shrinkage"** only if LeafletFA − B is below 0 with the 95 % CI excluding 0, **overall, in both
  atlases.**
- Otherwise the result is reported as is, and the text says smoothing accounts for the gain.
- LeafletFA − A and the per-bin rows are reported; they do not enter the reading.
- The result is kept whichever way it goes. No figure changes.

**Limits stated in advance.** TEST cells are new cells of the same mice (per-cell split). B's c is one number per
atlas; per-ATSE or per-type shrinkage strengths are not tried. The comparators borrow only across cell types at the
same event; LeafletFA also borrows across events through the programs.

**Cost.** Estimated < 2 CPU-h for both atlases (counts only, no posterior draws); one CPU job per atlas.

### Amendment 9, note 1 (Karin 2026-11-13; written after the check 14 result above, before any reference level was computed)
Amendment 9 had already been committed and run when this request came, so the reference levels are added here as a
dated note, not inside the original amendment. They are **reference levels, not tests**: no pass rule, no effect on
the check 14 reading. Same units and bins as 10.1; per-bin means with the 10.1 bootstrap over cell types (1,000,
seed 0) for display.
- **Upper reference, "no cell-type information":** per unit, TV between the across-type pooled TRAIN ψ (p_j over all
  TRAIN cells, the ψ raw falls back to at 0 reads; B's limit as c → ∞) and the TEST pseudobulk.
- **Lower reference, "noise floor" — method chosen now: split halves of the held-out cells, scaled to full depth.**
  For each type, its TEST cells are split at random into two halves (10 splits, seeds 0–9; split by cell, not by
  mouse). Per unit and split, with ψ^A, ψ^B the two halves' pseudobulks and n_A, n_B their reads at the ATSE:
  floor = TV(ψ^A, ψ^B) · √(n_A n_B) / (n_A + n_B). The factor converts the half-vs-half difference to the expected
  error of a perfect estimate against the full TEST pseudobulk, if the error scales like its standard deviation
  (Var(ψ^A − ψ^B) ∝ 1/n_A + 1/n_B vs Var(ψ^TEST − ψ) ∝ 1/(n_A + n_B)); equal halves give a factor of ½. Splits with
  0 reads in either half are skipped; the unit's floor is the mean over the remaining splits; units with none are
  left out of the floor mean, and their number is reported.
- **Limits stated now:** the scaling is an approximation (TV is not exactly proportional to the SD at very low depth);
  splitting cells keeps between-cell overdispersion in the floor, but reads within a cell are not independent, so
  the floor is approximate. The simulation alternative (sampling counts from ψ^TEST at its depth) was not chosen
  because a multinomial draw ignores between-cell overdispersion and would understate the floor.

## Amendment 10 (Karin 2026-11-13; written and committed before any code ran) — check 15: semi-synthetic recovery of cell-type ψ and planted differences vs depth

**Purpose:** a supplementary figure showing whether LeafletFA recovers known cell-type ψ, and known differences, as a
function of read depth. Descriptive: **no pass rule; every result is reported whichever way it goes.** The real-data
checks 10.2 and 10.7 (failed) stay as reported; this check does not replace them.

**Base data.** TMS TRAIN (54,694 cells): real cells, real labels (`tissue_cell_type`; types with ≥ 10 TRAIN cells, as
check 10; other cells are simulated but not scored), real mice, real per-cell ATSE totals N_ce, the shared event list.

**Truth ψ (not from a LeafletFA fit).**
- Well covered (t, e): ≥ 100 real TRAIN reads and ≥ 10 TRAIN cells with reads at e (the check 12 depth rule).
- Truth ψ_t,e = real raw TRAIN pseudobulk where (t, e) is well covered.
- Elsewhere ψ_t,e is copied from one well-covered type at e, chosen at random (seed 11). This keeps between-type
  differences real at every scored pair, without shrinking toward the all-cells average, which would favor comparator B.
  The source (own vs copied) is recorded, and results are also reported separately for own-truth pairs.
- ATSEs with no well-covered type: truth = pooled TRAIN ψ; simulated, not scored. Unscored cells (types < 10 cells) use
  the pooled ψ.

**Overdispersion (not from a LeafletFA fit).** Per ATSE, γ_e by method of moments on the real counts:
- Cells in well-covered types with N_ce ≥ 2; S_c = Σ_j (y_cj − N_ce ψ_t,j)² / ψ_t,j over junctions with ψ_t,j > 0
  (m_c of them).
- Under DM, E[S_c] = (m_c − 1) N_ce (N_ce + γ) / (1 + γ). Solving Σ_c gives γ_e = (S2 − X) / (X − S1), with X = Σ S_c,
  S1 = Σ (m_c − 1) N_ce, S2 = Σ (m_c − 1) N_ce².
- Clipped to [0.5, 10⁴]. ATSEs without a well-covered type take the median γ_e.
- ψ is estimated from the same cells (a small bias, stated). The distribution is reported next to the production fit's
  γ_e (descriptive).

**Part 2, planted differences (one deviation from the request, stated before any run).** The request was a shift between
two groups of cell types. With real truth ψ, two groups of types already differ at almost every ATSE, so there are no
clean nulls. Instead:
- Every type's cells are split at random into halves A and B within type × mouse (seed 10).
- A random 30 % of scored ATSEs (seed 12) are planted, with δ ∈ {0.1, 0.3, 0.5} in equal thirds (random). In every
  type's B half, usage δ moves from the type's largest-ψ junction to the ATSE's target junction (the one with the
  lowest pooled ψ).
- The realized shift is min(δ, ψ_from); it is recorded per (t, e) and used as the truth difference.
- Null ATSEs: the A and B truths are identical. The real cell-type structure is unchanged in A.
- **Karin to confirm this design before the GPU fits run.**

**Counts.** Per depth f ∈ {1, 0.3, 0.1}: N′_ce ~ Binomial(N_ce, f) (seeds 21, 22, 23); then
y_c ~ DirichletMultinomial(N′_ce, γ_e ψ_{group(c), e}), with group = type × half (`vlib.simulate_fast` with one-hot φ).
The same truth is used at all depths.

**Fits (GPU).** Production recipe unchanged: `run_leafletfa_baseline.py --K 20 --seed 0 --phi-dtype float64 --device cuda`
(defaults = production), one fit per depth, **3 fits**, seed 0 only. LeafletFA sees no labels and no halves.

**Estimates.** Per group (type, or type × half), from the simulated counts:
- LeafletFA: the catalog estimator (mean ρ̃ over the group's cells, the sim fit's γ_e).
- Raw pseudobulk: with pooled fallback.
- Comparator B: c chosen per depth on an inner 70/30 split of the simulated cells, amendment 9 grid; truth never used.

**Measure 1 — recovery of cell-type ψ** (null ATSEs only, where A and B truths are equal; per real type t, halves
pooled):
- Error TV(estimate_t, truth_t) per scored (t, e).
- Bins: simulated TRAIN reads at (t, e) (0, 1–19, 20–99, ≥ 100) and type size (10–49, 50–199, 200–999, ≥ 1,000);
  per depth.
- Mean error for LeafletFA, raw and B. Paired differences LeafletFA − raw and LeafletFA − B with the 10.1 bootstrap
  (types, 1,000, seed 0).
- Also reported on own-truth pairs only.

**Measure 2 — ranking planted vs null differences.** Units = (t, e) over scored ATSEs; label = planted or null.
- LeafletFA score: TV between its t|A and t|B estimates. Raw TV: descriptive.
- **LeafCutter** on pseudobulk: per type, A vs B on per-mouse pseudobulks of each half (`leafcutter_ds`; ≥ 4 samples
  per group per intron, as the LeafCutter control). Score = −log10 p of the cluster.
- Types with ≥ 4 mice having cells in both halves.
- AUROC and precision at recall 0.5, overall, by δ, and by simulated reads bin, per depth.
- Units LeafCutter does not test are reported as a coverage fraction; the AUROC comparison uses units both methods
  score, and LeafletFA alone is also reported on all units.
- CIs by bootstrap over types (1,000, seed 0).

**Limits stated now.**
- Simulated counts follow the model's likelihood family (DM). This favors the model's noise assumptions, though the
  truth ψ and γ_e do not come from LeafletFA.
- There is no within-type heterogeneity beyond DM noise.
- Copied truth breaks the co-variation of ATSEs within a type for those pairs (if anything, against LeafletFA's shared
  patterns).
- TMS only; one seed. Halves are random within type × mouse, so A and B share mice.

**Compute.**
- GPU: 3 fits at about 13 min (full), ≤ 10 min (thinned), check 4 run times; about **0.6 GPU-h**, with b6k allowed.
- CPU: simulation about 1 h. LeafCutter ≤ 3 × 123 type runs, upper bound about 230 core-h (scaled from the 77 core-h
  TMS catalog). Scoring about 3 CPU-h.

### Amendment 10, note 1 (Karin 2026-11-13; written after the simulation ran, before any fit was submitted or any score computed)
Karin approved the halves design and the 3 GPU fits. Facts below are read from `c15_simulate.py` and `$V/c15/truth.npz`.

1. **Source of copied truth.** Where a (type, ATSE) pair is not well covered (< 100 reads or < 10 cells with reads),
   its truth ψ is the raw TRAIN pseudobulk of **one randomly chosen well-covered type at the same ATSE** (uniform draw,
   seed 11; donor stored in `truth.npz["donor"]`). It is **not** the all-cells average. Well-covered types per scored
   ATSE: median 29 (10th–90th percentile 3–85); 4.2 % of scored ATSEs have a single well-covered type, so every type
   copies the same donor there. Cells of types below MIN_TRAIN (unscored) get the pooled TRAIN ψ in both halves.
   Remaining tilt: the pooled ψ that comparator B shrinks toward is computed from simulated data whose truths are
   draws from the well-covered types, so for copied pairs B's target is an average of the same donor values. This is
   why the copied pairs are secondary.
2. **Primary reading for Measure 1** (cell-type ψ against truth) = **own-truth pairs only** (27 % of scored pairs).
   The all-pairs result is reported as secondary. The Measure 1 reading in amendment 10 applies to the own-truth pairs.
3. **Same events, same direction.** The planted set is one global set of 6,996 ATSEs, shifted in the B half of
   **every** scored type. The receiving junction is fixed per ATSE (its lowest pooled-ψ junction), so the direction
   is the same in every type. The donating junction is that type's most-used other junction, so it can differ between
   types. Size = min(δ, donor ψ): the full δ in 94.3 % of planted (type, ATSE) pairs (95.0 % of own-truth pairs);
   by δ: 0.1 → 99.7 %, 0.3 → 98.1 %, 0.5 → 85.2 %; zero shift in 0.16 %. The applied shift is stored per pair
   (`truth.npz["shift"]`); Measure 2 truth labels use applied shift > 0.
4. **Recipe check.** `run_leafletfa_baseline.py` defaults equal the production recipe and the production TMS fit's
   `run_record.json` settings: 16,200 steps × 8,192 cells, lr 0.025 → 0.1× (`--gamma 0.1`), γ floor 2
   (`--conc-min 2`), sparse likelihood (`dense` False), float64 φ, δ = K (hard-coded, `fixed_dir_conc = K`).
   The driver now also writes these values to a `recipe` field in each `run_record.json`.

## Amendment 11 (Karin 2026-11-14; written and committed before any code ran) — check 15b: check 15 rerun with the design flaw removed

**Reason.** In amendment 10 the same 6,996 ATSEs were shifted, in the same direction, in half B of every cell type.
That made one atlas-wide signal: in every fit one program separated half A from half B (AUROC 1.00), and about half
of the mean φ mass went to that axis. The amendment 10 results stay in `VALIDATION_REPORT.md` as reported; this check
does not replace them. Descriptive: **every result is reported whichever way it goes.**

**Shared with amendment 10, unchanged:** TMS TRAIN cells, labels, mice and real per-cell ATSE totals; truth ψ
construction (seed 11; own where well covered, else copied from a random well-covered type at the ATSE); γ_e by
method of moments; depths f ∈ {1, 0.3, 0.1} by Binomial thinning; DM counts (`vlib.simulate_fast`); fits with the
production recipe (`run_leafletfa_baseline.py --K 20 --seed 0 --phi-dtype float64 --device cuda`, defaults =
production, recipe recorded in `run_record.json`); comparators raw (pooled fallback) and B (c on an inner 70/30 split
of the simulated cells within type × mouse, seed 1, amendment 9 grid; truth never used); TV error; bootstrap over
types (1,000, seed 0). Outputs under `$V/c15b/{acc,diff}/`.

**Own-truth pairs are, by construction, the well-covered ones** (≥ 100 real TRAIN reads and ≥ 10 cells with reads).
That reading tests LeafletFA where raw is expected to be strongest.

### Accuracy arm (primary for cell-type ψ) — no planted shifts
- Counts from the truth ψ of each cell's type only (no halves, no shifts). Thinning seeds 31, 32, 33; DM seeds per depth
  as given to `simulate_fast`.
- Units: every scored (type, ATSE). Error TV(estimate_t, truth_t) for LeafletFA (catalog estimator, the fit's γ_e),
  raw and B.
- **Primary:** own-truth pairs, overall, per depth. Also within own-truth pairs by simulated TRAIN reads at (t, e)
  (0, 1–19, 20–99, ≥ 100) and by type size (10–49, 50–199, 200–999, ≥ 1,000). Secondary: all pairs, same bins.
- **Reading rule (per depth and bin):** Δ = LeafletFA − comparator, 95 % CI by type bootstrap. CI upper < 0 →
  LeafletFA better; CI lower > 0 → LeafletFA worse; otherwise tie.
- **Comparison with amendment 10 (own-truth, overall, per depth):** the amendment 10 gap vs raw (+0.0118 full, +0.0140
  f30, +0.0140 f10) is
  - *gone* if the new Δ raw CI includes 0 or lies below 0;
  - *survives in full* if the new CI lower > 0 and the CI includes the amendment 10 point;
  - *survives in part* if the new CI lower > 0 and the CI lies below the amendment 10 point.
  The ratio new Δ / amendment 10 Δ is reported.

### Differential arm — type-specific shifts
- Halves A/B within type × mouse (seed 10, as amendment 10). Planted ATSEs: the same random 30 % of scored ATSEs and the
  same δ ∈ {0.1, 0.3, 0.5} per ATSE (seed 12, as amendment 10).
- **For each planted ATSE, half B is shifted in a random 20 % of scored types only** (independent per ATSE, seed 13).
  The receiving junction is chosen at random per (type, ATSE) among the junctions other than the donor (seed 13), so
  the direction also differs between types. Donor = that type's largest-ψ junction; applied shift = min(δ, donor ψ),
  recorded per pair. Per type, about 6 % of scored ATSEs are planted.
- Thinning seeds 41, 42, 43.
- **Check before scoring (fixed now):** for each fit, the best single-program AUROC for half B over scored cells,
  max_k max(AUC_k, 1 − AUC_k). The same number is reported for the accuracy-arm fits (no halves in the data; reference).
  - ≤ 0.60: no atlas-wide half signal; Measure 2 read as intended.
  - 0.60–0.75: partial; reported, and Measure 2 read with that caveat.
  - > 0.75: Measure 2 is confounded; numbers reported, not read as type-specific detection.
- **Measure 2:** units (t, e) over scored ATSEs in types with ≥ 4 mice having cells in both halves; label = applied
  shift > 0 (planted ATSEs in unchosen types are nulls). Scores: LeafletFA TV(t|A, t|B); raw TV (descriptive);
  LeafCutter A vs B on per-mouse pseudobulks (check 10 settings), −log10 p of the cluster.
- **Ranking per cell type:** AUROC within each type; reported as the median and mean over types (mean with a type
  bootstrap CI), plus the pooled AUROC and precision at recall 0.5.
- Bins by simulated reads in the type at the ATSE (both halves): **0, 1–19, 20–99, ≥ 100**, reported separately; also by δ.
- LeafCutter: coverage fraction; AUROC on units both methods score; Δ AUROC (LeafletFA − LeafCutter) with a type
  bootstrap CI. CI excludes 0 → that method ranks better on shared units; otherwise tie.
- **Reading at 0 reads:** with no shared signal, LeafletFA should not rank above chance where the type has no reads. CI
  of the per-type mean AUROC including 0.5 → as expected; above 0.5 → reported as borrowing from outside the type's reads.

### Limits (unchanged from amendment 10, plus)
- DM counts favor the model's noise family; no within-type heterogeneity beyond DM; copied truth breaks within-type
  co-variation; TMS only; one seed; A and B share mice.
- The 20 % fraction and random direction are choices made to remove the shared signal, not estimates of real biology.

### Compute
- GPU: 6 fits (3 accuracy + 3 differential), about 13 / 12 / 11 min each by depth: about **1.2 GPU-h**, b6k allowed.
  Approved by Karin 2026-11-14.
- CPU: two simulations about 15 min each (peak 14.3 GB, so 20 GB); scoring about 35 min per depth (peak 10.7 GB, so
  14 GB); LeafCutter 3 × ≤ 132 tasks (peak 1.1 GB, so 2 GB).

## Record, scaling and runtime for the supplement (Karin 2026-11-14; written and committed before any scaling run)
Descriptive cost measurement; no pass rule; reported as measured. GPU approved by Karin for these short runs.

**1. Existing records (no runs).** Production fits, the 10.6 atlas-size run, scQuint 22725257 and the LeafCutter catalog
array 22780123. Sources: job logs, `run_record.json`, sacct.
- Correction found while assembling the records: the production TMS and Allen run records (jobs 22400620, 22400623) are
  resumed jobs (`resumed_from_step` 16,200), so their 10.6 and 31.6 min are projection time, not fit time. The fits ran in
  jobs 22400619 (TMS) and 22400622 (Allen), which crashed while saving, after step 16,199. Fit times come from those logs.

**2. Controlled LeafletFA scaling (GPU).**
- Data: TMS TRAIN, random cell subsets via the driver's `--n-train` (`--subset-seed 0`): 5,000, 10,000, 25,000, 50,000,
  plus the full 54,694.
- Recipe: production (`run_leafletfa_baseline.py --K 20 --seed 0 --phi-dtype float64 --device cuda`), with two
  changes, both stated:
  - **Fixed number of steps = 2,000** (`--fit-steps 2000`), the length of the 10.6 atlas timing run. The lr decay
    spans those 2,000 steps; it does not affect time per step.
  - Minibatch = min(8,192, n): `--fit-batch 5000` at n = 5,000 (all cells per step). Every other size uses 8,192.
- Hardware: all GPU runs on b6k (RTX PRO 6000 Blackwell, `--gres=gpu:b6k:1`), 8 CPUs. A node holds 2 GPUs, so another
  job may share the node; the node and card are recorded per run.
- Measured per run:
  - seconds per step: median over the 500-step checkpoint intervals after step 500;
  - time to 2,000 steps: fit minutes in `run_record.json`, plus whole-job wall time from sacct;
  - peak GPU memory: `torch.cuda.max_memory_allocated`;
  - peak host memory: `ru_maxrss` and sacct MaxRSS.
- No projection or E-step is run.
- **CPU reference:** one run at n = 5,000, same settings, `--device cpu`, 16 CPUs, on the cpu partition. Reported:
  CPU / GPU seconds-per-step ratio at 5,000 cells.
- Full-recipe times (16,200 steps) are taken from item 1, not rerun.

**3. scQuint at the same sizes (CPU, as in 10.6).**
- `c10_benchmark_scquint.py` settings: scQuint VAE, `use_cuda=False`, its default early stopping, seed 20261170, 16 CPUs.
- Subsets: the same cell indices as item 2 (`np.random.default_rng(0).choice` over the same TRAIN row order).
- The full size is job 22725257 (not rerun): 20 epochs to early stop, about 245 s per epoch, training 86.0 min.
- Measured: training time to early stop, epochs, seconds per epoch, peak memory (sacct).
- Wall limit 3 h per size. A run that hits the limit is reported as "> 3 h, stopped".
- LeafCutter: item 1 only.

**4. Output.** `$V/scaling/scaling.tsv` (one table) and `scaling.{pdf,png}`: time and peak memory against cells, built with
the figure theme.

**Memory requests** (measured peak + 30 %):
- GPU runs 16 GB host: TMS full fit peak 9.2 GB (job 22400619).
- CPU LeafletFA 16 GB.
- scQuint 4 GB: peak 1.7 GB (job 22725257).

**Compute:** GPU about 5 × ≤ 3 min plus load, under 0.5 GPU-h. CPU LeafletFA 1 run, ≤ 3 h. scQuint 4 runs, ≤ 3 h wall each.

## Amendment 12 (Karin 2026-11-15; written and committed before any code ran) — check 16: held-out confirmation of between-cell-type calls at a matched number of calls

**Scope (Karin).** Item 2 of the 2026-11-15 proposal only, CPU only. Item 1 (pool reads, then shrink) is check 13's
group-pooled estimate and is not rerun. Items 3 and 4 are not approved. The 10.2 LeafCutter note stays descriptive. No
verdict of 10.2, 10.7, 15 or 15b changes.

**Question.** On real TMS data, when LeafletFA ranks cell type vs rest differences by an evidence-based statistic, do
its top calls confirm in held-out cells at least as often as the same number of top calls from raw pseudobulk and from
LeafCutter, within each coverage bin?

**Separation of cells (Karin's condition: nothing used to make a call is fit on the cells used to confirm it).**
- Mouse halves A/B: the 10.2a split (`mouse_halves`, seed 0 over TRAIN mice). TEST cells take the half of their
  `mouse.id`. TEST cells of mice with no TRAIN cells are not used.
- **Calls:** half-A TRAIN cells only, for all three methods.
- **Model-error term:** check 12's τ² calibrated on half A (`tsv/c12_TMS_tau2.tsv`, combined draws, `calibrated_on = A`):
  0.01024 (100–499 TEST reads) and 0.00256 (≥ 500). It was set on half-A TEST cells, so it is half A only. The
  τ² from B and the shipped interval (set on both halves) are not used.
- **Confirmation:** half-B TEST cells only. They were not in the LeafletFA fit (fit on TRAIN cells), not in τ², and
  not in any call.
- δ = 0.10 (the 10.2 TMS value, from within-type split halves of TRAIN cells; no TEST cell).
- **Stated limit:** the seed-0 fit (programs ψ_k, per-cell φ, γ_e) was trained on TRAIN cells of all 18 mice. That
  includes half-B mice's TRAIN cells, so the confirmation cells share mice with cells the fit saw. Removing this needs
  a refit on half-A mice (GPU, not approved).
- **Recorded fact about 10.2 (no verdict changes):** 10.2a (ii), 10.7 and the LeafCutter control confirmed on half-B
  **TRAIN** cells. For LeafletFA these cells were in the fit. Check 16 confirms on half-B TEST cells instead.

**Cell types and units.**
- Types: the 100 TMS types of the LeafCutter control (`c10_lc/TMS_halfA/types.tsv`: eligible in check 10, ≥ 4 mice
  in each half).
- Contrast: type vs rest. Rest = all other cells of the same half, as in 10.2 and the LeafCutter control. Pairwise
  contrasts (10.7) are not part of this check.
- Universe U: (type, ATSE) with ≥ 1 half-A TRAIN read of the type at the ATSE that is evaluable in half-B TEST. The
  evaluability condition is ≥ 20 half-B TEST reads and ≥ 3 covering half-B TEST cells of the type at the ATSE.
  - Evaluability uses B's coverage, not B's ψ.
  - It is applied identically to every method.
- **Coverage bins:** the type's half-A TRAIN reads at the ATSE: **1–19, 20–99, 100–499, ≥ 500**.

**Ranking statistics, all from half-A TRAIN cells.** Lead junction = argmax |Δψ| of the method's own estimate. The sign
of Δψ there is the call's direction.
- **LeafletFA (primary): z = |Δψ_lead| / √(Var_s(Δψ_lead^(s)) + τ²_b).**
  - Δψ is the check 10 estimator restricted to half-A TRAIN cells: type mean of ρ̃ minus rest mean, point estimate from
    the fit's φ and ψ.
  - Var_s is over 200 combined draws: q(φ), q(ψ) and the mouse bootstrap of amendment 6 note 3, within half A, seed 101
    as in 10.2a.
  - The mouse bootstrap resamples the type's half-A mice, i.e. the per-mouse pseudoreplicates. It carries the variance
    across mice.
  - τ²_b = 0.01024 if half-A TRAIN reads < 500, else 0.00256. Below 100 reads this extrapolates check 12's lowest bin.
    It is also applied by TRAIN reads, where check 12 binned by TEST reads.
- **Raw pseudobulk:** |Δψ_lead|, from half-A TRAIN Σy/ΣN of the type minus the rest.
- **LeafCutter:** the existing half-A runs of the LeafCutter control (`c10_lc/TMS_halfA/<type>/`; settings as recorded
  there; not rerun).
  - Scorable = cluster status Success.
  - Ranked by the cluster's `p`, ties broken by |deltapsi| at the lead intron.
  - Lead and sign from `deltapsi_armT`, as in the control.
- **Descriptive only:** LeafletFA ranked by |Δψ_lead| (point estimate, no uncertainty), to separate the effect of the
  statistic from the estimator.
- Ties: stable sort, then by (type code, ATSE index).

**Matched number of calls.**
- In each bin b, each method calls its top **N_b = min(500, ⌊0.10 · |U_b|⌋)** units of U_b.
- A method with fewer than N_b scorable units in the bin is reported but not compared in that bin (expected for
  LeafCutter at 1–19 reads).
- Sensitivity, descriptive: N_b′ = min(2,000, ⌊0.25 · |U_b|⌋).
- Also descriptive: one pooled ranking over all bins at N = 2,000.

**Confirmation (model-free, half-B TEST cells only).**
- A call is **confirmed** if both hold in half-B TEST raw pseudobulk:
  - TV between the type and the rest > δ/2 = 0.05;
  - the same sign at the call's lead junction.
  - This is 10.2 (ii) applied to half-B TEST cells.
- **Null:** the type labels of half-B TEST cells are permuted within each mouse, 20 permutations (seeds 1000–1019),
  and the same confirmation is recomputed for each method's calls.
- **Excess confirmation** of a method in a bin = mean over its calls of (confirmed − null confirmation rate of that
  call).
- Also reported, descriptive:
  - the confirmed fraction and null;
  - sign agreement alone, as in SpliceVI Fig 4d.
- **Intervals:** bootstrap over the 100 cell types, 2,000 resamples, seed 0. Each method's excess is recomputed on its
  own calls in the resampled types. The difference LeafletFA − comparator is taken within each resample, giving a 95 %
  percentile interval.

**Pass rule (fixed now).**
- **Comparisons:** LeafletFA (z) vs raw pseudobulk and vs LeafCutter, in bins 20–99, 100–499 and ≥ 500. Only bins where
  both methods have ≥ N_b scorable units are compared.
- **Non-inferior:** in every comparison, the lower 95 % bound of (LeafletFA − comparator) excess is > −0.05.
- **Better:** non-inferior, and against both comparators the lower bound is > 0 in at least one of the three bins.
- **Fail:** anything else.
- **Thin data, bin 1–19 (separate reading, "more useful"): supported** if LeafletFA's excess has a lower bound > 0
  **and** (LeafletFA − raw pseudobulk) has a lower bound > 0. LeafCutter's scorable count in this bin is reported.

**Readings (stated now).**
- *Better or non-inferior:* LeafletFA's between-type calls confirm in held-out cells of held-out mice at least as
  often as pseudobulk and LeafCutter at an equal number of calls, within the stated limit on the fit.
- *Fail:* reported as is. Differential splicing stays unclaimed, and no further estimator or statistic variant is
  tried for this question.
- In every case, the use of this result in any text is Karin's decision.

**Other limits stated in advance.**
- Half-B TEST cells are about 15 % of cells, so confirmation is noisy for every method. The per-method null accounts
  for each method's own selection.
- Units of one type and junctions of one ATSE are not independent; the type bootstrap is the only interval.
- LeafCutter's own call rule (p.adjust < 0.05 and δ) is not used. At a matched N, its ranking is by `p`.

**Cost and code.**
- CPU only, from the seed-0 checkpoint and counts; no refit.
- One job: the half-A draws plus confirmation and 20 nulls for three rankings. Expected ≤ 3 h on 4 CPUs.
- A 4-draw smoke run first, to measure peak memory (request = peak + 30 %) and to check that outputs parse. Only cost
  and status are read from it.
- Code: `scripts/validation/c16_ds_heldout.py`. Results: `$V/c16/`, section "Check 16" in VALIDATION_REPORT.md.
- Report on Monday 2026-11-16, whichever way it goes.

### Amendment 12, note 1 (Karin 2026-11-15; written and committed after the check 16 result, before any follow-up code ran) — check 16b: is the advantage explained by concentration in few cell types?
**Why.** In check 16, LeafletFA's top calls came from 17–50 types per bin, against 73–85 for raw pseudobulk and
LeafCutter. If its calls concentrate in a few very different types, they could be easier to confirm. Check 16's
verdict (FAIL) and readings stand. 16b only asks whether the measured differences survive matching on types.

**"Pair" in check 16 terms.** Every check 16 contrast is one cell type vs the rest, so the unit to cap and match on is
the **cell type**. Karin's "per cell-type pair" is applied as "per cell type (type vs rest)".

**Inputs, unchanged from amendment 12.**
- The same units U, bins, rankings, lead junctions, half-A calls, half-B TEST confirmation, within-mouse null (20
  permutations, seeds 1000–1019) and excess measure.
- Type bootstrap: 2,000 resamples, seed 0, over the design's types.
- The full per-unit table (every unit, every method: statistic, scorable, confirmed, null) is regenerated by rerunning
  `c16_ds_heldout.py` with a dump flag, to `$V/c16b/`. Its summary must equal check 16's `summary.tsv` exactly
  (asserted), or nothing is scored.

**Design 1, capped and type-matched.**
- Cap **c = 5 calls per cell type per bin** (N_b = 500 over 100 types).
- For each comparison (LeafletFA z vs raw pseudobulk; LeafletFA z vs LeafCutter) and bin, T_b = the types in which
  both methods have ≥ 5 scorable units in U_b.
- In every type of T_b, each method calls its own top 5 units of that type and bin. Both methods therefore draw the
  same number of calls from the same types.
- Ties are broken as in check 16.

**Design 2, restricted to types where all methods make calls.**
- The check 16 primary call sets (N_b = 500), unchanged.
- In each bin, they are restricted to the types in which every method compared in that bin has ≥ 1 call: LeafletFA z,
  raw pseudobulk, and LeafCutter where it is compared (100–499, ≥ 500).
- Each method's excess is recomputed on its own calls in those types.

**Rule and readings (fixed now), applied to each design separately.**
- The amendment 12 rule unchanged (non-inferior: every compared lower bound > −0.05; better: non-inferior and, against
  both comparators, a lower bound > 0 in ≥ 1 of 20–99, 100–499, ≥ 500), plus the thin-data reading at 1–19.
- **Concentration reading:** the check 16 thin-data advantage over raw is called **not explained by concentration** if,
  in both designs, LeafletFA z − raw has a lower bound > 0 at 1–19 **and** at 20–99. Otherwise it is called
  **at least partly explained by concentration**, stated per design and bin.
- The LeafCutter comparison is read the same way at 100–499 and ≥ 500 (lower bound > 0 in both designs, or not).
- Also reported: |T_b|, calls per method, and the excess per method per bin.

**Cost.** CPU only. The rerun with dump takes about 15 min (check 16: 13.9 min, peak 10.1 GB; request 14 GB). Scoring
takes a few minutes. Code: `c16_ds_heldout.py` (dump flag) and `c16b_type_matched.py`. Results: `$V/c16b/`, section
"Check 16b" in VALIDATION_REPORT.md. Report on Monday 2026-11-16, whichever way it goes.

## Amendment 13 (Karin 2026-11-15; written and committed before any code ran) — check 17: held-out confirmation of pairwise cell-type calls, TMS; and an Allen donor count

**Scope (Karin).**
- CPU only. TMS pairwise check: LeafletFA against raw pseudobulk on all sampled pairs, and LeafCutter on a pre-set
  random subset of 50 pairs.
- The Allen part is a donor count, reported only.
- The design is 16b design 1, applied to pairs. No earlier verdict changes.
- No significance cutoff is tested here, so the browser shows no significance flag.

**Question.** For two TMS cell types, at an equal number of calls per pair, do LeafletFA's top differential ATSEs
confirm in held-out cells of other mice more often than raw pseudobulk's (and LeafCutter's, on the subset), by
coverage bin and by kind of pair?

**Cells and separation (as amendment 12).**
- Calls: half-A TRAIN cells. Confirmation: half-B TEST cells only.
- The halves are the 10.2a split (`mouse_halves`, seed 0); TEST cells take the half of their `mouse.id`.
- δ = 0.10. No τ² is used, because the ranking is by the size of the difference.
- Stated limit: the seed-0 fit saw half-B mice's TRAIN cells.

**Pair sample (fixed now).**
- **Types:** the 100 TMS types of the LeafCutter control (`c10_lc/TMS_halfA/types.tsv`: ≥ 4 mice in each half).
  - Type name = `<tissue>_<cell type>`, split at the first underscore.
- **Classes over the 4,950 unordered pairs of these types:**
  - (i) **same tissue**, different cell type;
  - (ii) **same cell type across tissues** (identical cell-type part, different tissue);
  - (iii) **unrelated** (different tissue and different cell type).
- **Sample:** 100 pairs per class, drawn with `np.random.default_rng(17)` over the class's pairs in sorted order. If a
  class has fewer than 100 pairs, all of them are used.
- **Pair orientation:** the type that sorts first is A.
- **LeafCutter subset:** 50 pairs drawn from the sample with `np.random.default_rng(1717)`, stratified 17 / 17 / 16
  over classes (i) / (ii) / (iii), fewer if a class is short.

**Units and bins.**
- A unit is a (pair, ATSE) with ≥ 1 half-A TRAIN read in each type that is evaluable in half-B TEST: ≥ 20 reads and
  ≥ 3 covering cells in **each** type.
- **Coverage bin:** the smaller of the two types' half-A TRAIN reads at the ATSE: 1–19, 20–99, 100–499, ≥ 500.

**Rankings (half-A TRAIN cells).** Lead junction = argmax |Δψ| of the method's own estimate; its sign gives the
direction.
- **LeafletFA:** |Δψ_lead| between the two types' check 10 estimates restricted to half-A TRAIN cells (point estimates
  from the fit's φ and ψ).
- **Raw pseudobulk:** |Δψ_lead| from half-A TRAIN Σy/ΣN.
- **LeafCutter (subset only):**
  - Input: one pseudobulk sample per half-A mouse for each type (≥ 1 cell), ATSEs as clusters.
  - Run: `leafcutter-ds` with type B as baseline, and `--min_samples_per_intron 4 --min_samples_per_group 4`, as in
    the LeafCutter control.
  - Scorable = Success. Ranked by the cluster `p`, ties broken by |deltapsi| at the lead intron.

**Matched calls (16b design 1).**
- Cap **c = 5 per pair per bin**.
- For each comparison (LeafletFA vs raw on the full sample; LeafletFA vs LeafCutter on the subset) and bin, the pairs
  used are those where both methods have ≥ 5 scorable units in the bin.
- In each such pair, each method takes its own top 5. Ties are broken as in check 16.

**Confirmation and null.**
- **Confirmed:** in half-B TEST raw pseudobulk of the two types, TV > δ/2 = 0.05 and the same sign at the call's lead
  junction.
- **Null:** the two types' labels are permuted among their half-B TEST cells within each mouse, 20 permutations (seeds
  1000–1019).
- **Excess** = confirmed − null, per call, averaged.
- **Intervals:** bootstrap over pairs, 2,000 resamples, seed 0. The difference is taken within each resample. Pairs share
  types, so this interval is approximate (stated).

**Pass rule (fixed now; the amendment 12 rule).**
- **Comparisons:** vs raw on the full sample, and vs LeafCutter on the subset, in bins 20–99, 100–499 and ≥ 500. A
  comparison is made only where ≥ 10 pairs qualify.
- **Non-inferior:** every compared lower bound of (LeafletFA − comparator) excess is > −0.05.
- **Better:** non-inferior, and against each comparator a lower bound > 0 in ≥ 1 of those bins.
- **Fail:** anything else.
- **Thin data (1–19), separate reading:** supported if LeafletFA's excess and (LeafletFA − raw) both have lower bounds > 0.

**Descriptive, reported beside the rule.**
- The same table per pair class (i)–(iii).
- **The rate shown in the browser:** confirmed fraction and excess for LeafletFA and raw by bin and by rank band within
  a pair (ranks 1–5, 6–20, 21–100), on the full sample. Unit selection is as above; no cap within a band.
- Count of qualifying pairs per bin.

**Allen donor count (report only).**
- For each of the 2,278 pairs of the 68 Allen cell groups: the number of donors with ≥ 1 TRAIN cell of **both** groups.
- Also reported: the same count within mouse half A and within half B (the 10.2a donor split), and among TEST cells.
- Reported as a distribution (0, 1, 2–3, ≥ 4 donors) and per pair (TSV).
- No analysis follows from it without Karin.

**Cost and code.**
- `c17_pairs.py` (LeafletFA and raw on all sampled pairs, LeafCutter inputs and collection, scoring): one CPU job,
  ≤ 1 h, memory from the check 16 peak (10.9 GB) + 30 %.
- LeafCutter: a 50-task array with settings as the control, peak 1.1 GB per task there, so 2 GB. Expected about 15–30
  CPU-h; a ceiling of 60 CPU-h, at which the array is cancelled and the partial state reported.
- `c17_allen_donors.py`: one small CPU job.
- Results: `$V/c17/`, section "Check 17" in VALIDATION_REPORT.md. Report on Monday 2026-11-16, whichever way it goes.

## Amendment 14 (Karin 2026-11-16; written and committed before any code ran) — check 18: refit on half the mice, confirm in mice the model never saw (TMS)

**Scope (Karin).**
- One GPU refit of TMS, approved, with the production recipe. Everything else is CPU.
- Check 17's pairwise design and check 16's type-vs-rest design are rerun with calls from half-A mice and confirmation
  only in half-B mice. The model never saw half-B mice.
- No earlier verdict changes.

**Mouse split (fixed now).**
- **Mice:** the 18 TMS mice (`mouse.id`), with all cells of the TMS-only TRAIN and TEST files pooled. The per-cell 70/30
  split is not used in check 18.
- **Candidates:** 10,000 random 9 / 9 splits, `np.random.default_rng(18).permutation` over the sorted mouse ids, first 9
  = half A.
- **Score, compared lexicographically, with mice per type counted over all cells:**
  1. the number of the 100 check 16 types (`c10_lc/TMS_halfA/types.tsv`) with ≥ 4 mice in each half;
  2. the number of TMS types (≥ 10 cells in total) with ≥ 1 mouse in each half;
  3. the same with ≥ 2 mice in each half.
- **Choice:** the highest-scoring split. Ties go to the lowest draw index.
- Reported: the chosen split, its three scores, and the scores of the 10.2a split for comparison.

**Refit (GPU, production recipe).**
- Command: `run_leafletfa_baseline.py` with the seed-0 production settings:
  - K = 20, seed 0, `--fit-steps 16200`, `--fit-batch 8192`, `--fit-lr 0.025`, γ 0.1, `--conc-min 2.0`;
  - `--phi-dtype float64`, sparse, `--device cuda`.
- Input: the junction counts of every half-A cell (TRAIN and TEST rows), as an npz with its cell list. The junction
  order is fixed by the TRAIN file (`--npz-var`). No projection.
- Hardware: b6k (RTX PRO 6000 Blackwell), allowed for LeafletFA, 8 CPUs, 16 GB host (production peak 9.2 GB + 30 %).
- Wall limit 2 h.
- A fit with NaN restarts exhausted, or with non-finite φ, is reported and not scored.

**Calls (half-A cells only) and confirmation (half-B cells only).**
- **LeafletFA:** the check 10 estimator from the half-A fit: type mean of ρ̃ over the type's half-A cells, point
  estimate. Ranked by |Δψ_lead|, as in check 17. Check 16's z is not used (in check 16 it matched the point ranking).
- **Raw pseudobulk:** half-A cells, Σy/ΣN.
- **LeafCutter:** check 17's 50-pair subset, unchanged.
  - Rerun on half-A cells: one pseudobulk per half-A mouse per type (≥ 1 cell).
  - Settings as check 17. A pair whose type lacks 4 half-A mice gives no scorable unit.
- **Confirmation:** raw pseudobulk of **all half-B cells** (TRAIN and TEST): TV > δ/2 = 0.05 and the same sign at the
  call's lead junction.
- **Null:** labels permuted among the compared half-B cells within each mouse, 20 permutations, seeds 1000–1019.
- **Excess** = confirmed − null.
- **Evaluable unit:** ≥ 20 half-B reads and ≥ 3 covering half-B cells in each compared type. For type vs rest, the
  rest needs no minimum.
- **Bins:** reads of the type at the ATSE in half-A cells; for pairs, the smaller of the two types: 1–19, 20–99,
  100–499, ≥ 500.

**Design 18a, pairwise (check 17 design).**
- Check 17's 300 sampled pairs and pair classes, unchanged.
- A unit needs ≥ 1 half-A read in each type.
- Cap 5 per pair per bin; pairs where both methods have ≥ 5 scorable units.
- Comparators: raw on all pairs, LeafCutter on the subset.
- Bootstrap over pairs, 2,000, seed 0. Reported by class as in check 17, plus rank bands.

**Design 18b, type vs rest (check 16 design, matched as 16b design 1).**
- The 100 check 16 types; rest = all other half-A cells.
- Cap 5 per type per bin; types where both methods have ≥ 5 scorable units.
- Comparator: raw pseudobulk only. LeafCutter type-vs-rest is not rerun: about 70 CPU-h, and not in scope.
- Bootstrap over types, 2,000, seed 0.

**Pass rule (check 17's, applied to 18a and 18b separately).**
- Comparisons in bins 20–99, 100–499 and ≥ 500, made where ≥ 10 pairs (types) qualify.
- **Non-inferior:** every compared lower bound of (LeafletFA − comparator) excess is > −0.05.
- **Better:** non-inferior, and against each comparator a lower bound > 0 in ≥ 1 of those bins.
- **Fail:** anything else.
- **Thin data (1–19):** supported if LeafletFA's excess and (LeafletFA − raw) both have lower bounds > 0.

**Readings (stated now).**
- If 18a and 18b keep check 17's / 16b design 1's direction, the stated limit "the model saw cells from the confirming
  mice" no longer applies to them.
- If either fails, the result is reported as it is, beside checks 16–17.

**Cost and code.**
- `c18_split.py` (split and half-A input; CPU, about 10 min).
- The GPU fit: about 0.86 s/step at production was a 16,200-step CPU-checkpointed run, and on b6k it is expected to
  take about 15 min; wall limit 2 h.
- LeafCutter: 50-task array, 18-min cap per task (≤ 60 CPU-h).
- `c18_score.py` (CPU, about 20 min).
- Results: `$V/c18/`, section "Check 18" in VALIDATION_REPORT.md. Report whichever way it goes.
