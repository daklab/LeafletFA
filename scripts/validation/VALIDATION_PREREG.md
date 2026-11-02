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
