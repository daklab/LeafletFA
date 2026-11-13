# LeafletFA validation — report

> **Decisions (Karin, 2026-10-25):** program stability = (b) + (a); aging = option (b), partial
> replication, moved to the Supplement; both drafted on Leaflet-analysis `leafletfa-methods-fixes`
> (77d1953). No extra seeds for the flatter prior. SpliceVI Methods sentence as in check 2 below.

2026-10-25. Pre-registration: `VALIDATION_PREREG.md` (written before any check ran; amendments 1–2 written
before the checks they change). Code: LeafletFA branch `leafletfa-validation`, `scripts/validation/`.
TSVs in `tsv/`; summary figure `plots/validation_summary.{pdf,png}`. GPU used: ~2.9 GPU-h of the ~3
approved. No model default was changed; the one new option (`dir_alpha_min`, prior floor) is off by default.

## Verdict

| Check | Result | Pass/fail |
|---|---|---|
| 0 Which model is the paper's model | Text, figures and current main all use the DM simplex model; 6 text problems, 1 serious (aging replication) | **pass**, text fixes required |
| 1 Held-out likelihood | Beats population, cell-type and K = 1 nulls on TMS and Allen, all CIs > 0; scorer verified | **pass** (stop rule not triggered) |
| 2 K and prior (TMS) | K = 20 within 0.02 kNN of the best (K = 10); K = 30 is worse; higher K adds no used programs | **pass**; "similar at K = 10 and 30" **does not hold** |
| 3 Seed stability | kNN stable (TMS range 0.025, Allen 0.003); centred-ψ replication TMS 3 (need ≥ 4), Allen 1 (need ≥ 3) | kNN **pass**, programs **fail** |
| 4 Simulation recovery | TMS depth: ψ r 0.979, centred cos 0.884, 6/7 programs, cell φ r 0.98; EasySci depth: ψ not recovered | **pass** (A); B descriptive |
| 5a Biology | Live programs are coherent lineages; *Ptprc* and *Fgfr2* hit in their lineage programs, which do not replicate | **fail** as pre-registered (0/4 in replicating programs; 2/4 in live) |
| 5b Claims reproduce | 6 of 7 key numbers reproduce from saved fits/outputs with current code | **pass** with 1 correction |
| 8 Consensus programs (15 seeds, TMS) | 4 clusters recur in ≥ 60 % of fits, 3 of them with top-50 Jaccard ≥ 0.5 (need 8) | **fail**: current wording stays |

In one line: LeafletFA's per-cell representation is sound — it fits held-out cells better than fair
nulls, is stable across seeds, reads splicing ratios rather than detection, and is recovered on data from
the model — but individual programs are mostly not reproducible across seeds on real data, so claims about
named programs need the stability rule.

## Check 0 — the paper's model (`tsv/c0_model_provenance.tsv`)
- The manuscript Methods (`methods_CURRENT.tex`) describe the Dirichlet-multinomial simplex model; every
  main figure was built from DM fits; training commits 6c4fb9b–e43fb91 (Aug 9–11) differ from current main
  only by E-step/API additions and this week's opt-in options. No default-path change alters results.
- Text fixes, applied on Leaflet-analysis branch `leafletfa-methods-fixes` (b1af325, not pushed): stale
  "BetaBinomial / DO NOT EDIT" header removed; π hyperprior α_π ~ Gamma(1,1) added; lr 0.015 for the
  per-dataset K30 fits; 3,000-step simulation schedule; 75 % of per-ATSE concentrations at the floor of 2.
- Serious: "the aging program replicates (r 0.95–0.97)" conflicts with A70 (centred cosine 0.65). Options in
  `manuscript/AGING_CLAIM_OPTIONS.md` (on that branch); decision pending.

## Check 1 — held-out likelihood (`tsv/c1_*`)
- **Origin of −3.1e7:** A44 lines 324–356 scored every held-out cell with ONE average φ, against the
  read-weighted pooled usage, at a single concentration of 2, on 78,012 cells of which 54,571 were Allen
  brain cells the TMS model never saw. It tested a population average, not the model.
- **Fair score:** each TEST cell's observed events split 50/50; φ inferred on one half, the other half
  scored (no read scored by a φ that saw it); all null parameters fitted on TRAIN, per-ATSE ML concentration.

| Gain per read (model − null), 95 % CI over mice | TMS (18 mice) | Allen (489 donors) |
|---|---|---|
| population ψ (i) | +0.00178 [0.00154, 0.00211] | +0.00324 [0.00309, 0.00340] |
| cell-type ψ (ii), supervised | +0.00110 [0.00096, 0.00126] | +0.00172 [0.00166, 0.00179] |
| K = 1 (iii) | +0.00073 [0.00059, 0.00095] | +0.00201 [0.00185, 0.00221] |
| naive − split (optimism of same-reads φ) | 0.00015 | 0.00019 |

- Verification: (a) scorer = package `dm_log_prob` + coefficient to 7e-10 per cell (both datasets);
  (b) on data simulated from the TMS fit the true parameters beat null (i) (+0.0011/read, 99.95 % of
  cells); (c) shuffling labels removes null (ii)'s advantage entirely.
- 1c not needed (no loss). Per-read gains are small because reads are heavily duplicated (~2 effective
  molecules per event; 76–85 % of the null models' concentrations also sit at the floor of 2).

## Check 2 — K and prior, TMS (`tsv/c2_K_prior.tsv`, `tsv/c2_replication_*`)

| config | kNN real | content-only | exact avg | drop | programs in use | participation ratio | gain vs (i) |
|---|---|---|---|---|---|---|---|
| K 5 | 0.393 | 0.197 | 0.205 | 0.196 | 5 | 3.3 | 0.00172 |
| K 10 | **0.479** | 0.244 | 0.254 | 0.235 | 8 | 5.7 | 0.00188 |
| **K 20** | 0.476 | 0.245 | 0.309 | 0.231 | 7 | 8.8 | 0.00178 |
| K 30 | 0.429 | 0.207 | 0.264 | 0.222 | 7 | 9.8 | 0.00172 |
| K 20, prior ≥ 1 | 0.501 | 0.312 | 0.339 | 0.189 | 9 | 9.7 | 0.00179 |

Amendment-1 rules (pre-registered):
- "Results similar at K = 10 and 30": **false** — K = 10 is within 0.003, K = 30 is 0.047 lower (0 non-finite
  restarts at lr 0.025, so no 0.015 rerun was triggered).
- K = 30 replaces K = 20 in Figure 1: **no** (it is worse).
- "Higher K adds unused programs, not information": **true** — programs in use 7 → 7, participation ratio
  +11 %, kNN not better.
- **For the SpliceVI Methods** (decided by Karin, 2026-10-25): "K = 20 performed as well as K = 10 and
  better than K = 30."
- Prior floored at 1: higher kNN (0.501) but a smaller content-only drop (0.189 vs 0.231) — the flatter
  prior reads more from detection, less from ratios. Not a better baseline for Figure 1's question.

Replication by K (three seeds, centred ψ ≥ 0.8, live in all seeds; existing grids, no new training):

| grid | K 10 | K 20 | K 30 |
|---|---|---|---|
| TMS (programs replicating / in use) | 2 / 7 | 3 / 7 | 2 / 5 |
| mouse Allen | 3 / 5 | 2 / 3 | 2 / 5 |
| atlas | 3 / 6 | 1 / 6 | 2 / 8 |

Lower K does not produce replicating programs. Prior-floor replication was not measured (needs 2 extra
seeds, ~0.5 GPU-h, not approved).

## Check 3 — seed stability (`tsv/c3_*`)

| | kNN real (seeds 0/1/2) | mean [range] | content-only drop mean | programs replicating (centred) |
|---|---|---|---|---|
| TMS | 0.476 / 0.475 / 0.451 | 0.467 [0.025] | 0.22 | **3** of 6 live in all seeds |
| Allen | 0.419 / 0.421 / 0.422 | 0.421 [0.003] | 0.20 | **1** of 2 live in all seeds |

Raw-ψ correlation is ≥ 0.95 for every matched live program in every seed pair — the inflation that
amendment 2 removed. The embedding is stable; the individual programs are not.

## Check 4 — simulation recovery (`tsv/c4_*`)

| depth | ψ r (median, live) | centred cos | programs recovered ≥ 0.8 | per-cell φ r | kNN fitted / true φ |
|---|---|---|---|---|---|
| A, TMS | **0.979** | **0.884** | 6 / 7 | **0.982** | 0.41 / 0.50 |
| B, EasySci-like (×0.011) | 0.745 | 0.337 | 0 / 7 | 0.892 | 0.20 / 0.50 |

When data follow the model, one fit recovers the programs, so the seed instability on real data reflects
misfit between the data and a 20-program mixture (or genuine non-identifiability of the minor programs),
not an optimiser failure. At EasySci depth programs are not recoverable; per-cell φ r stays high only
because one program (π 0.66) dominates.

## Check 5 — biology and claims
**5a** (`tsv/c5_*`): TMS seed-0 live programs: P1 myeloid/immune (π 0.66), P2 keratinocyte/basal (replicates),
P3 islet/neuron/hepatocyte, P4 stromal/mesenchymal (cos 0.77, just misses), P6 brain glia (replicates),
P7 endothelial/smooth muscle (replicates), P9 intestinal epithelium. *Ptprc* is in the immune program's
top 1 % ATSEs and *Fgfr2* in the stromal program's; both programs fail replication. No muscle- or
neuron-dominant program exists at K 20 in TMS, so those events cannot hit. 0/4 in replicating programs →
fail as pre-registered; the biology of the live programs is nonetheless coherent.

**5b** (`tsv/c5b_claims.tsv`; figure scripts not rerun because they write into the Genome Biology figure
folders):

| fig | claim | stated | found | status |
|---|---|---|---|---|
| 1 | atlas size | 132,706 × 79,768 × 24,475 | same | reproduced |
| Methods | concentrations at floor | 75 % | 75.0 % (recomputed) | reproduced |
| 4/S5 | programs replicating, atlas K20, raw ψ r ≥ 0.9 | 11 | 11 (recomputed) | reproduced — but the metric is inflated (centred: 1–4) |
| 3b | calls per contrast at matched FDP | 840 vs 641 | median 840 vs 641.5 over 70 contrasts | reproduced; say "median" |
| 3c | posterior calls, gated; shared | 49,865 → 7,041; 3,551 | same | reproduced |
| 4b | aging program P; types up; 3m → 24m | 0.0159; 94 %; 0.13 → 0.22 | same | reproduced |
| 5 | conserved pair r over 9 seed pairs | 0.586–0.589 | 0.581–0.589 | **not reproduced** — range too narrow |

Not checked (outputs present, scripts compile, numbers not recomputed): 2d/e silhouette, 2g calibration,
2i 0.998, imputation r, uncertainty coverage, Fig 3 TV ceiling and precision/recall, K robustness Jaccard,
δ choice, φ variance 92.8 %.

## PROGRAM_STABILITY_OPTIONS

What is settled: claims about the per-cell representation and the ATSE-level tests are supported (checks 1,
3-kNN, 4). Claims about individually named programs are supported only for programs that replicate under
the 27 Sept rule (centred ψ ≥ 0.8 in both other seeds, used by cells). On the atlas K20 that is 4 programs
by A70's liveness (SP5 hippocampal/excitatory, SP7 inhibitory, SP8 cortical excitatory, SP15
keratinocyte/basal) and 1 by the stricter argmax ≥ 1 % liveness. Lower K does not help (check 2).

### (a) Consensus programs across seeds
- What: pool centred ψ over seeds (× K 20/30), cluster at cosine 0.8, keep clusters present in ≥ 80 % of
  fits and used by some cell type; consensus ψ = member mean; re-infer φ by the frozen E-step.
  (`PROPOSAL_program_stability_and_consensus.md`, Option A: existing fits, CPU minutes + one E-step per grid.)
- How many recur: bounded by the replication counts above — about 4 on the atlas K20 (A70 liveness),
  3 on TMS, 1–2 on Allen; K 30 adds no extra recurring programs. Option A has not been run.
- Claims it supports: program-level statements for the recurring programs only — neuronal programs
  (SP5/7/8), keratinocyte/basal (SP15), and the cross-species conserved pair (passes, 0.97/0.98). It does
  **not** rescue the aging program (0.65), myeloid (0.52), mesenchymal (0.67) or endothelial programs.
- Manuscript affected: Methods L131 (replication definition), a new Methods paragraph on consensus
  programs; Fig 3 supp program_sharing heatmap and Fbln2 decomposition (mark consensus programs); Fig 4a rows;
  Fig 4 supp a, b, d; SP-by-cell-type heatmaps (TMS-only, AB-only, human); Results draft Fig4/5 L10.

### (b) Restrict claims to the per-cell representation and the ATSE catalog
- What: present programs as a basis (a low-dimensional representation and the ψ dictionary used by the DS
  test), not as named biological entities; keep every ATSE-level result (Fig 3 calibration, matched calls,
  posterior calls, LRT), imputation, held-out fit (new check 1), the embedding stability (check 3 kNN) and
  the junction-level examples (Apbb2, Mbd3, Kcnip3 as a junction).
- Manuscript affected: Fig 4 title and panels b–f (aging as a junction-level/application result or cut, per
  AGING_CLAIM_OPTIONS.md); Fig 4a program rows; Fig 3 text on Gna13 ("program-level, amplified" →
  basis-level wording); Fig 3 supp Fbln2 "shared SP"; Fig 5 kept only for the conserved pair (which passes);
  Methods L131, L142; Results draft Fig4/5 L10, L14–27; caption CAPTION_FIG4.md L3, L10–25.
- Strongest fit with the method-plus-resource framing; least analysis; the held-out and simulation results
  from this validation become supporting evidence.

### (c) A different K or prior
- Check 2 shows no K gives replicating programs (2–3 at K 10/20/30 on TMS; 1–3 on other grids), and K 30
  lowers kNN. The prior floored at 1 raises kNN but lowers the content-only drop (reads more from
  detection); its replication is unmeasured (2 more seeds, ~0.5 GPU-h).
- Not recommended on current evidence. If pursued, it changes the model and needs retraining of every
  figure's fits (atlas, TMS, Allen, human × 3 seeds), so every number in Figs 2–5 would change.
- Manuscript affected: everything that cites a fit (Methods model and fitting sections, all figures).

**Recommendation:** (b) as the framing, plus (a) restricted to the programs that pass, for the few
program-level statements worth keeping (neuronal programs, keratinocyte/basal, conserved mouse–human pair).
(c) only if Karin wants to test the flatter prior's replication (~0.5 GPU-h) first.

## What the manuscript must change
1. Methods L131: define replication on centred ψ; raw ψ r ≥ 0.9 passes nearly every program.
2. Aging claim (Methods L142, Fig 4 caption, Results draft): decide per AGING_CLAIM_OPTIONS.md.
3. Fig 5 conserved pair: report r 0.581–0.589 over the 9 seed pairs (not 0.586–0.589).
4. Fig 3b: say "median per contrast" for 840 vs 641.
5. Add the held-out likelihood result (event-split, fair nulls) — it answers the reviewer question the old
   −3.1e7 would have raised; drop or correct the old A44 held-out number wherever it appears.
6. The four Methods text fixes already on `leafletfa-methods-fixes` (header, π hyperprior, lr 0.015,
   simulation schedule) and the 75 %-at-floor sentence.
7. SpliceVI paper Methods: "K = 20 performed as well as K = 10 and better than K = 30." (Karin, 2026-10-25)

## Deviations from the pre-registration
- Verification (b) scored null (i) only (null (iii) K = 1 not refitted on the simulation; its ML
  counterpart is null (i)).
- Verification (c): the shuffled-label null reuses null (ii)'s concentration; it fell slightly below
  null (i) rather than to it.
- Liveness for replication: argmax ≥ 1 % of cells (amendment 2); A70 used "some cell type's mean φ ≥ 0.05".
  Both reported for the atlas (1 vs 4 programs).
- 5b: figure build scripts not rerun (they write into the Genome Biology folders); numbers recomputed from
  saved fits or re-read from saved outputs instead.
- Check 2 prior-floor replication not measured (needs extra seeds beyond the approved budget).

## Check 6 — program stability up to resolution (amendment 3; `tsv/c6_*`)

Pre-registered before computation (VALIDATION_PREREG.md, amendment 3).
- **Fits:** TMS and Allen baseline fits, K 20, seeds 0–2, training cells, live programs only (argmax ≥ 1 %
  of cells).
- **Many-to-one test:** each program's centered ψ is regressed by non-negative least squares on the live
  programs of the other seed. "Reproduced up to splitting" means R² ≥ 0.8 with ≤ 3 programs.
- One-to-one (Hungarian, centered cosine ≥ 0.8, matched program live) is checked first. A single program
  reaching R² ≥ 0.8 needs cosine ≥ 0.894, stricter than one-to-one.

**Counts per ordered seed pair (i → j: live programs of seed i, classified against seed j)**

| dataset | 0→1 | 0→2 | 1→0 | 1→2 | 2→0 | 2→1 |
|---|---|---|---|---|---|---|
| TMS (7 live) one-to-one / up to splitting / not | 4 / 0 / 3 | 4 / 0 / 3 | 4 / 0 / 3 | 4 / 0 / 3 | 4 / 0 / 3 | 4 / 0 / 3 |
| TMS reproduced fraction | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 | 0.57 |
| Allen (3–4 live) one-to-one / up to splitting / not | 2 / 1 / 1 | 1 / 1 / 2 | 2 / 0 / 2 | 1 / 0 / 3 | 1 / 0 / 2 | 1 / 2 / 0 |
| Allen reproduced fraction | 0.75 | 0.50 | 0.50 | **0.25** | **0.33** | 1.00 |

**Reading (pre-registered): "programs are reproducible up to resolution" may NOT be used.**
- TMS reaches ≥ 0.5 in every pair, but every TMS reproduction is one-to-one: splitting rescues no TMS
  program.
- Allen falls to 0.25 and 0.33 in two pairs.
- The current wording stays: program-level statements only for one-to-one reproduced programs.

**Splits and merges (all in Allen; none in TMS)**

| pair | program (top cell types) | explained by (top cell types) | R² |
|---|---|---|---|
| 0→1 | seed-0 P1: hippocampal + isocortex excitatory and inhibitory neurons | seed-1 P1 (isocortex excitatory) + P3 (isocortex inhibitory) | 0.81 |
| 0→2 | seed-0 P1 (same) | seed-2 P1 (inhibitory) + P2 (excitatory) | 0.82 |
| 2→1 | seed-2 P1: isocortex + hippocampal inhibitory neurons | seed-1 P2 + P3 (two inhibitory programs) | 0.85 |
| 2→1 | seed-2 P2: isocortex excitatory neurons | seed-1 P1 (excitatory) + P3 (inhibitory) | 0.86 |

- Read the other way, these are merges: seed 0's P1 is one mixed neuron program where seeds 1 and 2 have
  separate excitatory and inhibitory programs.
- Seed 1's inhibitory programs P2 and P3 merge into seed 2's P1.
- These are resolution differences of the kind the test was designed to find. They are confined to the
  Allen neuron programs.

**Meaning-level match (descriptive; medians over programs and pairs)**

| class | centered cosine (Hungarian) | R² with all live programs | cell-type profile r (best single) | marker-junction Jaccard (best single) |
|---|---|---|---|---|
| TMS one-to-one | 0.90 | 0.83 | 0.99 | 0.64 |
| TMS not reproduced | 0.69 | 0.54 | **0.81** | **0.19** |
| Allen one-to-one | 0.98 | 0.96 | 1.00 | 0.67 |
| Allen up to splitting | 0.72 | 0.85 | 0.96 | 0.07 |
| Allen not reproduced | 0.66 | 0.47 | **0.92** | **0.02** |

- Programs that are not reproduced still have a counterpart used by the same cell types: profile r 0.81
  (TMS) and 0.92 (Allen).
- But they share almost none of their top-50 marker junctions (Jaccard 0.19 and 0.02), and even all the
  other seed's live programs together explain only about half of their centered composition (R² 0.54 and
  0.47).
- So **which cell types a program describes is reproducible, but the splicing content assigned to it
  mostly is not**: different seeds encode the same cell-type structure with different junction
  compositions.
- This agrees with the rest of the validation:
  - the per-cell representation is stable (check 3 kNN);
  - claims should rest on the per-cell representation and on ATSE-level tests;
  - a program's junction content is a claim only for one-to-one reproduced programs.
- **Possible wording** (not pre-registered as a reading, offered for Karin):

  > Cell-type usage of programs was reproducible across seeds, but for most programs the junction
  > composition was not; we therefore make program-level statements only for programs whose composition
  > reproduced.

- Code: `scripts/validation/c6_stability_resolution.py`.

## Check 8 — consensus programs, cNMF-style (amendment 4; `tsv/c8_*`)

Pre-registered before any fit (VALIDATION_PREREG.md, amendment 4). 2026-10-30.
- **Fits:** TMS, K 20, production recipe, training cells (54,694). Seeds 0–2 reused; seeds 3–14 new
  (`c8_consensus/fits/`). All 15 completed with 0 non-finite restarts. GPU: 5.19 GPU-h for the 12 fits
  (L40S, about 26 min each) plus 0.07 for the E-step, 5.26 GPU-h of the 6 approved.
- **Pooling:** 110 live programs (argmax ≥ 1 % of cells), 7.3 per fit. Average-linkage clustering, cosine
  distance on centered ψ, cut at 0.2 (mean pairwise centered cosine ≥ 0.8): 36 clusters.
- **Consensus fit:** clusters in ≥ 60 % of fits, member-mean ψ, π and per-ATSE concentration as
  pre-registered (`c8_consensus/consensus_fit.npz`); φ re-inferred by the frozen E-step
  (`c8_consensus/estep/`).

**Result: 4 clusters recur in ≥ 60 % of fits; 3 also pass the Jaccard bar.**

| Consensus program | Fits | Mean pairwise cosine | Top-50 Jaccard | Passes | π | Top cell types (mean consensus φ) |
|---|---|---|---|---|---|---|
| CP1 | **15 / 15** | 0.89 | 0.64 | yes | 0.14 | brain oligodendrocyte 0.94, astrocyte 0.83, neuron 0.77, inhibitory neuron 0.75 |
| CP2 | 12 / 15 | 0.86 | **0.71** | yes | 0.30 | heart, brain, kidney, BAT endothelial 0.82–0.85; lung smooth muscle 0.85 |
| CP3 | 11 / 15 | 0.89 | 0.53 | yes | 0.24 | large-intestine intestinal and epithelial 0.90–0.91; kidney epithelial 0.79; hepatocyte 0.75 |
| CP4 | 10 / 15 | 0.83 | 0.42 | no | 0.32 | mammary stromal, adipose mesenchymal stem cells, trachea fibroblast 0.75–0.78 |

- **Reading (pre-registered): fewer than 8 consensus programs pass (3), so the current wording stays.**
- The bar was out of reach by construction: each fit has only about 7 live programs, so 8 recurring
  programs would need nearly every live program to recur.
- **Next-most recurrent clusters** are in 7 of 15 fits (two clusters, the two keratinocyte/basal
  compositions below; Jaccard 0.60 and 0.55); every other cluster is in ≤ 4 fits. So beyond the four broad lineages (brain, endothelial, epithelial, mesenchymal),
  live programs are seed-specific.
- **Sensitivity (descriptive):**

  | Clustering | Clusters | Recur ≥ 60 % | Also Jaccard ≥ 0.5 |
  |---|---|---|---|
  | average, cut 0.1 | 76 | 0 | 0 |
  | **average, cut 0.2 (primary)** | 36 | 4 | 3 |
  | average, cut 0.3 | 24 | 4 | 3 |
  | complete, cut 0.2 | 43 | 1 | 1 |

- **Top junctions** (by marker score; full lists in `c8_consensus_programs.tsv`):
  - CP1: *Rcan2*, *Etv1*, *Tsc22d4*, *Exoc7*, *Sept8*, *Fez2*, *Dpf2*, *Capzb*.
  - CP2: *Utrn*, *Tmem184b*, *Sos1*, *Nhlrc2*, *Cald1*, *Slc7a1*.
  - CP3: *Tpm1* (two junctions of the mutually exclusive exon pair), *Rps24*, *Akap9*, *Arid5b*, *Pdgfa*, *Gne*.
  - CP4: *Steap3*, *Sh3d19*, *Abcg2*, *Sash1*, *Aff3*, *Sept9*, *Pgap2*.
- **Relation to check 6** (seed-0 programs, check 6 numbering from 1, check 8 from 0):
  - The seed-0 brain, endothelial and mesenchymal programs (check 6 P6, P7, P4; one-to-one in check 6) are
    members of CP1, CP2 and CP4.
  - The keratinocyte/basal program (P2), one-to-one in both seed pairs in check 6, is in a cluster found in
    only **7 of 15** fits (seeds 0, 1, 2, 6, 8, 9, 10). The other 7 fits (3, 4, 5, 11–14) carry a second
    keratinocyte/basal cluster: same top cell types (tongue keratinocyte and basal, skin epithelial and
    basal), centroid centered cosine 0.77 with the first, never both in one fit. So a keratinocyte/basal
    program is present in 14 of 15 fits but in two compositions, split by the 0.8 cut. Even counted as
    one, it would give at most 4 passing programs, so the reading does not change. Seeds 0–2 happened to
    agree on one composition, which overstated its stability.
  - CP3 (intestinal and kidney epithelial, hepatocyte) recurs in 11 of 15 fits but has no seed-0 member;
    seed 0's intestinal program (P9, not reproduced in check 6) is in a cluster found in 4 of 15 fits.
  - So 15 seeds add no recurring program beyond the broad lineages, and they show that one program three
    seeds called reproducible has two alternative compositions. This is the usage-vs-content pattern of
    check 6 at larger scale: the cell types a program describes recur; its junction composition does not.

**Pseudobulk check of the strongest 3 (descriptive; `c8_pseudobulk_*.tsv`)**
- Strongest 3 by within-cluster Jaccard: CP2, CP1, CP3. Samples: tissue–cell type × mouse, ≥ 20 reads at the
  ATSE. Two-sided Mann–Whitney, users vs non-users, BH over the 30 tested ATSEs.

| Program | User / non-user types | Top 10 ATSEs confirmed (concordant, q < 0.05) | Concordant | Random ATSEs with P < 0.05 |
|---|---|---|---|---|
| CP1 (brain) | 7 / 14 | **9 / 10** | 9 | 0.23 |
| CP2 (endothelial) | 95 / 1 | 3 / 10 | 4 | 0.22 |
| CP3 (epithelial) | 81 / 1 | **8 / 10** | 8 | 0.20 |

- CP1 and CP3 junctions are confirmed well above the random-ATSE rate. Examples:
  - CP1: *Tsc22d4* Δ −0.98, *Arhgap21* Δ −0.98, *Eml2* Δ +0.99.
  - CP3: *Tpm1* +0.90 / +0.91, *Rps24* +0.98, *Arid5b* +0.95.
- **CP2 is a weak test, not a negative result.**
  - It is used by 95 of 96 cell types, so the only non-user type gives 6–12 samples.
  - 6 of its 7 unconfirmed ATSEs have equal medians (Δ = 0), which the pre-registered concordance rule
    counts as discordant.
  - The same rule fails CP1 *Etv1* (P < 1e-4, Δ = 0) and CP3 *Gne*.
- CP2 and CP3 have π 0.30 and 0.24 and are used by most cell types, so they are broad lineage axes
  rather than cell-type-specific programs.
- Caveat stated in advance: samples from the same mouse are not independent across cell types.

**What this means for the text**
- Consensus over 15 seeds does not restore more programs than the one-to-one rule already gives. The
  reproducible program-level structure in TMS is four broad lineage axes (brain, endothelial, epithelial,
  mesenchymal), and the mesenchymal axis does not keep a stable junction identity (Jaccard 0.42).
- The current wording stays: program-level statements only for reproduced programs.
- The brain and epithelial consensus programs (CP1, CP3) are the best-supported program-level examples:
  they recur in 15/15 and 11/15 fits and their top ATSEs are confirmed by pseudobulk tests.
- Code: `scripts/validation/c8_consensus.py`, `c8_estep.sh` (LeafletFA `leafletfa-validation`, c1d24fd).

## Check 9 — is differential splicing reproducible and calibrated? (amendment 5; `tsv/c9*`)

Pre-registered before computation (VALIDATION_PREREG.md, amendment 5). 2026-11-01. CPU only, the 15 TMS K20 fits of
check 8; q(ψ) and q(φ) read from each fit's checkpoint (verified: the checkpoint's ψ median equals the saved ψ).

**Reading: 9a FAILS. Differential splicing cannot be main Fig. 3; simulation recovery stays.**

**9a. Reproducibility across 15 seeds.** Shipped posterior test (δ = 0.15, 200 draws, Bayesian FDR q ≤ 0.05, ≥ 25
covered cells per arm), 5 GB Fig. 3b contrasts chosen by GB LRT call count (all 70 contrasts were eligible).

| Contrast (GB Fig. 3b LRT calls) | ATSEs tested | Calls, seed 0 | Calls across fits | Pairwise Jaccard | Per-cell ψ r across seeds |
|---|---|---|---|---|---|
| Marrow NK vs T (188) | 3,366 | 0 | 0 in all 15 | undefined (no calls) | 0.999 |
| Skin basal vs epithelial (509) | 8,370 | 0 | 0 in all 15 | undefined | 0.999 |
| Marrow granulocyte vs T (833) | 4,046 | 0 | 0–105 (calls in 5 of 15 fits) | median 0.0, max 0.6 | 0.998 |
| Tongue basal vs keratinocyte (1,239) | 11,124 | 0 | 0 in all 15 | undefined | 0.998 |
| Marrow B vs HSPC (2,954) | 11,628 | 1 | 0–150 (median 14) | median 0.0, max 0.6 | 0.998 |

- **Median pairwise Jaccard over the fit pairs where any call exists: 0.0** (pass needed ≥ 0.7). Fraction of seed-0
  calls found in ≥ 12 of 15 fits: 0 (one seed-0 call in total).
- **The per-cell imputed values are reproducible; the calls are not.** ρ = φ·ψ correlates at r 0.996–0.9996 across all
  105 fit pairs and 5 contrasts. The posterior TV of most ATSEs sits far below δ = 0.15 (median 0.001–0.012; 99th
  percentile 0.01–0.10), so the few calls are events whose TV lands just above δ in one fit and just below in another.
- **Not a reimplementation artifact.** The GB atlas run of the same test (`LEAFCUTTER_CMP_DELTAONLY/a51_delta0.15`)
  also called 0 in three of these five contrasts (130 and 141, ungated, in the two marrow contrasts), with the same TV
  distributions.
- **What GB Fig. 3b counted was the LRT** (A49, group design), not the posterior test; the LRT was not part of this
  pre-registration.

**9b. Calibration on a real-data null** (5 cell types: heart cardiac muscle, microglia, skin keratinocyte, skin basal,
heart fibroblast; 13–15 mice; cells split in half within each mouse; 20 splits each).
- **LeafletFA: 0 calls in all 100 splits** (6,300–12,800 ATSEs tested per split; maximum P(TV > 0.15) = 0). False-call
  rate 0 in every type: passes the pre-registered rule. **Wording for the paper (Karin, 2026-11-01):** "LeafCutter is anti-conservative on mouse-split nulls; LeafletFA's posterior test made no calls, consistent with its conservative behavior". Not
  to be described as LeafletFA being calibrated: the test calls almost nothing in any within-tissue contrast.
- **LeafCutter** on the same splits (one pseudobulk per mouse and arm, 13–15 vs 13–15 samples, 100 runs, array
  22596857): **12.0–14.3 % of tested clusters at p < 0.05** (nominal 5 %) and **1.0–1.7 % at BH q < 0.05**, i.e. about
  100–140 false calls per split (6,400–11,100 clusters tested), in every one of the 5 types. Anti-conservative on this
  null, where LeafletFA makes no calls.

| Cell type | LeafletFA false-call rate | LeafCutter p < 0.05 | LeafCutter q < 0.05 |
|---|---|---|---|
| Microglia | 0 | 0.120 | 0.010 |
| Heart cardiac muscle | 0 | 0.134 | 0.017 |
| Heart fibroblast | 0 | 0.133 | 0.014 |
| Skin basal | 0 | 0.142 | 0.013 |
| Skin keratinocyte | 0 | 0.143 | 0.014 |

**What this means for the paper**
- The shipped posterior DS test at δ = 0.15 is very conservative (no calls on the null; not described as calibrated): it is silent on within-tissue
  contrasts, and where it does call, the calls do not reproduce across seeds. Differential splicing stays out of the
  main text.
- The reproducible part is the per-cell imputed usage (r ≈ 0.998 across seeds), which is what the paper's per-cell
  claims already rest on.
- Not tested here: the LRT (group design), which is what produced GB Fig. 3b's counts. Making it a figure would need
  its own pre-registered seed and null check.
- Code: `scripts/validation/c9_lib.py`, `c9_ds_seeds.py`, `c9_ds_null.py` (LeafletFA `leafletfa-validation`, a4e6c7f).

### Check 9a′ (amendment 5a): the GB Fig. 3c setting

- **Why:** amendment 5's contrasts were chosen by LRT counts, and four of them had 0 gated posterior calls in GB Fig. 3c.
  Its settings also differed from GB's (TMS-only fits; q after the gate).
- **Setting:** the shipped GB call sets (pooled-atlas K20 seeds 0–2, q over all ATSEs, then the ≥ 25-cell gate), on the
  5 contrasts with the most gated consensus calls.

| Contrast (GB gated consensus calls) | Calls per seed (0 / 1 / 2) | Pairwise Jaccard | Seed-0 calls in both other seeds | Per-cell ρ r |
|---|---|---|---|---|
| Brain endothelial vs oligodendrocyte (744) | 818 / 786 / 673 | 0.59–0.70 (median 0.63) | 0.63 | 0.95 |
| Brain oligodendrocyte vs pericyte (410) | 419 / 423 / 372 | 0.63–0.82 (0.63) | 0.70 | 0.92 |
| Heart cardiac muscle vs monocyte (369) | 345 / 451 / 292 | 0.28–0.55 (0.42) | 0.39 | 0.95 |
| Heart endothelial vs fibroblast (345) | 413 / 377 / 329 | 0.41–0.53 (0.51) | 0.46 | 0.94 |
| Heart fibroblast vs monocyte (330) | 321 / 349 / 228 | 0.30–0.58 (0.47) | 0.36 | 0.91 |

- **Median pairwise Jaccard 0.548 over 15 pairs: FAIL** (pass needed ≥ 0.7).
- Where the test calls hundreds of events, about half reproduce between two seeds: 36–70 % of seed-0 calls are called by
  both other seeds.
- Per-cell ρ over the called events' junctions correlates at 0.91–0.97 across seeds, lower than check 9a's 0.998 over
  all tested junctions: called events are where the seeds differ most.
- **Reading unchanged:** DS stays out of the paper. Code: `scripts/validation/c9_ds_atlas.py`; tables `tsv/c9a2_*`.

### Check 9a′ — POST HOC, descriptive (amendment 5b; the pre-registered result above stands)

Same 5 contrasts, 3 atlas seeds, gated ATSEs (`tsv/c9a3_*`, `c9_ds_atlas_posthoc.py`).

| Measure | Median | Range (pairs × contrasts) |
|---|---|---|
| Spearman of per-event effect (TV) between seeds, all gated ATSEs | 0.86 | 0.72–0.95 |
| Spearman of posterior probability between seeds | 0.75 | 0.48–0.89 |
| Relaxed replication: called in A → q < 0.2 and same sign in B | 0.75 | 0.37–0.94 |
| Sign agreement at the lead junction, events called in both seeds | 0.99 | 0.90–1.00 |
| Fraction of a seed's calls in the ≥ 2-of-3 consensus set | — | 0.70–0.94 |

- The effect ranking agrees between seeds, and events called in two seeds agree in direction about 99 % of the time.
- What does not reproduce is the threshold crossing: a quarter of calls (median) are not even at q < 0.2 in another seed.
- Brain contrasts reproduce better than heart contrasts on every measure.
- The August consensus rule (≥ 2 of 3) keeps 70–94 % of each seed's calls; its stability cannot be tested with 3 seeds.
- Descriptive only: no claim is made from these numbers beyond what the pre-registered test showed.

## Check 10 — the cell-type-resolved catalog (amendment 6 with notes 1–4; `tsv/c10_*`, `c10/`)

Seed-0 runs: TMS job 22597576, Allen 22597577; 10.2c call sets 22597578/79 and 22606140/41; 10.4 bug fix (note 4)
22607878; diagnostics (read-only, exploratory) 22607879. **Pre-registered reading: 10.1 passes, 10.2 fails → Application
Note** (accepted by Karin, 2026-11-02).

**Split.** Per cell, 70/30, stratified by broad cell type × age, seed 42; not grouped by mouse. All 18 TMS mice have
TRAIN and TEST cells; in Allen 482 of the 489 TEST donors also have TRAIN cells. TEST accuracy is accuracy on new cells
of the same mice.

| Subcheck | TMS | Allen | Rule | Result |
|---|---|---|---|---|
| 10.1 overall error LeafletFA vs raw (diff, 95 % CI) | 0.0940 vs 0.1012 (−0.0072; −0.0094, −0.0050) | 0.0625 vs 0.0647 (−0.0021; −0.0039, −0.0005) | ≤ raw, and CI < 0 in smallest size or 1–19-read bin | **PASS / PASS** |
| 10.1 smallest types (10–49 TRAIN cells) | −0.0194 (−0.0242, −0.0153) | −0.0116 (−0.0143, −0.0090) | | |
| 10.1 1–19 TRAIN reads | −0.0625 (−0.0681, −0.0571) | −0.0543 (−0.0599, −0.0482) | | |
| δ (99th pct split-half TV) | 0.079 → 0.10 | 0.074 → 0.10 | | |
| Ineligible types (cells) | 5 (1,312) | 1 (26) | | |
| 10.2 calls (full / half A) | 46,262 / 30,496 | 34,190 / 29,487 | | |
| 10.2a confirmed (i) / (ii) | 0.783 / 0.813 | 0.636 / 0.837 | (ii) ≥ 0.5 | |
| 10.2b null (i) / (ii) | 0.003 / 0.422 | 0.351 / 0.771 | (ii) ≥ 5× null | **FAIL / FAIL** |
| 10.7 pairwise, 69 contrasts (pooled) | confirmed 0.780, null 0.473 | — | same | **FAIL** |
| 10.2c median Jaccard (pairs) | 0.637 (105; 0.515–0.761) | 0.770 (3) | ≥ 0.7, reported | fail / pass |
| 10.3 coverage, combined (param-only), 100–499 / ≥ 500 | 0.136 (0.118) / 0.218 (0.145) | 0.256 (0.180) / 0.346 (0.232) | reported, nominal 0.95 | far below nominal |
| 10.4 *Fgfr2* | epithelial min 0.704 > mesenchymal max 0.649; IIIb specific in 10 epithelial types | — | stated | **PASS** |
| 10.4 *Ptprc* (note 4: B or T only, TEST direction) | 27 B/T calls (all atse_2), 25 confirmed in TEST | — | ≥ 1 confirmed | **PASS** |
| 10.4 *Nrxn1* early Nrxn site (note 4 30-nt exon; **not SS4**, note 6) | — | atse_2: 270 of 1,891 neuronal pairs differ (226 with the read/cell condition) | ≥ 1 pair | **PASS** (verdict unchanged) |
| 10.4 *Nrxn2* early Nrxn site (note 4 30-nt exon; **not SS4**, note 6) | — | atse_1: 872 of 1,891 (109 with the read/cell condition) | ≥ 1 pair | **PASS** (verdict unchanged) |
| 10.4 *Nrxn3* early Nrxn site (note 4/5 30-nt exon; **not SS4**, note 6) | — | atse_2: 1,201 of 1,891 (696 with the read/cell condition) | ≥ 1 pair | **PASS**, limited (below; verdict unchanged) |
| 10.4 *Nrxn1* SS4 (note 6, SpliceVI §111) | — | atse_7: **1,256 of 1,891** neuronal pairs differ (1,256 with the read/cell condition) | ≥ 1 pair | **PASS** |
| 10.4 *Nrxn2* SS4 (note 6) | — | atse_6: **946 of 1,891** (581 with the read/cell condition) | ≥ 1 pair | **PASS** |
| 10.4 *Nrxn3* SS4 (note 6) | — | atse_7: **1,129 of 1,891** (812 with the read/cell condition) | ≥ 1 pair | **PASS** |
| 10.5 export | 6,478,290 rows | 4,420,578 rows | | done |

- 10.4 *Ptprc*: B cells have lower usage of the lead junction than the rest in TEST (all B-cell calls), T cells mostly higher;
  the two unconfirmed calls are Diaphragm T (898 TEST reads) and Marrow T.
- 10.4 *Nrxn3* (note 5): the first report said "not testable" because of a bug in the SS4 rule, which looked for a junction starting
  at the exon's end + 1, while catalog starts are the exon's last base. Corrected before any *Nrxn3* usage was read. Limit: the
  catalog has no junction ending at the exon's start − 1 and none spanning it, so this ATSE compares the exon's downstream splice
  choices, not inclusion against skipping. For all three genes the rule is ATSE-level (TV over the ATSE's junctions): a differing
  pair shows that the ATSE holding SS4 differs, not that the SS4 junctions drive the difference.
- **10.4 correction (amendment 6 note 6, recorded 2026-11-11 before the SS4 rows were computed).** The note 4 "SS4" exon (the
  gene's single annotated 30-nt exon) is not SS4. In *Nrxn1* it is one of three small early exons of Nrxn1α (18 / 30 / 12 nt,
  `atse_2`; residues ~259–274 of UniProt Q9CS84, after EGF-like 1), consistent with SS#1 (SS number not verified); the *Nrxn2*
  and *Nrxn3* 30-nt exons are likewise not SS4. Those three rows stay, relabeled "early Nrxn site", with their verdicts unchanged;
  the *Nrxn3* limit above applies to them, not to SS4.
- **10.4 SS4 rows (note 6; job 23230218, 4.0 h, peak 0.6 GB; `c10_known_biology_fix.py nrxn_ss4`).** SS4 = the 90-nt exons and
  ATSEs of the SpliceVI STAGE5 §111 event list (fixed 2026-10-06): *Nrxn1* chr17:90,163,792–881 (`atse_7`), *Nrxn2*
  chr19:6,513,716–805 (`atse_6`), *Nrxn3* chr12:90,199,147–236 (`atse_7`). Same 10.4 rule: 62 neuronal groups (Allen class
  Glutamatergic or GABAergic), 1,891 pairs; a pair differs if point TV at the ATSE > δ = 0.10 and the check 10 primary intervals
  do not overlap at ≥ 1 junction. Outputs `tsv/c10_Allen_s0_nrxn_pairs_ss4.tsv` (3,331 pairs; md5 4da03153225bb80462cf15ab7551a527),
  key `nrxn_ss4` in `tsv/c10_known_biology_fix.json` (the `nrxn` key is byte-for-byte unchanged).
  - The note 5 junction rule, run as a cross-check, finds exactly the §111 ATSE for each gene (no other ATSE).
  - All three genes **pass** (≥ 1 differing pair). Each SS4 ATSE holds both the inclusion and the skipping junction, so these
    rows compare inclusion against skipping (unlike the *Nrxn3* early-site row).
  - Descriptive, not part of the rule: median LeafletFA ψ of the SS4 inclusion junction over groups with ≥ 20 TRAIN reads and
    ≥ 3 covering cells, excitatory vs inhibitory: *Nrxn1* 0.153 vs 0.396; *Nrxn2* 0.185 vs 0.334; *Nrxn3* 0.361 vs 0.525.
    Lower in excitatory groups in all three genes, the direction of Nguyen et al. 2016 (SS4 skipped in pyramidal neurons,
    included in PV and SST interneurons). The figure (panel e, `atse_7`) shows the *Nrxn1* split by PV / SST / VIP / Lamp5.
- 10.6 (atlas size, 132,706 cells, b6k RTX PRO 6000 Blackwell): full 16,200-step fit 119 min, 0.27 s/step, peak host RSS
  40.7 GB (job 22597359). scQuint and the LeafCutter catalog: not run yet.
- **10.3 diagnostics (exploratory, Karin 2026-11-02; `c10_diag/`).** Median interval width vs median |estimate − TEST|:
  TMS 0.019 vs 0.017 (100–499), 0.015 vs 0.012 (≥ 500); Allen 0.020 vs 0.015, 0.011 vs 0.007. Coverage (a) as 10.3 /
  (b) overlap with the Jeffreys pseudobulk interval / (c) binomial-widened predictive: TMS 0.136 / 0.654 / 0.570 and
  0.218 / 0.461 / 0.375; Allen 0.256 / 0.844 / 0.759 and 0.346 / 0.753 / 0.635. Causes (read-only, `c10/c103_evidence_readonly.json`):
  γ_e at the floor of 2 for 81 % (TMS) and 68 % (Allen) of ATSEs, with lower coverage there; parameter-only widths below the
  median error; the mouse bootstrap adds little. Amendment 7 (draft, `AMENDMENT7_DRAFT.md`) tests predictive and
  recalibrated intervals; not approved, not run.

## §189 DS tables for the SpliceVI comparison (exports only, not scored)

Written by `ds_pair_tables.py` at 9d03fe4 from fits at bea2b12 (K 20, seed 0, 16,200 steps, minibatch 8,192, lr 0.025, gamma 0.1, conc_min 2,
float64 φ, sparse; CPU, while the other LeafletFA paper fits ran on GPU). Columns `pair, junction_id, score, score_param, dpsi`, every junction, no read filter.
`score` = `p_absdpsi_ge_0.1_mouseboot` (primary, §189 amendment 3); `score_param` = `p_absdpsi_ge_0.1`, parameter draws only. 200 draws, draw seed 0.

| | Test A | Test B |
|---|---|---|
| Input | `allen_thinning/seed0/mdata/thinned.h5mu`, md5 e55915c234285dcdfa9755ea7466add2 | `easysci_v2_mdata_in80_1000_2026-10-24.h5mu`, md5 db2d3ca88643f8cfd14a523f78a057b6 |
| Table | `ds189/LFA_TEST_A.tsv`, md5 c3f4f657fc698f365962499eb86d7e4d | `ds189/LFA_TEST_B.tsv`, md5 5f407f40f84ba7ad5b3554ebe2c9e553 |
| Size | 45 pairs, 3,589,560 rows | 55 pairs, 4,387,240 rows |
| Fit / export job | 22620585 / 22644185 | 22620587 / 22644186 |
| Fit md5 | ed89eea358dca7329be78e1558d397ef | 4f159dea2f2e3e648fd3e0130977655c |

- **Bootstrap unit:** the bootstrap resamples animal labels (`obs.animal`, 20 labels in each file), not individual mice. In the thinned file one unit can
  cover up to 8 mice (`obs.n_mice`).
- **Reproducibility:** `score` and `dpsi` are identical to the first export (22620633, 22620634) in every row of both tests. In the per-pair tables, one
  parameter-only value differs: Test B, Oligodendrocytes vs Interbrain and midbrain neurons, `chr12_103104895_103108217_+`, 0.170 vs 0.165, one draw of 200.
  The re-export ran on a different node; a draw at the 0.1 threshold rounding differently there is the likely cause, not tested. Earlier tables are kept in `ds189/prior_score_only_2026-11-03/`.
- **Test B data:** fit on in80 as Karin instructed, not `easysci_only.h5mu`; the held-out 20 % was not used. `broad_cell_type` recomputed with Smriti's function, 98.3 % agreement with the stored column.

## Check 10.2 null diagnostics (read-only, exploratory; Karin 2026-11-04; `c10_null_diag/`)

**The pre-registered verdict stands: 10.2 fails, Application Note.** Nothing here changes it and no amendment is proposed. Scripts:
`c10_null_diag.py` (TMS job 22652424, Allen 22652425; reproduces the saved calls and null (ii)), `c10_null_diag_extra.py`. 10.7 was not decomposed.

**Definitions (frozen text, 10.2a/b).** Calls are made with half-A cells. (i): the same call rule holds on half B's cells. (ii): B's raw
pseudobulk TV between t and the rest exceeds δ/2 (0.05) with the same sign at the call's lead junction. Null: labels permuted among B's
cells within each mouse (20 permutations), same A calls. The prereg's only stated reason for (ii) is "model-free"; no reason is recorded for the δ/2 bar.

| | TMS | Allen |
|---|---|---|
| Calls in half A | 30,496 | 29,487 |
| Real (i) (type bootstrap 95 %) | 0.783 (0.731–0.824) | 0.636 (0.536–0.719) |
| Null (i) (range over 20 permutations) | 0.003 (0.002–0.003) | 0.351 (0.336–0.363) |
| Real (ii) | 0.813 (0.791–0.832) | 0.837 (0.802–0.867) |
| Null (ii) | 0.422 (0.409–0.435) | 0.771 (0.739–0.804) |
| Real − null (ii), paired | 0.391 (0.361–0.416) | 0.065 (0.050–0.082) |
| Null (ii) = P(TV > 0.05) × P(same sign given TV) | 0.836 × 0.505 | 0.921 × 0.837 |
| Null (ii) by variance decile, lowest to highest | 0.330 to 0.473 | 0.721 to 0.784 (peak 0.807, decile 7) |
| Share of null confirmations, top variance decile (10 % if uniform) | 11.2 % | 10.2 % |
| Null (ii) by permuted-group read decile, lowest to highest | 0.435 to 0.244 (peak 0.469, decile 4) | 0.584 to 0.845 (peak 0.853, decile 9) |
| Same sign given TV, read decile 1 to 10 | 0.444 to 0.561 | 0.597 to 0.986 |
| Label retained by the within-mouse shuffle vs base rate (medians) | 0.020 vs 0.014 | 0.260 vs 0.006 |
| Null (ii), shuffle ignoring mouse (5 permutations) | 0.390–0.401 | 0.300–0.319 |
| Null (ii), B reads > 0 / ≥ 20 | 0.424 / 0.422 | 0.786 / 0.802 |

- High-variance hypothesis: not supported as the main cause. Null confirmations are close to uniform across variance deciles in both atlases.
- TMS: sign is a coin flip, so the null is noise clearing the 0.05 bar; it falls only in the deepest read deciles.
- Allen: the null is inflated. Sign is retained and sharpens with depth, the within-donor shuffle keeps about a quarter of the label, and ignoring donors lowers the null to about 0.31. See the donor check below for what the metadata shows; a cause of the inflation has not been tested.

### Allen donor check (read-only, Karin 2026-11-05; `c10_allen_donor_check.py`, `c10_null_diag/allen_donor/`)

**Summary (Karin, 2026-11-06):** Allen donors were sampled by Cre line, so cell classes are confounded with donor; this inflates the within-donor null. (The first two clauses are measured below; the effect on the null is inferred from the retained-label fraction, not tested separately.)

**Column.** Check 10 and §176 use obs `mouse.id`, set from the Allen metadata field `external_donor_name_label` (`metadata_prep.py:106`); all 54,571 cells match exactly after joining `meta_sample_name` to `sample_name`. Donors: 519 overall, 512 with TRAIN cells, 489 with TEST cells.

| Is it one animal? (519 donors) | Value |
|---|---|
| Donors with one genotype / one sex | 100 % / 100 % |
| Donors with one FACS plan / one region | 99.2 % (max 2) / about 93 % (max 2) |
| Library plates per donor, median (max) | 3 (18) |
| Plates spanning more than one donor | 86.5 % of 617 |

This rules out a batch, plate, sort or dissection: plates span donors and donors span plates. The metadata has no animal-level identifier other than this field, so "one animal" is supported, not proven.

| Per donor, all cells | Donors ≥ 1 cell (519) | ≥ 10 cells (420) | ≥ 50 cells (248) | ≥ 100 cells (163) |
|---|---|---|---|---|
| One class only | 64.2 % | 58.8 % | 50.4 % | 48.5 % |
| Top class ≥ 90 % of cells | 81.1 % | 79.8 % | 75.8 % | 76.1 % |
| Median top-class share | 1.000 | 1.000 | 1.000 | 0.998 |
| Median cell groups | 4 | 6 | 9 | 11 |
| Median top-group share | 0.575 | 0.495 | 0.347 | 0.311 |

Cells per donor: median 46 (10th to 90th percentile 4 to 342). Cell-weighted top-class share 0.94, top-subclass share 0.47. Classes: glutamatergic, GABAergic, non-neuronal.

| Genotype (all cells; 81 values with missing, 79.8 % of donors carry a Cre allele) | Cells | Donors | GABAergic | Glutamatergic | Non-neuronal |
|---|---|---|---|---|---|
| Slc17a7-IRES2-Cre;Ai14 | 19,277 | 59 | 0.000 | 1.000 | 0.000 |
| Gad2-IRES-Cre;Ai14 | 7,520 | 30 | 0.916 | 0.021 | 0.064 |
| Snap25-IRES2-Cre;Ai14 | 7,331 | 33 | 0.242 | 0.715 | 0.043 |
| Slc32a1-IRES-Cre;Ai14 | 4,240 | 28 | 0.994 | 0.000 | 0.005 |
| Slc17a6-IRES-Cre;Ai14 | 2,944 | 11 | 0.000 | 0.999 | 0.001 |
| Rbp4-Cre_KL100;Ai14 | 2,355 | 28 | 0.032 | 0.968 | 0.000 |
| Pvalb-IRES-Cre;Ai14 | 867 | 29 | 0.999 | 0.000 | 0.001 |

Sorting: `facs_population_plan_label` has 6 values; RFP-positive is 52,415 of 54,571 cells (482 donors), RFP-negative 385 cells of which 0.966 non-neuronal. Cramér's V with class: donor 0.81, genotype 0.71, FACS plan 0.36, region 0.24. Other fields present: `injection_*`, `cortical_layer_label`. Consequence for any mouse-level design: most donors contain one class, so class contrasts are largely between donors and hence between genotypes. Not done: the source paper's methods were not read, and class composition was not separated from genotype as a cause of the null.
- Treating an unread group as ψ = 0 does not inflate null (ii): restricting to read groups leaves it unchanged or higher.

## Check 12 — amendment 7 intervals: projected cost above the 30 CPU-h cap (2026-11-06; superseded by the run below)

Amendment 7 appended unchanged to the prereg (a080dcc) before any code ran. Code: `c12_intervals.py` (literal reading; notes in its docstring).
Smoke (10 of 200 draws, 3 types per dataset; jobs 22780809, 22780810; 4 CPUs): TMS 13 min, peak 9.6 GB; Allen 24 min, peak 22.2 GB;
together **2.5 CPU-h** charged. No coverage result was computed or read.

| | TMS | Allen |
|---|---|---|
| Units (type, ATSE) / unit junctions | 548,745 / 1,829,935 | 434,294 / 1,530,439 |
| Load + 200 draws (projected) | 0.2 h + 0.6 h | 0.4 h + 0.4 h |
| (a) simulations, 1,000 per unit, both draw sets (projected) | 36.1 h | 38.0 h |
| Total wall, single-threaded | about 37 h | about 39 h |

Projection: **about 76 CPU-h at 1 CPU per job** (about 300 CPU-h at the 4 CPUs of the smoke), linear in simulation work
(cell-ATSE pairs x junctions: TMS 112.2 M, Allen 291.9 M). A micro-benchmark puts the random-number floor at about 45 % of the
measured time, so optimizing the code cannot bring it under 30 CPU-h. Per the cap, nothing was submitted. Options (each needs Karin):
1. One replicate per draw (200 simulated values, not 1,000): about 17 CPU-h; changes "r in 1..5" in the amendment.
2. A fixed random 25 % of (type, ATSE) units: about 21 CPU-h; coverage then rests on about 110,000 (TMS) and 87,000 (Allen) units.
3. Raise the cap to about 80 CPU-h, as written; about 39 h wall in parallel, which meets Tue 10 Nov only if the jobs start by Sun 8 Nov.


## Browser ranking vs raw Δψ: diagnosis (read-only, Karin 2026-11-06; `c10_browser_gap.py`, `c10_browser_gap_rank.py`, `c10_browser_gap/`)
Jobs 22780597, 22780642 (decomposition), 22782041, 22782042 (rank depth, TRAIN and held-out TEST). Reading rule written in the script before results.

**(a) Same table, no bug.** The browser and 10.1 read the same `point` array; recomputed from the fit it matches the saved file exactly (max difference 0), the per-cell rebuild reproduces it, and the browser's junction reads equal the counts.

| Median over rows, ranked by LeafletFA |Δψ| | TMS example (neuron vs astrocyte) | TMS, 20 random pairs | Allen example (L2/3 IT CTX vs Pvalb) | Allen, 20 random pairs |
|---|---|---|---|---|
| Top 100: gap LeafletFA vs TRAIN raw | 0.392 | 0.260 | 0.175 | 0.126 |
| Top 100: error vs TEST, LeafletFA / TRAIN raw | **0.387 / 0.067** | **0.224 / 0.151** | **0.166 / 0.026** | **0.123 / 0.060** |
| Top 100: \|Δψ_L\| − \|Δψ_TEST\| | −0.387 | −0.121 | −0.166 | −0.081 |
| Top 100: same sign as TEST, LeafletFA / raw | 1.000 / 1.000 | 0.854 / 0.848 | 1.000 / 1.000 | 0.975 / 0.971 |
| 101–1,000: error vs TEST, LeafletFA / raw | 0.251 / 0.151 | 0.123 / 0.147 | 0.073 / 0.048 | 0.077 / 0.077 |
| All: error vs TEST, LeafletFA / raw | 0.048 / 0.064 | 0.040 / 0.052 | 0.019 / 0.019 | 0.033 / 0.038 |
| Raw's own top 100: \|Δψ_raw\| − \|Δψ_TEST\| | — | +0.226 | — | +0.056 |

Decomposition of the top-100 gap (example pairs): cells with no reads at the ATSE, which enter the type mean at their program prior φ_c ψ, dominate: TMS 87 of 100 rows (61 of the 62 with gap > 0.3), median 0.288, with 23 % and 7 % of the two types' cells covered; Allen 46 of 100 (16 of 19), with covered-cell shrinkage second (39 of 100).

**(c) Cause: model smoothing**, by the pre-stated rule (LeafletFA's error vs TEST at the top exceeds raw's on the same rows), in both atlases, stronger in TMS. Not selection: LeafletFA's top-ranked differences are smaller than TEST's, not larger (raw's own ranking does show selection, +0.226 TMS). Not raw noise: on these rows raw TRAIN agrees with TEST closely. The ranking finds real differences (sign agrees with TEST as often as raw's), but LeafletFA compresses their size, mainly through uncovered cells; the same mechanism as the *Fgfr2* uncovered values and the 10.3 undercoverage. Across all events LeafletFA stays closer to TEST than raw, consistent with 10.1, which stands. TEST cells are new cells of the same mice.

## LeafCutter descriptive control for 10.2 (record in VALIDATION_PREREG.md, written before any run; `c10_leafcutter.py`, `c10_lc/`)
Descriptive; the pre-registered verdict (10.2 fails) stands. (ii) only; 141 types with ≥ 4 mice in each half (TMS 100, Allen 41); arrays 22780121, 22780122, smoke 22725798, 22725799; **72.7 CPU-h** (ceiling 200), peak 1.1 GB per task; scoring jobs 22783265, 22783266. Each method is scored on its own calls. Intervals: type bootstrap (2,000).

| | TMS LeafCutter | TMS LeafletFA (same 100 types) | Allen LeafCutter | Allen LeafletFA (same 41 types) |
|---|---|---|---|---|
| Calls in half A (types with calls) | 120,472 (99) | 27,762 (100) | 31,301 (41) | 25,428 (41) |
| Real, confirmed (ii) | 0.521 (0.505–0.538) | 0.819 (0.797–0.839) | 0.740 (0.705–0.776) | 0.858 (0.823–0.893) |
| Null, within mouse | 0.348 (0.336–0.361) | 0.422 (0.407–0.436) | 0.604 (0.565–0.646) | 0.795 (0.763–0.831) |
| Null, ignoring mouse | 0.337 (0.326–0.348) | 0.394 (0.384–0.405) | 0.229 (0.210–0.251) | 0.296 (0.275–0.319) |
| Real − null within mouse | 0.173 (0.152–0.196) | 0.397 (0.367–0.423) | 0.136 (0.112–0.161) | 0.063 (0.044–0.080) |

- Neither method reaches the pre-registered rule (≥ 0.5 and ≥ 5× the within-mouse null) in either atlas: real / null is TMS 1.5 (LeafCutter) and 1.9 (LeafletFA), Allen 1.2 and 1.08.
- Allen: both methods' within-donor nulls are inflated and fall to 0.23–0.30 when donors are ignored, so the inflation is a property of the Allen design (donors sampled by Cre line), not of LeafletFA.
- TMS: LeafCutter makes 4.3× more calls on the same types, and fewer of them confirm.
- Not comparable event for event: the call sets differ. LeafCutter's δ is applied to its own deltapsi over tested introns.

## Check 13 — group-pooled cell-type estimate (amendment 8 and its note 1; `c13_pooled.py`, `c13_pooled/`)
Jobs 22814975 (TMS, 6 min), 22814976 (Allen, 34 min, peak 20 GB). Asserts passed: current = saved check 10 point; covered-only and
ρ̄ match a per-cell rebuild; the 10.1 units and the 10.3 junction counts are reproduced exactly.

| Readout | TMS | Allen |
|---|---|---|
| Top-100 error vs TEST, median of 20 random pairs: current / covered-only / pooled / raw TRAIN | 0.224 / 0.178 / **0.253** / 0.237 | 0.123 / 0.127 / **0.100** / 0.101 |
| Top-100 \|Δψ\| − \|Δψ_TEST\|: current / covered-only / pooled / raw | −0.121 / −0.017 / +0.253 / +0.226 | −0.081 / −0.077 / +0.048 / +0.056 |
| Top-100 TEST-evaluable rows per pair (median): current / covered-only / pooled / raw | 51 / 25 / 23 / 22 | 75 / 46.5 / 39.5 / 35.5 |
| Overall 10.1 error: current / covered-only / pooled / raw | 0.0940 / 0.0955 / 0.1004 / 0.1012 | 0.0625 / 0.0610 / 0.0640 / 0.0647 |
| Pooled − current, overall (95 % CI over types) | +0.0064 (+0.0042, +0.0084) | +0.0014 (−0.0001, +0.0031) |
| Pooled Beta coverage, 100–499 / ≥ 500 TEST reads (check 10 interval) | 0.178 / 0.164 (0.136 / 0.218) | 0.393 / 0.418 (0.256 / 0.346) |
| Pooled median interval width, 100–499 / ≥ 500 | 0.023 / 0.010 | 0.038 / 0.012 |

**Pre-registered reading: pooled does not improve on current.**
- (a) Top-100 error lower in both atlases: **fails** (TMS higher, 0.253 vs 0.224; Allen lower, 0.100 vs 0.123).
- (b) Overall no worse (CI of pooled − current not entirely above 0): TMS **fails** (CI entirely above 0); Allen holds.
- Coverage, descriptive (amendment 8 note 1): 0.16–0.42, not within 0.05 of nominal in any bin. It has no consequence
  for check 12.
- Descriptive, outside the reading:
  - The pooled estimate sits within about 0.01 of raw TRAIN at the top of its own ranking. It inherits raw's selection
    (|Δψ| above TEST's, +0.25 in TMS) instead of LeafletFA's shrinkage (−0.12).
  - Each estimate's top 100 is a different set of rows, with different TEST-evaluable counts (row above).
  - Covered-only has the lowest TMS top-100 error and is close to current overall (no pass rule for it).
- **Decision (Karin 2026-11-07):** reading accepted; the current estimate stays in the catalog and the browser.
  Covered-only is recorded as descriptive only, with no switch and no new rule.

## Check 12 — result (amendment 7 with note 1, R = 1; `c12_intervals.py`, `tsv/c12_*`, `c12/`)
Jobs 22816659 (TMS, 40 GB requested, peak 13.9 GB, 1.9 h) and 22815095 (Allen, 28 GB, peak 24.4 GB, 5.3 h); 1 CPU each,
**7.2 CPU-h** in total (under the 30 CPU-h cap). The first TMS job (22815094, 12 GB) ran out of memory and produced no output.
200 draws per dataset, no Dirichlet underflow. Coverage at ≥ 100 TEST reads, combined draws, DM (a):

| | TMS 100–499 | TMS ≥ 500 | Allen 100–499 | Allen ≥ 500 |
|---|---|---|---|---|
| Junctions (all units) | 563,548 | 1,266,387 | 783,901 | 746,538 |
| Check 10 interval | 0.136 | 0.218 | 0.256 | 0.346 |
| (a) binomial | 0.586 | 0.373 | 0.766 | 0.635 |
| **(a) DM**, all | 0.863 | 0.838 | 0.891 | 0.823 |
| (a) DM, half A / half B | 0.867 / 0.868 | 0.856 / 0.860 | 0.891 / 0.898 | 0.855 / 0.867 |
| **(a)+(b), τ² from A, read on B** | **0.963** | **0.956** | **0.963** | **0.961** |
| **(a)+(b), τ² from B, read on A** | **0.958** | **0.950** | **0.960** | **0.958** |
| τ² chosen (A; B) | 0.01024; 0.01024 | 0.00256; 0.00256 | 0.00128; 0.00128 | 0.00032; 0.00032 |
| Median width, (a) DM / (a)+(b) | 0.104 / 0.225 | 0.069 / 0.131 | 0.087 / 0.121 | 0.034 / 0.074 |

**Pass rule** ((a)+(b) coverage in [0.90, 1.00] in both read bins, both datasets, both directions): **passes**
(all eight cells 0.950–0.963).
- (a) alone does not pass: 0.82–0.90 (Allen 100–499 is 0.891).
- **Pre-registered reading: "(a)+(b) passes, (a) fails → ship (a)+(b), with τ² reported."**
- Descriptive, outside the rule:
  - The parameter-only draws give the same picture ((a)+(b) 0.950–0.972).
  - The (a)+(b) subsets by mice per type and by TRAIN reads all stay within 0.91–0.99. The lowest is TRAIN reads
    < 1,000 at ≥ 500 TEST reads (0.911–0.948, 384–8,592 junctions).
  - τ² is the same in both directions in every bin. τ (SD on the ψ scale) is 0.10 / 0.05 for TMS and 0.036 / 0.018
    for Allen. The calibrated intervals are about 2× as wide as (a) DM in TMS and 1.4–2.2× in Allen.

### Check 12 export and estimator port (Karin 2026-11-07; `c12_export.py`, `catalog_port_check.py`, `c12/export/`)
- **Export** (jobs 22869947, 22869948). The catalog TSVs now carry the (a)+(b) interval for the check 12 units only: TMS 1,829,935 of 6,478,290 rows, Allen 1,530,439 of 4,420,578. Other rows get NA plus `interval_status`. New columns: `test_reads`, `test_cells`, `raw_psi`, `junction_reads`, `mice`; `reads` is renamed `atse_reads`.
  - Asserts passed:
    - the check 12 cross-half coverage is reproduced exactly from the saved arrays;
    - τ² is the same in both directions;
    - TEST reads match check 12;
    - the units are exactly the ≥ 100 reads, ≥ 10 cells set;
    - the old columns are unchanged.
  - Descriptive, not out-of-sample (τ² was set on halves of the same TEST cells). The shipped interval covers all-TEST pseudobulk at TMS 0.958 / 0.949 and Allen 0.959 / 0.949 (100–499 / ≥ 500 reads).
- **Re-export without the call columns** (Karin 2026-11-07; jobs 22879099, 22879100). The Zenodo TSVs drop `tv`, `specific` and `call_eligible`, which stay only in the internal check 10 export.
  - Row counts, interval rows and asserts are unchanged; the interval arrays are byte-identical.
  - New md5: TMS d5ad92c81dec67ada5c942b09a1eb45d (167 MB), Allen 1969a84f2efd275dbc28372c34663b98 (117 MB).
- **Port** (jobs 22870583, 22870584). `leafletfa.catalog.celltype_psi` reproduces the saved check 10 point estimates: max |diff| is 5.9e-8 in both atlases. ATSE reads, junction reads, cells with reads, mice, cells and raw ψ all equal the browser's TRAIN tables exactly. **Pass.**

## Check 14 — simple-shrinkage comparators for 10.1 (amendment 9; `c14_shrinkage.py`, `c14_shrinkage/`)

Pre-registered before any code ran (07a0d56). Jobs 23263717 (TMS), 23263718 (Allen), about 2 min each, 2 CPUs. Facts only.

**Step 0:** no earlier check computed a simple-shrinkage comparator (check 13 shrinks toward the program prior, a
LeafletFA quantity).

**Setup check:** raw error recomputed from counts matches the saved 10.1 `err_raw` on every unit (max |diff| 1.1e-16,
both atlases). Same 10.1 units: TMS 1,038,848 (130 types), Allen 763,823 (68 types).

**B's c, chosen on the inner TRAIN split (TEST not used):** TMS **c = 200**, Allen **c = 100**; neither at a grid end.
Inner mean error (TV), c = 0 → chosen: TMS 0.10116 → 0.09759 (c = 500 gives 0.09759 to 5 decimals; 200 is the exact
minimum); Allen 0.06460 → 0.06252.

**Overall** (mean TV error vs TEST pseudobulk; differences with 95 % bootstrap CI over cell types):

| Atlas | LeafletFA | Raw | A (pseudocount) | B (shrink to across-type mean) | LeafletFA − A | LeafletFA − B | LeafletFA − raw (10.1) |
|---|---|---|---|---|---|---|---|
| TMS | 0.09399 | 0.10121 | 0.10352 | 0.09687 | −0.00953 (−0.01198, −0.00704) | **−0.00288 (−0.00435, −0.00135)** | −0.00722 (−0.00939, −0.00495) |
| Allen | 0.06255 | 0.06467 | 0.06559 | 0.06194 | −0.00305 (−0.00497, −0.00126) | **+0.00060 (−0.00077, +0.00188)** | −0.00213 (−0.00386, −0.00046) |

**Per bin, LeafletFA − B (95 % CI):**

| Bin | TMS | Allen |
|---|---|---|
| 10–49 TRAIN cells | −0.01016 (−0.01198, −0.00860) | −0.00374 (−0.00469, −0.00281) |
| 50–199 | −0.00863 (−0.01016, −0.00718) | −0.00365 (−0.00439, −0.00302) |
| 200–999 | −0.00282 (−0.00447, −0.00136) | +0.00008 (−0.00075, +0.00100) |
| ≥ 1,000 | +0.00647 (+0.00438, +0.00908) | +0.00654 (+0.00519, +0.00804) |
| 0 TRAIN reads | −0.01376 (−0.01639, −0.01077) | −0.00954 (−0.01724, −0.00001) |
| 1–19 | −0.01563 (−0.01732, −0.01375) | −0.00941 (−0.01290, −0.00630) |
| 20–99 | −0.01367 (−0.01488, −0.01252) | −0.00507 (−0.00594, −0.00426) |
| ≥ 100 | −0.00164 (−0.00315, −0.00008) | +0.00203 (+0.00056, +0.00335) |

- At 0 TRAIN reads B equals raw's population fallback (p_j), so the 0-read row of B equals LeafletFA − raw there.
- A at 0 reads is uniform 1/m_e, as defined (error 0.50 TMS, 0.47 Allen in that bin); this drives A's overall error.
- Per-bin and LeafletFA − A rows do not enter the reading.

**Reading (pre-registered):** "the model adds beyond shrinkage" needs LeafletFA − B below 0 with the CI excluding 0
overall in **both** atlases. TMS meets it; **Allen does not** (+0.0006, CI −0.0008 to +0.0019). **Verdict: not met.**
Per the amendment, the result is reported as is and the text says smoothing accounts for the gain.
- Facts beside the verdict: in TMS LeafletFA − B is about 40 % of LeafletFA − raw (−0.0029 vs −0.0072); in Allen,
  B alone is slightly better than LeafletFA overall. Below 200 TRAIN cells and below 100 TRAIN reads LeafletFA is
  better than B in both atlases; at ≥ 1,000 TRAIN cells B (and raw) are better than LeafletFA in both.
- Limits (stated in advance): TEST cells are new cells of the same mice; one c per atlas; B borrows only across cell
  types at the same event.
- Outputs: `c14_shrinkage/{TMS,Allen}_{accuracy.tsv,units.tsv.gz,inner_grid.tsv,summary.json}`. No figure changed.

### Check 14 — reference levels (amendment 9 note 1; `c14_reference.py`, jobs 23263938, 23263939; not tests)
Same 10.1 units; mean TV vs TEST pseudobulk with the 10.1 bootstrap CI (cell types, 1,000, seed 0). Floor defined on
every unit (no unit had all 10 splits with an empty half). LeafletFA and raw means asserted equal to the saved 10.1 table.
"Closed" = share of the gap from "no cell-type information" (upper) to the noise floor covered by the method:
(upper − method) / (upper − floor).

| Atlas | Bin | Upper | Raw | LeafletFA | Floor | Closed: raw | Closed: LeafletFA | Catalog rows |
|---|---|---|---|---|---|---|---|---|
| TMS | 10–49 cells | 0.119 | 0.126 | 0.107 | 0.066 | −0.12 | 0.24 | 15 % |
| TMS | 50–199 | 0.114 | 0.118 | 0.103 | 0.064 | −0.08 | 0.23 | 36 % |
| TMS | 200–999 | 0.103 | 0.099 | 0.093 | 0.059 | 0.08 | 0.22 | 35 % |
| TMS | ≥ 1,000 | 0.087 | 0.076 | 0.082 | 0.049 | 0.28 | 0.14 | 14 % |
| TMS | 0 reads | 0.164 | 0.164 | 0.150 | 0.081 | 0 | 0.17 | 0 % |
| TMS | 1–19 | 0.169 | 0.215 | 0.152 | 0.078 | −0.50 | 0.18 | 16 % |
| TMS | 20–99 | 0.154 | 0.160 | 0.136 | 0.075 | −0.08 | 0.23 | 21 % |
| TMS | ≥ 100 | 0.098 | 0.094 | 0.089 | 0.057 | 0.10 | 0.21 | 63 % |
| Allen | 10–49 cells | 0.081 | 0.087 | 0.075 | 0.063 | −0.32 | 0.31 | 14 % |
| Allen | 50–199 | 0.088 | 0.083 | 0.075 | 0.058 | 0.17 | 0.43 | 43 % |
| Allen | 200–999 | 0.070 | 0.063 | 0.061 | 0.047 | 0.30 | 0.40 | 24 % |
| Allen | ≥ 1,000 | 0.054 | 0.043 | 0.048 | 0.033 | 0.53 | 0.26 | 20 % |
| Allen | 0 reads | 0.186 | 0.186 | 0.176 | 0.101 | 0 | 0.11 | 0 % |
| Allen | 1–19 | 0.157 | 0.200 | 0.146 | 0.099 | −0.74 | 0.19 | 25 % |
| Allen | 20–99 | 0.110 | 0.112 | 0.100 | 0.077 | −0.07 | 0.31 | 25 % |
| Allen | ≥ 100 | 0.062 | 0.052 | 0.053 | 0.040 | 0.43 | 0.40 | 50 % |

- Values rounded from `c14_shrinkage/{TMS,Allen}_abs_panel.tsv` (closed fractions computed from the unrounded values).
- At 0 TRAIN reads, raw is the pooled TRAIN ψ (its fallback), so raw = upper there by construction.
- **At 1–19 TRAIN reads raw is worse than using no cell-type information** (TMS 0.215 vs 0.169; Allen 0.200 vs 0.157); LeafletFA is below the upper reference in every bin.
- Catalog rows = shipped catalog rows (junction × type) by TRAIN cells per type and TRAIN reads at the ATSE; no catalog row has 0 TRAIN reads.
- LeafletFA closes 11–43 % of the gap from no cell-type information to the noise floor, depending on bin; the floor is approximate (split halves scaled by √(n_A n_B)/(n_A + n_B); limits in the note).

### Fact: cell types with TRAIN and TEST cells from ≥ k mice (Karin 2026-11-13; `mice_per_type_fact.py`, job 23263810; no analysis)

| Atlas (mice/donors) | ≥ 2 | ≥ 3 | ≥ 4 | ≥ 5 |
|---|---|---|---|---|
| TMS types (of 132), 18 mice | 131 | 126 | **123** | 116 |
| TMS share of rows with an interval | 1.000 | 1.000 | **1.000** (1,829,401 of 1,829,935) | 0.956 |
| Allen groups (of 68), 512 donors | 68 | 65 | **60** | 53 |
| Allen share of rows with an interval | 1.000 | 0.991 | **0.970** | 0.934 |

- Mice counted per type from TRAIN and TEST obs labels (`mouse.id`). The catalog's `mice` column is per row (mice with reads at the event), not per type.
