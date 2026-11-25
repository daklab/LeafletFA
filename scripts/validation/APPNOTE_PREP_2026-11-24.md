# Application Note prep (for writing Thu 26 – Fri 27 Nov 2026); facts only, no prose

**Source keys used below:**
- **R** = `~/LeafletFA/scripts/validation/VALIDATION_REPORT.md` (section named).
- **L** = ledger `Bioinformatics2026/manuscript/numbers.tex`, generated from `ledger/results_ledger.tsv`.
- **NL** = not in the ledger yet; it must be added before the number goes in the text.

## 1. Claims table

### Claims the note can make

| Claim | Check | Result | Exact numbers | Source |
|---|---|---|---|---|
| One catalog over two atlases | — (descriptive) | built | 132,706 cells (TMS 78,135; Allen 54,571); 79,768 junctions; 24,475 ATSEs; 11,233 genes; ATSE classes alt 3′ 24 %, alt 5′ 20 %, skipped exon 23 %, complex 33 % | L (`CatalogCellsMouse` …, `EventShare*`) |
| Catalog export | 10.5 | done | 6,478,290 (TMS) / 4,420,578 (Allen) rows; interval on 28.2 % / 34.6 % of rows | R "Check 12 export", "Fact: 20–99 band"; NL |
| Held-out accuracy | 10.1 | PASS / PASS | TV error TMS 0.0940 vs raw 0.1012, Δ −0.0072 [−0.0094, −0.0050]; Allen 0.0625 vs 0.0647, Δ −0.0021 [−0.0039, −0.0005]; at 1–19 reads Δ −0.0625 / −0.0543 | R "Check 10"; NL |
| Recalibrated intervals | 12 | PASS (shipped) | (a)+(b) coverage 0.950–0.963 in all 8 cells (2 read bins × 2 atlases × 2 directions); τ² TMS 0.01024 / 0.00256, Allen 0.00128 / 0.00032 | R "Check 12"; NL |
| Pairwise DS, TMS (mice unseen by the fit) | 18a | BETTER | LeafletFA − raw excess +0.088 / +0.068 / +0.094 / +0.061 (1–19 / 20–99 / 100–499 / ≥ 500); vs LeafCutter +0.156 [+0.093] at 100–499 (49 pairs), +0.059 [−0.0002] at ≥ 500 (50 pairs) | R "Check 18"; NL |
| Pairwise DS, TMS (full-data fit) | 17 | BETTER | vs raw +0.079 / +0.068 / +0.070 / +0.055, all lower bounds > 0; vs LeafCutter +0.081 [+0.008], +0.061 [+0.004] (41 / 48 pairs) | R "Check 17"; NL |
| Type vs rest, TMS (mice unseen by the fit) | 18b | BETTER | vs raw +0.214 / +0.087 / +0.079 / +0.036 [−0.006] | R "Check 18"; NL |
| Allen pairs with ≥ 4 shared donors | 19 | BETTER | vs raw +0.029 / +0.091 / +0.060 / +0.064, all > 0 (564 pairs) | R "Check 19"; NL |
| Allen differences behave like known biology | sense checks (descriptive) | — | *Nrxn1/2/3* SS4 in the expected direction in 97 / 100 / 99 % of excitatory × inhibitory pairs (median rank 16 / 94.5 / 152); Irimia 2014 microexons 48× enriched, 98 % neuronal direction; share of ATSEs with \|Δψ\| ≥ 0.1: 0.259 (neuron vs glia) → 0.004 (same subclass), cophenetic Spearman 0.81; skipped-exon frame 0.816 vs 0.478 | R "Allen sense checks"; NL |
| Speed and scaling | 10.6, scaling record | descriptive | Fits on one RTX PRO 6000 Blackwell (seeds 1–2, RUNTIME_TABLE.md; Karin 25 Nov; ledger `FitMinutesTMS`, `FitMinutesAllen`): TMS 12.9 min, Allen 37.7 min; 132,706 cells 73.2 min. GPU 0.032–0.046 s/step, flat ≥ 10,000 cells. CPU 102× slower per step. scQuint 86.0 min (16 CPUs). LeafCutter catalog benchmark 19.2 task-h / 77 core-h (different product: per-type tests) | R "Check 10" 10.6, "Scaling and runtime record"; NL |
| Known biology, TMS | 10.4 | PASS | *Fgfr2* IIIb: epithelial min 0.704 > mesenchymal max 0.649; *Ptprc* 25 of 27 confirmed; *Nrxn1–3* SS4 in 1,256 / 946 / 1,129 of 1,891 pairs | R "Check 10"; NL |

### Claims the note cannot make, and the check that rules each out

| Claim | Ruled out by | What the check found |
|---|---|---|
| Type-specific calls replicate (≥ 0.5 and ≥ 5× null) | 10.2 | FAIL; LeafCutter fails the same rule |
| Pairwise calls replicate (same rule) | 10.7 | FAIL |
| A group-pooled estimate beats the current one | 13 | Not adopted |
| LeafletFA beats simple shrinkage in both atlases | 14 | Not met: TMS better, Allen tie |
| More accurate than raw at high depth, or ranks better than LeafCutter, in simulation | 15b | Worse than raw and B; LeafCutter better on shared units |
| Type vs rest non-inferior to raw at ≥ 500 reads (full-data fit) | 16 | FAIL; lower bound −0.077 |
| The type-vs-rest gain is free of concentration in a few types | 16b | "at least partly explained by concentration" |
| Programs are reproducible units | 3, 5a, 6, 8 | — |
| The posterior DS test is reproducible | 9a | — |
| DS gain for the same cell type across tissues below 100 reads | 17, 18a | No gain; CIs include 0 |
| *Grin1* exon 5 and *Gria2* flip/flop as positive controls | Allen sense checks | Expected direction UNSOURCED (*Gria2* rank 688) |
| Frame preservation for alt 5′ / alt 3′ sites | Allen sense checks | No enrichment (0.320 vs 0.354; 0.405 vs 0.467) |
| Type vs rest at ≥ 500 reads as a gain (limit, not a claim) | 16, 16b, 18b | 16 FAIL; 16b concentration; 18b ≥ 500 lower bound −0.006 |

## 2. Journal format (Bioinformatics, OUP)
- **Source:** https://academic.oup.com/bioinformatics/pages/author-guidelines (fetched 25 Nov 2026). The
  `instructions_for_authors` page now points there.
- **Length:** "(Up to 4 pages; this is approx. 2,600 words or 2,000 words plus one figure)".
  - The page does not say what counts toward the limit for Application Notes.
- **Required content:**
  - "Availability and Implementation must be clearly stated in the article."
  - "The name of the application should be included in the title."
  - Supplementary material "should be referred to in the abstract of the Application Note."
- **Availability:**
  - "Software or data must be freely available to non-commercial users."
  - "available for a full two years following publication."
  - Source code "freely available at stable URLs (e.g. github, bitbucket)".
  - The submitted version and test data must be archived "at dedicated repositories such as Zenodo, Figshare, Software
    Heritage…".
  - An open-source license is encouraged.
  - Web servers: "support / maintenance for at least 3 years"; web services "must not require mandatory registration".
- **Data:**
  - "Authors are required to include a data availability statement".
  - "All data on which the conclusions given in the publication are based must be publicly available".
- **Supplement:** online only; "presented in a single file"; it "cannot be altered or replaced after the paper has
  been accepted".
- **VERIFY:** the end of the page was not read (truncated): the "Preparing your manuscript" section. That leaves
  unconfirmed:
  - the structured-abstract headings (Summary / Availability and implementation / Contact / Supplementary information);
  - any reference limit;
  - the exact figure count.
  - Check the live page.

## 3. Figure
**Current figure AN** (183 × 164 mm, built 13 Nov; `paper_figures/figAN/figAN.pdf`):
- a: catalog over the two atlases and the ATSE event types. The model-schematic part is a placeholder.
- b: accuracy against held-out cells (10.1), with the check 14 shrinkage comparator.
- c: *Gria2* flip/flop.
- d: *Fgfr2* IIIb/IIIc.
- e: *Nrxn1* SS4.
- f: *Ptprc*.

**What check 17 or 18 changes:**
- **Panel b** would be replaced or split. Candidates at the same size (85 × 50 mm):
  `figAN_b_candidate_check17.pdf`, `figAN_b_candidate_check18.pdf`.
- Panels c–f are unchanged.
- *Gria2* is the weakest Allen control (median rank 688; expected direction UNSOURCED).

**One alternative layout (not built):**

| Panel | Content | Data source |
|---|---|---|
| a | Schematic + catalog counts | L `Catalog*`, `EventShare*` |
| b | Accuracy vs held-out by read bin, raw and B beside | `$V/c10/*_accuracy_units.tsv.gz`, `$V/c14_shrinkage/*_accuracy.tsv` |
| c | Held-out DS excess by read bin, LeafletFA / raw / LeafCutter, check 18a | `$V/c18/18a_summary.tsv` |
| d | *Fgfr2* IIIb (TMS) | catalog export |
| e | *Nrxn1* SS4 (Allen) | catalog export |
| f | Browser screenshot of one pair (ψ ± interval, raw beside) | `portal_browser/screenshots/` |

## 4. Fact skeleton (one line per paragraph; NL = add to the ledger first)
- **Summary:** atlas of cell-type splicing for 132,706 cells (L), one catalog with predictive intervals; held-out accuracy
  and DS (NL); package + browser + Zenodo (placeholders).
- **Intro 1:** single-cell splicing is sparse; median reads per observed ATSE TMS 20, Allen 4 (L
  `MedianReadsPerObservedATSE*`).
- **Intro 2:** pseudobulk and LeafCutter need depth. LeafCutter can test 0 subset pairs at 20–99 reads in check 18a, and
  13 of 75,292 units at 20–99 in check 16 (NL).
- **Methods, data:** 169,509 cell files → 132,706 cells (L `Flow*`). Split 70/30: TMS 54,694 / 23,441, Allen 38,199 /
  16,372 (L `Split*`).
- **Methods, ATSEmapper:** 39,729 ATSEs from the joint run → 24,475 after the 75-cell filter (L `FlowATSEsATSEmapper`,
  `CatalogATSEsMouse`).
- **Methods, model:** DM factor model, K = 20, 16,200 steps, minibatch 8,192, float64 φ (R check 18 / run_record); NL.
- **Methods, estimate and interval:** type mean of ρ̃; predictive interval with τ² from held-out mice (R check 12); NL.
- **Results, catalog:** counts and event shares (L).
- **Results, accuracy:** 10.1 numbers (NL).
- **Results, intervals:** check 12 coverage 0.950–0.963 (NL).
- **Results, DS:** check 18a / 17 table and the Allen check 19 (NL); sense checks (NL).
- **Results, scale:** TMS 12.9 min, Allen 37.7 min (seeds 1–2; ledger), GPU flat per step, CPU 102× (ledger `ScaleCPUOverGPU`).
- **Availability:** GitHub, PyPI, Zenodo DOI (catalog, fits), browser URL; all placeholders (section 5).
- **Limitations:**
  - No gain for the same cell type across tissues below 100 reads.
  - LeafCutter comparisons rest on 41–50 pairs.
  - 1,094 Allen pairs share no donor (cell type and donor not separable).
  - Programs are not reproducible units (checks 3, 8).
  - Not better than simple shrinkage in Allen (14).
  - Simulation is worse than raw (15b).

## 5. Cover letter facts
- **Earlier preprint:** "Aging-associated alternative splicing programs conserved between human and mouse tissues",
  bioRxiv, **doi 10.64898/2025.12.30.697080** (https://www.biorxiv.org/content/10.64898/2025.12.30.697080v1.full).
  - Found by search, 25 Nov. Not in any local file.
  - VERIFY the posting date and the model description: the search snippet says "Beta-Dirichlet"; the current model is DM.
- **What differs in this submission:**
  - aging and human–mouse focus → a cell-type atlas and catalog for TMS and Allen with calibrated intervals;
  - held-out validation, including differential splicing;
  - a browser.
- **Shared with SpliceVI:**
  - the TMS and Allen Smart-seq2 data and processing (`shared_data_catalog.tex`);
  - ATSEmapper;
  - the 70/30 splits (VERIFY that SpliceVI uses the same).
  - SpliceVI preprint found: doi 10.1101/2025.11.26.690853 (VERIFY that it is the companion to cite). The bib entry
    `SpliceVI2026` is a placeholder.
- **Placeholders still open:**
  - Availability line (`leafletfa_main.tex:65`: GitHub, PyPI, Zenodo DOI).
  - Data availability (`:163`: TMS AWS, Allen GEO GSE185862, Zenodo).
  - Subject section (`:49`).
  - Browser URL (repo not created).
  - `SpliceVI2026` bib entry.
  - `Kotliar2019` marked TO VERIFY.
  - Article DOI is set at production (`:39`).

## 6. Open decisions (recommendation; reason)
- **Panel b → check 18a candidate.** It is the design with no fit leakage; check 17 goes to the supplement.
- **Drop *Gria2* from the main figure; keep it in the gallery.** Its Allen rank is 688 and its expected direction is
  unsourced.
- **DS section budget about 250 words, taken from Limits and Results slack.** It carries the new main claim.
- **Fit times: decided by Karin 25 Nov: 12.9 / 37.7 min** (TMS / Allen; seeds 1–2, same recipe and GPU; `SpliceVI-utils/manuscript/age_analysis/audit_2026_09/RUNTIME_TABLE.md`; ledger `FitMinutesTMS`, `FitMinutesAllen`). Seed 0 is not quoted: jobs 22400619 and 22400622 failed while saving, so their log spans (13.2 / 38.5 min) are not measured fit times.
- **Browser labels:** "validated" for TMS pairs (17–18) and Allen pairs with ≥ 4 shared donors (19); "computed, not
  validated" plus the shared-donor count for the rest. This matches what was tested.
- **Push LeafletFA (14 commits) and bioinformatics (9) before writing.** The text should cite pushed commits.
- **Add the validation numbers to the ledger before writing.** Rule: numbers from the ledger only.
- **Ask David for the daklab/leafletfa-browser repo now.** It is needed for the availability URL.

## 7. Browser rebuild: what it would add (not rebuilt; Karin decides Thursday)
**What it would add.** Code changes are in `portal/web/app.js` and `config.js`, with data shards from `precompute.py` /
`build_portal.py`.
- **Rank-band rates** beside each pair's ranked list: confirmed fraction and excess by rank band (1–5, 6–20, 21–100) and
  read bin.
  - TMS from check 18a: `$V/c18/18a_rank_bands.tsv`. Allen from check 19: `$V/c19/rank_bands.tsv`. LeafletFA rates are ledger macros `RankBand<atlas><band><bin><Confirmed|Excess>` (48 rows, 25 Nov).
  - Shown only where the pair is in a validated class.
- **Allen donor counts:** donors contributing both groups, per pair (`$V/c19/allen_pairs_index.tsv`).
  - Pairs with 0 donors are flagged "cell type and donor not separable; differences may reflect donor or batch".
- **Labels:**
  - "validated (checks 17–18)" on TMS pairs;
  - "validated within donors (check 19)" on Allen pairs with ≥ 4 shared donors;
  - "computed, not validated" on the other Allen pairs.
  - The same cell type across tissues gets a note that there is no gain below 100 reads.
- **Updated notes:** add checks 18 and 19 (currently they cite checks 16–17 only).
- Unchanged: the 70 % TRAIN fit (seed 0); raw ψ beside each estimate; no significance flag.

**Build time** (last measured runs; the precompute steps are independent jobs):

| Step | Job | Time | Peak memory |
|---|---|---|---|
| Precompute catalog | 22652532 | 2.5 min | 0.58 GiB |
| Precompute TMS | 22652533 | 2.9 min | 6.05 GiB |
| Precompute Allen | 22652534 | 3.4 min | 13.8 GiB |
| `build_portal.py all` | 23156524 | 0.3 min | 1.24 GiB |
| Screenshot checks | 23156576 | 0.5 min | — |

- **About 5 min of compute**, plus writing the new views.
- The site is 597 MB today, and the new data are small: 2,278 Allen rows and rank-band tables.
- Publishing also needs the daklab/leafletfa-browser repo, which currently returns 404.
