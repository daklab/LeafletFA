# LeafletFA status since Wed 18 Nov 2026, 08:00 PT (written 25 Nov; facts only)

## 1. Commits since 18 Nov 08:00 PT
- **None** in LeafletFA (`leafletfa-validation`), Leaflet-analysis-bioinformatics (`bioinformatics`), Leaflet-analysis,
  Leaflet-analysis-methods-fixes, LeafletFA_overleaf, LeafletSC, ATSEmapper. (SpliceVI-utils: 63 commits, SpliceVI work.)
- **Still unpushed from 15–16 Nov:**
  - LeafletFA: **14 commits ahead**. Last pushed `cb3b05d` (check 16b report). Unpushed include checks 17, 18, 19 and the
    Allen sense checks: `66cd49d` … `713883d`.
  - Leaflet-analysis-bioinformatics: **9 ahead**. Last pushed `0281c16` (first status PDF). Unpushed include figures
    `2e1942a` `f314705` `c1a23ac`, OUTLINE `9b081cf`, bib `9f2a637`, gallery `54bbf47`, browser notes `3e4dbb0` `7c86aa9`,
    PDF `11758ed`.
  - Leaflet-analysis-methods-fixes: 1 ahead (older).

## 2. Results against pre-registered rules (all run 15–16 Nov, before this window; none since)

| Check (amendment) | Rule | Result | Verdict | Jobs |
|---|---|---|---|---|
| 16 type vs rest, held-out (am. 12) | all lower bounds > −0.05 vs raw and LeafCutter | ≥ 500 vs raw −0.028 [−0.077] | **FAIL** | 23361314 (smoke, code bug), 23362086, 23362809 |
| 16b design 1 (am. 12 n. 1) | same | all lower bounds > 0 | **BETTER** | 23365160, 23365161 |
| 16b design 2 | same | 100–499 vs raw −0.081 [−0.150] | **FAIL** | same |
| 16b combined reading | lower bound > 0 at 1–19 and 20–99 in both designs | design 2 at 20–99 [−0.024] | "at least partly explained by concentration" | same |
| 17 pairs, TMS (am. 13) | rule of 16 | vs raw +0.055 to +0.079, lower bounds > 0; vs LC +0.081 [+0.008], +0.061 [+0.004] | **BETTER** | 23367290, 23367291 (LC, 4.5 CPU-h), 23367292 |
| 18a pairs, mouse-held-out refit (am. 14) | rule of 17 | vs raw +0.088 / +0.068 / +0.094 / +0.061, all > 0; vs LC +0.156 [+0.093] | **BETTER** | split 23370697 (cancelled), 23370796; GPU fit 23370799; LC 23370800–01; score 23370802 |
| 18b type vs rest, refit | rule of 17 | vs raw +0.214 / +0.087 / +0.079 / +0.036 [−0.006] | **BETTER** | same |
| 19 Allen, ≥ 4 shared donors (am. 15) | rule of 17, within donors | vs raw +0.029 / +0.091 / +0.060 / +0.064, all > 0 | **BETTER** | 23370837, 23370838 (smoke), 23371384 |

**Allen sense checks:** descriptive, no rule. Recorded `a2a0d5d`. Job 23371679. Superseded runs, none used: 23371006,
23371598, 23371611, 23371627, 23371630, 23371653.
- **Positive controls:** *Nrxn1/2/3* SS4 median rank 16 / 94.5 / 152, expected direction in 97 / 100 / 99 % of
  exc × inh pairs. *Grin1* exon 5 rank 221, *Gria2* flip/flop rank 688; their expected direction is UNSOURCED.
- **Neuron vs glia:** Irimia microexons 48× (98 % neuronal direction); Irimia exons > 27 nt 18× (97 %); Rbfox targets 9× (88 %).
- **Hierarchy:** fraction of ATSEs with \|Δψ\| ≥ 0.1 goes 0.259 (neuron vs glia) > 0.079 > 0.024 > 0.004 (same
  subclass). Cophenetic Spearman 0.81; adjusted Rand 0.33 (class) and 0.22 (subclass).
- **Frame:** skipped exons 0.816 vs background 0.478 (holds). Alt 5′ 0.320 vs 0.354 and alt 3′ 0.405 vs 0.467 (no
  enrichment; alt 3′ depleted).

**Still running:** nothing (squeue holds only interactive bash 23629898).

**Never started:** coordinated-shift simulation and call-set biology (DS items 3–4, not approved); LeafCutter type vs
rest in 18b (out of scope); Nova and Srrm4 lists (not obtained); refit on 100 % of cells (deferred).

## 3. Numbers that contradict or qualify VALIDATION_REPORT.md as of 18 Nov
- **No later computation changed a reported number.** No LeafletFA commits since 18 Nov.
- **Qualifications already recorded in it:**
  - Composite mouse labels (`3_10_M/3_11_M`, `3_38_F/3_39_F`): the 10.2a halves used by 10.2, 10.7, 12, 16 and 17 put
    106 of about 78,000 TMS cells opposite one of their possible mice (amendment 14, note 1).
  - 10.2 (ii), 10.7 and the LeafCutter control confirmed on half-B TRAIN cells, which the fit saw (amendment 12).
- **Stale elsewhere:**
  - `OUTLINE_PREP.md` still says "check 18: pending".
  - The browser notes cite checks 16, 16b and 17, not 18 or 19.
  - Fit time differs by source: report 13.2 / 38.5 min (seed 0) vs RUNTIME_TABLE 12.9 / 37.7 min (seeds 1–2).
- **Wrong preprint DOI:** `10.1101/2025.02.20.639398` in the SpliceVI bib is Tahoe-100M, not the LeafletFA preprint.

## 4. Browser, simulation, scaling
- **Browser (daklab/leafletfa-browser, option B):**
  - The repo is not visible: the GitHub API returns 404 (not created, or private).
  - The site is the 10 Nov preview (`portal_browser/site`, 597 MB) on the 70 % TRAIN fit (seed 0, K = 20).
  - Raw TRAIN pseudobulk ψ is shown beside each estimate in the compare and one-vs-rest views and their TSV downloads.
  - The source notes were updated for checks 16–17 (`3e4dbb0`, `7c86aa9`); the site was not rebuilt.
  - Not yet in it: rank-band rates (17–18); Allen donor counts and the "not separable" flag (`c19/allen_pairs_index.tsv`
    is ready); "validated" / "computed, not validated" labels.
- **Simulation:** candidate supplementary figure from checks 15 and 15b
  (`paper_figures/figS_simulation/panels/Ssim_a_review.png`, `2e1942a`). No new simulation.
- **Scaling:**
  - `scaling/scaling.tsv`, plus the supplementary figure `figS_scaling/panels/Sscale_a_review.png`.
  - The full-recipe table is in the status PDF section 6.
  - First recorded peak GPU memory for the production recipe: 6.6 GB at 46,479 cells (check 18 refit).
  - Production fits (54,694 / 38,199 cells): GPU peak not recorded.

## 5. Crecerelle (tuVI / TRVI, unobserved junctions): answered 18 Nov
- Source: `~/SpliceVI_paper_notes/working_notes/LANGUAGE_PASS.md` §6, from fa5e04b, Zenodo 22959716 and Weberling
  2026 (doi 10.64898/2026.09.23.753495).
- Encoder: log1p counts, unobserved junctions entered as 0, no mask. DM: an ATSE with N = 0 contributes exactly 0.
  ZIDM adds an excess-zero term (all TRVI analyses used ZIDM); TRVI expression ZINB. `conceptual-model.md` is silent.
