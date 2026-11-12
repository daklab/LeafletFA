# LeafletFA docs and tutorials: audit and plan (2026-11-07)

The audit was static only: imports, names and keyword arguments resolved against the package, `py_compile`, and path checks. Nothing was executed. Repo HEAD a48a320.
- The current model is described in `METHODS_LEAFLETFA_CURRENT.md`: DM + simplex, `LeafletFAMinibatch.dm(...)`, production driver `scripts/splicevi_baseline/run_leafletfa_baseline.py`.
- `CODE_REVIEW.md:90, 687` (2026-07-20) already flagged most of the doc problems below.

## 1. Audit

**Bottom line:**
- Every user-facing doc page and the quickstart notebook teach the legacy Beta-binomial `LeafletFA` class with waypoints.
- Only `README.md:24-38` describes the shipped model correctly.

| Path | Last commit | Runs? | Outdated? | What is wrong |
|---|---|---|---|---|
| `README.md` | 2026-11-01 | partly | partly | See the list below |
| `docs/index.md` | 2026-05-10 | no | yes | Beta-Dirichlet and aging framing (:3, :23); legacy example raises ValueError (:40-51) |
| `docs/models/leafletfa.md` | 2026-05-10 | no | yes | Same ValueError (:13-16); legacy defaults (:26-36); `num_initializations` passed to the constructor is a TypeError (:33); `varm["psi_learned"]` and `obsm["X_PHI"]` are never written (:44-58) |
| `docs/data-layer/splicingdataset.md` | 2026-05-10 | yes | yes | Calls the cluster matrix a "beta-binomial denominator" (:17, :29); does not say `var.event_id` is required |
| `docs/outputs/splicing-programs.md` | 2026-05-11 | no | yes | Treats ψ as independent per-junction PSI; the \|ψ − 0.5\| marker rule (:29) is wrong under the simplex model |
| `docs/outputs/cell-activities.md` | 2026-05-11 | no | yes | Uses `assign_post` and `X_PHI`, which are never written |
| `docs/outputs/differential.md` | 2026-05-11 | no | yes | `test_differential_activity` (:11) does not exist; the current functions are `differential_splicing_groups.differential_splicing_junctions` and `differential_splicing_atses` |
| `docs/ecosystem.md` | 2026-05-11 | no | yes | Uses `X_PHI` (:7, :12, :20); colors by age (:15) |
| `docs/api/model.md`, `docs/api/waypoints.md`, `docs/api/differential_splicing.md` | 2026-05-10 | resolve | yes | Document only the legacy class and the superseded waypoints; `utils.compute_and_plot_albf` calls the missing `compute_albf` (utils.py:94) |
| `docs/models/splicevi.md`, `docs/data-layer/atsemapper.md` | 2026-05-10 | — | no | Link stubs |
| `mkdocs.yml` | 2026-05-10 | nav targets exist; `mkdocs build` UNVERIFIED | yes | No page for `LeafletFAMinibatch`, E-step or `differential_splicing_groups` |
| `notebooks/quickstart.ipynb` | 2026-05-01 | statically yes, but as the legacy BB model | yes | Fits the Beta-binomial at K = 10; keeps the top 10k junctions, which splits ATSEs and would break the DM; advertises ALBF (a dead path); no DM, E-step or cell-type ψ |
| `METHODS.md`, `METHODS_dirichlet_multinomial.md` | 2026-10-25 | — | history | Both carry a "not current" banner |
| `METHODS_MINIBATCH.md` | 2026-07-25 | — | yes | **No "not current" banner**; Beta-binomial and waypoints throughout |
| `METHODS_differential_splicing.md` | 2026-08-06 | — | UNVERIFIED | DM likelihood-ratio test; not linked from `METHODS_LEAFLETFA_CURRENT.md` |
| `scripts/splicevi_baseline/run_leafletfa_baseline.py` | 2026-10-25 | compiles | current | Defaults match `METHODS_LEAFLETFA_CURRENT.md` §4; `--device` defaults to cpu |
| `experiments/**` (about 35 .py) | ≤ 2026-08-04 | names resolve | dev | Development scripts, not tutorials |
| `notebooks/archive/**` | untracked, 2024–2025 | no | yes | Import modules that no longer exist (`BetaDirichletFactor.*`, `LeafletSC.*`); human and aging inputs; one SyntaxError |

**README.md problems:**
- `:44`: `--lr` is not a driver flag; the flag is `--fit-lr`.
- `:84-99`: the legacy class example raises ValueError.
- `:107`: names the datasets as TMS + EasySci; Allen is missing.
- `:126-128`: claims outputs (`psi_learned`, `X_PHI`, `SP_1..20`) that are never written.
- `:141-164`: calls the cluster matrix a "beta-binomial denominator".
- `:201-210`: the parameter table is for the legacy class.
- `:216`: mentions cross-species transfer.

**API gaps for the catalog use case:**
- **(a) Gene → ATSEs:** no package function.
  - Today it is done ad hoc (`scripts/validation/c10_catalog.py:446`) or by filtering the exported catalog TSV on `gene`.
- **(b) Cell-type ψ from a fit:** only in scripts, not in the package.
  - The estimator is the mean of ρ̃ = (γ_e φ_c ψ + y)/(γ_e + N) over a type's cells: `c10_catalog.py` `Data` (:49-66), `Groups.sums` (:121-132), `estimates` (:145), `raw_pb` (:193), `export` (:453).
  - These depend on hard-coded fit directories (`vlib.paths`, `c9_lib.fit_dir`).
  - The package has only read-free group averages φ̄·ψ in `differential_splicing_groups`.
- **(c) Plotting usage per cell type:** only legacy violins of observed ratios (`leafletfa/analysis_utils/visualization.py:5, :41`).
  - They are not exported, take no DM fit, and don't follow the figure theme.

**Data:**
- The quickstart file `notebooks/mouse_foundation_splicing.h5ad` (1,000 cells × 89,831 junctions) is gitignored, has no fitted results, and carries `aging_*` columns.
- The old Zenodo record (10.5281/zenodo.18158125) was not checked: UNVERIFIED.

## 2. Proposed tutorial: "Look up a gene and plot its cell-type splicing"

**Goal:** reproduce one browser event page in a notebook. Inputs:
- a gene name;
- the ATSEs of that gene;
- per cell type, the LeafletFA ψ with its check 12 interval (95 % predictive for held-out cells at their read depth; NA where held-out data are too thin);
- raw TRAIN pseudobulk ψ, reads and cells with reads beside it;
- a plot with the same encoding as the browser.

**Two entry points, one notebook:**
- **A. From the Zenodo catalog (no fit, CPU, minutes).** Recommended as the main tutorial. Steps:
  1. Load `catalog_{TMS,Allen}_s0.tsv.gz` and the event catalog (`catalog_events.json`, `catalog_junctions.parquet`).
  2. `gene_events("Fgfr2")` lists the ATSEs, with class, junction coordinates and annotation status.
  3. `celltype_table(event_id)` returns one row per cell type × junction:
     - ψ, lo, hi;
     - raw ψ, junction reads, ATSE reads, cells with reads, mice;
     - a grey flag (fewer than 2 mice), using the same rule as the browser.
  4. `plot_event(table)` draws one panel per junction:
     - filled dot = LeafletFA ψ with interval;
     - open ring = raw ψ, sized by reads;
     - types grouped by lineage, using the figure theme (LeafletFA teal, raw grey).
  5. A short "how to read it" cell:
     - Use the raw ring and the read count to judge large differences.
     - Estimates for cells without reads come from the shared programs (the smoothing diagnosed on 2026-11-06).
     - The interval predicts the pseudobulk ψ of held-out cells of the type; it is not an interval for the true cell-type ψ, and it is missing where held-out cells have < 100 reads or < 10 cells with reads.
- **B. From a fit (for a user's own data).** The tutorial must say: a user's own fit gives ψ, raw ψ, reads, cells and mice, but **no interval**. The interval is calibrated on held-out cells of the atlases (check 12) and does not exist for a new fit. Steps:
  1. Fit with the production driver.
  2. Call `celltype_psi(model, adata, groupby="cell_type")`, which computes the same estimator from the fit.
  3. Run the same `celltype_table` and `plot_event` as in A.

**Code needed** (port, not new methods; each function is checked against the saved check 10 output):
- `leafletfa/catalog.py`:
  - `load_catalog`, `gene_events`, `celltype_table`.
  - `celltype_psi`: the estimator from `c10_catalog.py` without hard-coded paths. Point estimate first; intervals as an option.
- `leafletfa/plotting.py`: `plot_event`, matching the browser's encoding and `FIGURE_THEME.md`.
- **Test:** `celltype_psi` on the TMS seed-0 fit reproduces `c10/TMS_s0_estimates.npz` `point` (max difference < 1e-4), and `celltype_table` matches the browser's per-event TSV download for one ATSE.
- **Entry point A blocker resolved (2026-11-07):** the catalog TSVs now carry `raw_psi`, `junction_reads`, `mice`, the check 12 interval, `interval_status`, `test_reads` and `test_cells`. The failed 10.2 call columns (`tv`, `specific`, `call_eligible`) are dropped (Karin 2026-11-07).
- **Done (2026-11-07):** `leafletfa/catalog.py` with `load_fit` and `celltype_psi` (point estimate, raw ψ, reads, cells, mice; no interval, since the check 12 interval needs held-out cells). Tests: `tests/test_catalog.py` (per-cell definition, toy data) and `scripts/validation/catalog_port_check.py` (saved check 10 point, both atlases).
- Not yet written: `load_catalog`, `gene_events`, `celltype_table`, `plot_event`. The interval labels in the tutorial must follow the check 12 wording.

**Worked examples:**
- *Fgfr2* IIIb/IIIc (TMS, epithelial vs mesenchymal, covered vs uncovered cells).
- *Nrxn1* SS4 (Allen).
- Both match the main figure, so a reader can check against it.

## 3. Doc cleanup (proposed order)

1. **README:**
   - Fix the `--fit-lr` flag.
   - Remove or label the legacy-class section.
   - Fix the claimed outputs.
   - Add Allen to the datasets.
   - Replace the parameter table with the driver defaults.
2. **Docs index and model page:** rewrite around `LeafletFAMinibatch.dm(...)`, E-step and cell-type ψ.
   - Move the legacy class to a "Legacy (Beta-binomial)" page.
3. **Outputs pages:**
   - Programs: simplex ψ, no \|ψ − 0.5\| rule, no named-program claims (checks 3 and 8).
   - Differential splicing: the current functions, labeled exploratory (checks 9 and 10.2 failed).
   - Delete cell-activities and ecosystem references to `X_PHI`, or write those outputs.
4. **`METHODS_MINIBATCH.md`:** add the "not current" banner.
5. **mkdocs:**
   - Drop `waypoints` from the API nav.
   - Add `minibatch`, `differential_splicing_groups`, `catalog` and `plotting`.
   - Run `mkdocs build` through sbatch.
6. **Quickstart:**
   - Replace it with the tutorial above, entry point A.
   - Keep a short DM fit on the 1,000-cell file, without junction subsetting that splits ATSEs, as entry point B.

## 4. Walkthrough: "How LeafletFA works" (plan only; nothing built, no values looked at; Karin 2026-11-12)

**Goal:** one event, followed from raw reads to the catalog value, with pictures. It doubles as a blog post.

**Event:** *Nrxn1* SS4, Allen (`ENSMUSG00000024109.18_atse_7`; SpliceVI §111 definition, the main-figure panel e). Fallback: *Gria2* flip/flop (`atse_3`) if Karin prefers an MXE.

**Two cell groups:** one excitatory and one PV group, fixed now: L2/3 IT CTX and Pvalb (the browser's example pair).

**Steps, one picture each, in the figure theme (LeafletFA teal, raw gray):**
1. **The event.** Exon structure (GENCODE vM19) with the SS4 exon and its inclusion and skipping junctions, drawn as in the browser.
2. **Raw reads in single cells.**
   - Shown: a fixed random sample of 12 TRAIN cells per group (seed 0, drawn from all cells of the group, including cells with no reads at the event).
   - Each cell's inclusion and skipping junction reads as a small bar.
   - Point to make: most cells have 0 reads at the event, and covered cells have a few, mostly duplicated, reads.
3. **Pseudobulk.** Sum the reads over all TRAIN cells of each group to get raw ψ = junction reads / ATSE reads, with the read counts beside it.
4. **The model in one picture.**
   - Per cell: ρ = φ·ψ (programs, cell weights).
   - The per-cell posterior-mean usage ρ̃ = (γ_e ρ + y) / (γ_e + N).
     - A cell with no reads contributes its prior ρ.
     - A covered cell is pulled toward its reads.
   - The 12 example cells are shown again, as ρ and ρ̃.
5. **The cell-type estimate.** The group mean of ρ̃ (the catalog ψ), beside raw pseudobulk, with the check 12 interval where held-out cells have ≥ 100 reads and ≥ 10 cells with reads.
   - The interval is a 95 % predictive interval for held-out pseudobulk at its read depth, not for a true ψ.
   - A user's own fit gets ψ but no interval.
6. **What it buys and what it costs** (facts from the reports, no new numbers):
   - Gain over raw pseudobulk: largest where reads are few (check 10.1).
   - Compression at low coverage, and in the largest types raw is slightly better (check 10.1).
   - Programs themselves are not reproducible (checks 3, 8), so the tutorial does not interpret them.

**Code needed:**
- `leafletfa.catalog.celltype_psi` (exists).
- A per-cell helper returning ρ and ρ̃ for chosen cells (new, about 20 lines).
- The figure's `event_panel` style.
- Reads from the seed-0 Allen fit and the TRAIN counts.

**Cost:** CPU only. Reading the Allen TRAIN counts dominates (about 20–30 min, about 25 GB, sbatch). Drawing takes minutes. No GPU.

## 5. Descriptive section: "Which cell types differ most in splicing?" (plan only; rule to be fixed before any value is looked at)

**Not LeafletFA calls.** LeafletFA's cell-type-specificity calls failed their pre-registered checks (10.2 and 10.7: not confirmed above the null, not reproducible across seeds; check 9: differential splicing not reproducible). This section makes no calls and uses no LeafletFA test. It is a descriptive, power-matched summary of how often two cell types' splicing differs, shown as a heatmap of cell-type pairs. The text and the figure must say so.

**Why matching is needed:** a raw count of differences mostly measures power. Types with more cells and reads show more differences.

**Proposed rule (to be written into VALIDATION_PREREG.md as a dated record before anything is run):**
- **Units:**
  - **Allen:** all pairs of the 68 cell groups (2,278 pairs). Inhibitory groups are labeled with the PV / SST / VIP / Lamp5 lineages of `cell_type_maps/mouse_lineage_map.json`.
  - **TMS:** all pairs of the 132 tissue-cell types (8,646 pairs), shown within tissue and within lineage blocks.
- **Testable event for a pair:** an ATSE where **both** types have ≥ 200 TRAIN reads at the ATSE and ≥ 10 TRAIN cells with reads (above the browser's 20 / 3 so that depth matching below is possible).
- **Depth matching:** for each testable (pair, ATSE), thin both types' pseudobulk junction counts to the same depth D = 200 reads at the ATSE.
  - Thinning: multivariate hypergeometric draw of D reads from each type's TRAIN junction counts.
  - 20 independent draws, seeds 0–19.
- **Quantity:** Δψ at the ATSE = the largest |ψ_A − ψ_B| over its junctions, computed on the thinned raw pseudobulk.
  - The pair's score = the fraction of its testable ATSEs with |Δψ| ≥ 0.10, averaged over the 20 draws.
  - Also reported: the number of testable ATSEs per pair.
- **Noise floor per type:** the same score for a type against itself.
  - Its TRAIN cells are split into two halves by mouse (the 10.2a mouse halves), thinned to the same D.
  - The heatmap shows score(A, B) − ½ [floor(A) + floor(B)], with the raw score and the floor alongside.
  - Types without cells in both mouse halves get no floor and are marked.
- **Primary view:** raw pseudobulk (model-free).
  - **Secondary view (descriptive):** the same rule on LeafletFA ψ, recomputed from per-cell reads thinned to the same depth. Needs per-cell thinning, not pseudobulk thinning.
  - **Optional third view:** SpliceVI's read-supported calls (posterior sample + read support ≥ 3) for the Allen pairs it covers, from the SpliceVI paper. Labeled as SpliceVI, not LeafletFA.
- **Minimum:** pairs with < 50 testable ATSEs are shown gray (no score).
- **Readout:** descriptive only, no pass rule.
  - Heatmap of cell-type pairs, ordered by lineage.
  - The most and least different pairs listed with their testable-event counts.

**What could bias it (stated in the text and the figure legend):**
- **Cell numbers:** depth matching equalizes reads at the event, not the number of cells contributing them. A type of 20 cells at 200 reads is noisier than one of 2,000 cells at 200 reads, because duplicates dominate (Supp. Note 1). The noise floor partly corrects this; cells contributing reads will be reported per pair.
- **Depth and testability:** deep, large types have more testable events, so their scores rest on different (more weakly expressed) events than small types. Scores across pairs compare different event sets. Report testable counts and a version restricted to events testable in all types of a lineage block.
- **Event complexity:** the largest |Δψ| over junctions grows with the number of junctions in the ATSE. Report by event class (alt 3′, alt 5′, skipped exon, complex) as well as overall.
- **Cell-type resolution:**
  - TMS types are tissue × broad cell type.
  - Allen groups are cluster names (finer, and defined from expression clusters, so expression-related splicing differences are built into the labels).
  - Pairs within one atlas are comparable; scores across atlases are not.
- **Per-cell split and donors:**
  - TRAIN cells only, from mice shared with TEST, so there is no held-out confirmation of the differences.
  - In Allen, donors were sampled by Cre line, so cell-group and donor composition are confounded (check 10.2 null diagnostics). The within-type mouse-half floor inherits that.
- **Duplicates and γ floor:** reads are largely PCR duplicates. Thinning reads, not molecules, overstates effective depth equally for both types of a pair, but not equally across pairs.

**Cost:**
- **Raw pseudobulk view:** reads only the browser's TRAIN tables (type × junction counts, 17 MB / 11 MB) and, for the floor, per-mouse-half pseudobulks (one pass over the TRAIN counts per atlas, about 20–30 min).
  - Thinning and scoring all pairs × 20 draws is light: about 10⁷ (pair, ATSE) thinning draws for TMS. Under 1 CPU-h per atlas.
  - Total about 2 CPU-h with I/O, sbatch, no GPU.
- **LeafletFA view:** per-cell thinning plus recomputing the catalog estimator for every type at each draw. With the estimator port (`leafletfa.catalog.celltype_psi`, a few minutes per atlas per draw after loading), about 20 draws × 2 atlases ≈ 3–5 CPU-h plus loading. Memory as the port check (TMS peak under 24 GB, Allen under 32 GB).
- **SpliceVI view:** no compute here; read from the SpliceVI paper's call tables.
- **Writing the rule into the pre-registration and building the heatmap figure:** about half a day.
- **Timing:** after Saturday 14 Nov. Tutorial due before the 29 Nov tags; the 28 Nov LeafletFA day has the tutorial slot.

**Decisions for Karin before the rule is recorded:**
- D (200 reads) and the testable thresholds (≥ 200 reads, ≥ 10 cells).
- |Δψ| ≥ 0.10 (δ of check 10, the 99th percentile of within-type split-half TV) or TV-based.
- Whether the LeafletFA view is included or only the raw view.
- Whether the SpliceVI view is added.

Decisions for Karin:
- ~~Raw columns in the export; port of the estimator~~: approved and done 2026-11-07.
- Choose whether the legacy Beta-binomial class stays in the package or only in docs history.
