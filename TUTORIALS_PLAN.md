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
- **B. From a fit (for a user's own data).** Steps:
  1. Fit with the production driver.
  2. Call `celltype_psi(model, adata, groupby="cell_type")`, which computes the same estimator from the fit.
  3. Run the same `celltype_table` and `plot_event` as in A.

**Code needed** (port, not new methods; each function is checked against the saved check 10 output):
- `leafletfa/catalog.py`:
  - `load_catalog`, `gene_events`, `celltype_table`.
  - `celltype_psi`: the estimator from `c10_catalog.py` without hard-coded paths. Point estimate first; intervals as an option.
- `leafletfa/plotting.py`: `plot_event`, matching the browser's encoding and `FIGURE_THEME.md`.
- **Test:** `celltype_psi` on the TMS seed-0 fit reproduces `c10/TMS_s0_estimates.npz` `point` (max difference < 1e-4), and `celltype_table` matches the browser's per-event TSV download for one ATSE.
- **Entry point A blocker resolved (2026-11-07):** the catalog TSVs now carry `raw_psi`, `junction_reads`, `mice`, the check 12 interval, `interval_status`, `test_reads` and `test_cells`.
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

Decisions for Karin:
- ~~Raw columns in the export; port of the estimator~~: approved and done 2026-11-07.
- Choose whether the legacy Beta-binomial class stays in the package or only in docs history.
