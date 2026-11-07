# LeafletFA release manifest (Application Note, 2026-11-07)

Read-only inventory, no new analyses. Paths are on the NYGC cluster.
- `B` = `/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_splicevi_baseline`
- `V` = `/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation`
- `D` = `/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526`

## 1. Final model

**Configuration** (identical for every fit, from each fit's `run_record.json`):
- **Model:** Dirichlet-multinomial over ATSEs, within-ATSE simplex ψ, δ pinned to K (`scripts/splicevi_baseline/run_leafletfa_baseline.py`).
- **Training settings:**
  - K = 20.
  - 16,200 steps, minibatch 8,192, learning rate 0.025.
  - γ (gamma) = 0.1, concentration floor (`conc_min`) = 2.
  - φ in float64, sparse likelihood (`dense: false`).
- **E-step:** 3,000 steps, minibatch 8,192, learning rate 0.1.
- **Versions:** leafletfa 0.2.0, torch 2.11.0+cu128, pyro 1.9.1, numpy 1.26.4, scipy 1.14.1, Python 3.10.13.
- **Hardware:** GPU, NVIDIA RTX PRO 6000 Blackwell (b6k).

**Fits:**

| Fit | Role | Checkpoint dir | Commit (branch) | Train / test cells | Wall time |
|---|---|---|---|---|---|
| TMS seed 0 | **primary** (catalog, browser, check 10) | `B/TMS/full_K20_s0_f64/fit/` | 0f304d9 (leafletfa-splicevi-baseline), **dirty** | 54,694 / 23,441 | 10.6 min |
| TMS seeds 1, 2 | seed checks | `B/TMS/full_K20_s{1,2}_f64/fit/` | 7446210, clean | same | 23 min |
| TMS seeds 3–14 | check 8 consensus only | `V/c8_consensus/fits/full_K20_s{3..14}_f64/fit/` | not read here | same | — |
| Allen seed 0 | **primary** | `B/Allen/full_K20_s0_f64/fit/` | 7446210, clean | 38,199 / 16,372 | 31.6 min |
| Allen seed 1 | seed checks | `B/Allen/full_K20_s1_f64/fit/` | 7446210, clean | same | 69 min |
| Allen seed 2 | seed checks | `B/Allen/full_K20_s2_f64/fit/` | 92e3834 (leafletfa-validation), clean | same | 69 min |

Notes on the fits:
- The `leafletfa/` package directory is identical at 0f304d9, 7446210 and 92e3834 (`git diff --stat` on `leafletfa/` is empty). Those commits differ only in driver scripts.
- **TMS seed 0 ran on a dirty tree.** The uncommitted change is not recorded: UNVERIFIED. The next commit (7446210, the same day) is a driver fix ("move psi/phi to CPU before numpy").

**Primary checkpoint md5** (computed 2026-11-07):

| File | Size | md5 |
|---|---|---|
| TMS `fit.npz` | 11.0 MB | 140c17e6af58dd9466607e864a0fb983 |
| TMS `checkpoint.pt` | 29.7 MB | 2e54ba4c87d40d6d2c85201bd95dda03 |
| Allen `fit.npz` | 9.7 MB | 974153b331677c443933f67de77a9f9f (matches the browser download header) |
| Allen `checkpoint.pt` | 24.7 MB | 3f4e01267647510adc3285c4dd0b375c |

## 2. Final datasets

**Model inputs** (`D/`, built 2026-06-17):
- File names are `{train_70_30,test_30_70}_{TMS,AB}_ONLY_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu`.
- The md5 values below are from the fits' `run_record.json`.
- Re-hashed 2026-11-07 (job 22817442, `V/release/datasets.md5`): all four md5 values match the run records, so the inputs are unchanged since the fits.

| File | Size (bytes) | mtime | md5 | Cells |
|---|---|---|---|---|
| TMS train | 45,777,319,017 | 2026-08-02 | d2ceebe490b7bfe1f1487fe2aaad7424 | 54,694 |
| TMS test | 19,672,332,337 | 2026-08-02 | c9149ef41fdc4592d27f42b3d26c4c8e | 23,441 |
| Allen train | 50,586,874,701 | 2026-09-05 | 3f4918928c75ae4758620108d65a424f | 38,199 |
| Allen test | 21,731,891,201 | 2026-09-05 | f77ba7aa16c6f00e4f5a7104fdb97c84 | 16,372 |

**How the split was made:** per cell, 70/30, seed 42, stratified by broad cell type × age, not grouped by mouse (`numbers.tex`, `CATALOG_PROVENANCE.md`).

**Catalog counts** (`numbers.tex` ledger):

| Count | Mouse model-ready object | TMS | Allen |
|---|---|---|---|
| Cells | 132,706 | 78,135 | 54,571 |
| Junctions | 79,768 | 77,562 | 77,204 |
| ATSEs | 24,475 | 24,390 | 24,162 |
| Genes | 11,233 | 11,210 | 11,050 |
| Cell types (≥ 10 TRAIN cells) | — | 132 (5 ineligible for calls) | 68 cell groups (1 ineligible) |

- **Event classes** (`V/portal_browser/precomputed/catalog_record.json`): complex 7,056; alt3 5,810; skipped exon 5,708; alt5 4,867; linked MXE 1,034.

**Reference inputs** (md5 values in the same `catalog_record.json`):

| Input | Path | md5 |
|---|---|---|
| GENCODE vM19 GTF | `.../TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf` | e5eba8699d1e92885c9f3d5c8153fe8a |
| ATSE file | `.../ATSE_mapper/ATSE_files/MOUSE_SMARTSEQ2_ATSE_FILE_unanno_also_2026-05-12_03-43-52.txt.gz` | dbdc000f7626ecb46b2c8edc8514a06e |
| Allen metadata | `.../allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv` | listed in the record |

## 3. Zenodo deposit (proposal)

| Item | Source | Size | Contents | Status |
|---|---|---|---|---|
| Cell-type ψ catalog, TMS | `V/c10/export/catalog_TMS_s0.tsv.gz` | 166 MB | per junction × cell type: ψ, interval, ψ_rest, reads, cells, cells covering, TV, specific, call_eligible | exists |
| Cell-type ψ catalog, Allen | `V/c10/export/catalog_Allen_s0.tsv.gz` | 112 MB | same | exists |
| Raw TRAIN ψ, reads, cells | `V/portal_browser/precomputed/{TMS,Allen}_train_tables.npz` | 17 MB + 11 MB | junction reads, ATSE reads, covered cells, mice per type | exists as npz only |
| Estimate arrays | `V/c10/{TMS,Allen}_s0_estimates.npz` | 387 MB + 199 MB | point, intervals (bootstrap and parameter-only), rest, reads, cov, eligibility | exists |
| Event catalog | `V/portal_browser/precomputed/catalog_events.json`, `catalog_junctions.parquet` | 4.8 MB + 2.8 MB | ATSE → junctions, class, gene, coordinates | exists |
| Browser site (static) | `V/portal_browser/site/` | 541 MB | built 2026-11-07, version preview-2026-11-06, all headless checks pass | exists |
| Model checkpoints, seed 0 | `B/{TMS,Allen}/full_K20_s0_f64/fit/` | 76 MB total | `fit.npz` and `checkpoint.pt` | exists |
| Model inputs (h5mu) | `D/` | 138 GB | train and test for both atlases | **decision needed:** deposit, or point to the source atlases plus code |

Gaps before deposit (no new analyses were run):
- **No raw ψ column in the catalog TSVs.** They carry reads and cells but not raw TRAIN ψ. Raw ψ exists only in the browser's `train_tables.npz` and in its per-event downloads. Adding `raw_psi` needs an export step. Karin to approve.
- **Intervals:** the current TSV interval columns are the check 10 intervals (coverage 0.14–0.35, check 10.3). They must carry the "not calibrated" label, or be replaced.
  - Check 12 passed with (a)+(b): coverage 0.950–0.963 with τ² from the other mouse half.
  - Shipping (a)+(b) in the TSVs and the browser needs an export step plus Karin's sign-off; nothing has been exported yet.
- **Size:** the deposit without the h5mu inputs is about 1.3 GB.

## 4. Open gaps in the manuscript

From `LeafletFA_overleaf/main.tex`, pulled 2026-11-07:
- **Funding:** `\unsure{GAP (Karin): funding.}` (line 359).
- **Zenodo:**
  - `\unsure{[Zenodo DOI of the new deposit for this submission]}` (line 347).
  - "both repository URLs, one Zenodo DOI, portal URL" in the abstract's availability line (line 72).
  - Versions and tags (29 Nov) and the portal URL (line 340).
- **Also open:**
  - Subject section (line 49).
  - Author list and corresponding author (line 62).
  - Contact email (line 73).
  - Portal URL and description (line 273).
  - The `--preset paper` entry point (line 180).
- **Stale and should be updated:**
  - Line 281 says the scQuint and LeafCutter catalog benchmarks are "not run yet". Both have now run (scQuint job 22725257; LeafCutter TMS_full array 22780123).
  - Line 193 says amendment 7 is a "draft, not approved". It was approved and has run as check 12: (a)+(b) passes, (a) alone fails.
