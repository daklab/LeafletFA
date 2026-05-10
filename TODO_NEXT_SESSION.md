# Remaining Work — LeafletFA Reproducibility

## LeafletFA repo (this repo) — mostly done

- [ ] **Add AnnData format tests to `tests/test_basic.py`**
  - Test that `from_anndata()` raises a clear error when required layers are missing (not a cryptic KeyError)
  - Test that it handles both CSR and COO sparse formats
  - Test that it rejects dense matrices gracefully with a helpful message
  - Example:
    ```python
    def test_missing_layer_error():
        adata = ad.AnnData(X=sp.eye(10))  # no layers
        with pytest.raises(KeyError, match="cell_by_junction_matrix"):
            LeafletFA(adata, K=2).from_anndata()
    ```

- [ ] **Add AnnData format description to README and docs**
  - Exact layer names, dtypes, sparsity expectations
  - What `cell_by_cluster_matrix` means conceptually (ATSE total counts = denominator for beta-binomial)
  - Minimal working example showing how to construct both layers from scratch
  - A diagram of how one ATSE maps to multiple junctions in the matrix

- [x] **Run quickstart notebook end-to-end** — works; `cell_ontology_class` is the column name

- [x] **Pin deps and promote scanpy/umap-learn to core** — done in `pyproject.toml`; fresh install verified

- [ ] **Fix COO sparse FutureWarning** (`anndata` 0.12 will reject COO matrices in layers)
  - `leafletfa/model.py:202-208` converts input to COO and stores back into `adata.layers`
  - Fix: convert to COO locally as a variable, don't store back into adata
  - Will break in anndata 0.12 if not fixed

- [ ] **Update analysis scripts** in Leaflet-analysis to import from `leafletfa` instead of the old HPC path pattern:
  ```python
  # old
  sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/")
  import BetaDirichletFactor.LeafletFA as LeafletFA
  # new
  from leafletfa import LeafletFA
  ```

---

## leafletfa-paper repo — not started
> **Repo rename:** `Leaflet-analysis` → `leafletfa-paper` (github.com/daklab/leafletfa-paper). Do this in GitHub Settings.

> **Purpose:** Frozen reproducibility record for the paper. NOT a user-facing tool.

- [ ] **Rename repo on GitHub:** Settings → Rename → `leafletfa-paper`

- [ ] **Add a figure index to the README** — a table mapping every paper figure to the exact script/notebook that generates it:
  | Figure | Script/Notebook | Input data |
  |--------|----------------|------------|
  | Fig 1E | `Mouse_Splicing_Foundation/figures/fig1e_atse_summary.ipynb` | `tms_splicing.h5ad` |
  | Fig 3D | `Mouse_Splicing_Foundation/figures/fig3d_sp_heatmap.ipynb` | `tms_leafletfa_results.h5ad` |
  | Fig 4A | `Multi_Species/figures/fig4a_conservation.ipynb` | both h5ads |
  | ... | ... | ... |

- [ ] **Update analysis script imports** — replace HPC `sys.path` hacks with `from leafletfa import LeafletFA` now that the package is pip-installable

---

## Phase 4: MkDocs Material documentation site

**Nav structure** (reflects three-tier architecture: data layer → model layer → outputs):

```
index.md                    ← ecosystem overview + architecture diagram
data-layer/
  splicingdataset.md        ← the h5ad format spec (layers, dtypes, what ATSE means)
  atsemapper.md             ← BAMs → SplicingDataset; link to ATSEmapper repo docs
models/
  leafletfa.md
  splicevi.md               ← stub; shows it consumes same SplicingDataset input
outputs/
  splicing-programs.md
  cell-activities.md
  differential.md
ecosystem.md                ← scanpy / scvi-tools / muon / pyroe interop
api/
  model.md
  waypoints.md
  differential_splicing.md
```

- [ ] **Set up MkDocs** with Material theme in LeafletFA repo
  - `pip install mkdocs-material mkdocstrings[python]`

- [ ] **Write `data-layer/splicingdataset.md` first** — anchors everything else:
  - Exact layer names, dtypes, sparsity expectations
  - What `cell_by_cluster_matrix` means (ATSE total counts = denominator for beta-binomial)
  - Minimal working example constructing both layers from scratch

- [ ] **Index page landing paragraph** — lead with ATSEmapper as the bridge:
  > ATSEmapper takes per-cell junction BED files from regtools — a bulk-RNA tool — and produces a SplicingDataset. It is the bridge between the bulk-sequencing infrastructure most labs already run and the single-cell-native format both LeafletFA and SpliceVI consume.

- [ ] **Pipeline diagram** (Mermaid, supported natively in Material):
  ```mermaid
  flowchart TD
    A[BAM files per cell] --> B[regtools junction extract]
    B --> C[ATSEmapper]
    C --> D[SplicingDataset.h5ad\ncell_by_junction_matrix\ncell_by_cluster_matrix]
    D --> E[LeafletFA]
    D --> F[SpliceVI]
    E --> G[Splicing programs\nCell SP activities]
    F --> H[Joint latent space\nDifferential analysis]
  ```

- [ ] **Auto-generate API reference** from docstrings using `mkdocstrings`
  - Will need docstrings on `LeafletFA.__init__`, `from_anndata`, `train`, `get_all_variables`

---

## Priority order for next session

1. ~~Rename repos on GitHub~~ — done (`ATSEmapper` renamed; `Leaflet-analysis` needs org owner)
2. ~~Run LeafletFA quickstart notebook end-to-end~~ — done
3. ~~Pin deps / fresh install test~~ — done
4. ~~Fix COO sparse FutureWarning / cluster count indexing bug~~ — done
5. ~~LeafletFA: AnnData format tests in `tests/test_basic.py`~~ — done
6. ~~LeafletFA: AnnData format description in README~~ — done
7. **MkDocs: scaffold site, write `data-layer/splicingdataset.md` first** ← start here
8. leafletfa-paper: rename repo + figure index in README
