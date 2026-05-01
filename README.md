# LeafletFA: Bayesian Factor Analysis for Single-Cell Splicing

LeafletFA is a scalable probabilistic Beta-Dirichlet factor model designed to decompose sparse single-cell splicing variation into interpretable, continuous Splicing Programs (SPs). It discovers coordinated modules of splicing events (Alternative Transcript Structure Events — ATSEs) that reflect biological states such as cellular aging or lineage specification, without requiring pre-defined cell type labels.

**Related repositories:**
- [ATSEmapper](https://github.com/daklab/ATSEmapper) — preprocessing pipeline: BAM files → regtools junction extraction → ATSE clustering → `SplicingDataset.h5ad`
- [Leaflet-analysis](https://github.com/daklab/Leaflet-analysis) — code and Snakemake pipelines used to produce all figures in the paper

**Full pipeline overview:**
```
BAM files (per cell)
     ↓  [ATSEmapper]  ← run this first if starting from raw data
     ↓  regtools junction extraction
     ↓  splice graph construction → ATSE clustering
SplicingDataset.h5ad
     ↓  [LeafletFA]  ← this repo
     ↓  Beta-Dirichlet factor model fitting
     ↓  Splicing Programs (K factors) + cell activities
Results
```

### Key Features
- **Scalable inference:** Powered by Pyro and Stochastic Variational Inference (SVI) for atlas-scale datasets (200,000+ cells)
- **Sparsity robust:** Specifically designed to handle the high dropout and sparse coverage inherent in single-cell splicing data
- **Biologically interpretable:** Learns a "splicing dictionary" where each factor represents a coordinated regulatory program
- **GPU-accelerated:** Triton-accelerated mini-batch training; CPU mode also supported

### Compatibility
LeafletFA is optimized for full-length transcript sequencing (e.g., Smart-Seq2) which provides the internal junction coverage necessary for alternative splicing analysis.

---

## Installation

```bash
git clone https://github.com/daklab/LeafletFA.git
cd LeafletFA
pip install -e .
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, and Pyro ≥ 1.9. GPU is optional but recommended for large datasets.

To enable Weights & Biases logging:
```bash
pip install -e ".[wandb]"
```

---

## Quick start

### Option A — Starting from raw BAM files

Run [ATSEmapper](https://github.com/daklab/ATSEmapper) first to extract junction counts and build the `SplicingDataset.h5ad` input file. ATSEmapper takes per-cell junction BED files produced by regtools and produces the format LeafletFA expects.

### Option B — Starting from an existing SplicingDataset

If you already have a `SplicingDataset.h5ad` (e.g. downloaded from Zenodo — see below), skip directly to model fitting:

```python
import leafletfa
import anndata as ad

adata = ad.read_h5ad("splicing_dataset.h5ad")

model = leafletfa.LeafletFA(adata, K=20)
model.from_anndata()   # convert to PyTorch tensors
model.train()          # variational inference
model.get_all_variables()

# Access results
model.psi          # (K × junctions) — splicing program loadings
model.assign_post  # (cells × K)     — cell factor activities
model.pi           # (K,)            — factor prevalences
```

See `notebooks/quickstart.ipynb` for a worked example with a real dataset.

---

## Data

### Mouse splicing atlas (Tabula Muris Senis + EasySci)

The dataset used in the paper is available on Zenodo: **[DOI: 10.5281/zenodo.18158125](https://doi.org/10.5281/zenodo.18158125)**

| File | Description | Size |
|------|-------------|------|
| `model_ready_aligned_splicing_data.h5ad` | SplicingDataset input — raw junction counts, ready to pass to LeafletFA | ~1 GB |
| `splice_adata_for_figures_mouse_foundation.h5ad` | Trained atlas object — fitted SPs, cell activities, all metadata | ~93 MB |

To download and load the trained atlas object:

```python
import urllib.request
import anndata as ad

url = "https://zenodo.org/records/18158125/files/splice_adata_for_figures_mouse_foundation.h5ad?download=1"
urllib.request.urlretrieve(url, "mouse_atlas.h5ad")

adata = ad.read_h5ad("mouse_atlas.h5ad")
# obs contains: cell_ontology_class, tissue, age, SP_1..SP_20, etc.
# varm contains: psi_learned  (K × junctions splicing program matrix)
# obsm contains: X_PHI        (cells × K activity matrix)
print(adata)
```

---

## Input format

LeafletFA requires an [AnnData](https://anndata.readthedocs.io/) object with two sparse layers, produced by [ATSEmapper](https://github.com/daklab/ATSEmapper):

| Layer | Shape | dtype | Contents |
|-------|-------|-------|----------|
| `cell_by_junction_matrix` | cells × junctions | float32, sparse CSR/COO | Read counts per junction per cell |
| `cell_by_cluster_matrix` | cells × junctions | float32, sparse CSR/COO | Total ATSE read counts per cell (denominator for beta-binomial) |

`cell_by_junction_matrix[c, j]` = reads from cell `c` supporting junction `j`.

`cell_by_cluster_matrix[c, j]` = total reads from cell `c` across **all junctions in the same ATSE** as junction `j`. This is the denominator in the beta-binomial likelihood — it ensures PSI estimates are compositionally consistent within each splicing event.

Each column `j` corresponds to one junction. Multiple columns map to the same ATSE (one per junction in that event). ATSEmapper handles this grouping automatically.

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 10 | Number of splicing programs (factors) |
| `num_epochs` | 500 | Training epochs per initialization |
| `lr` | 0.01 | Learning rate (ClippedAdam) |
| `waypoints_use` | True | Use PCA-based waypoint initialization (recommended — improves convergence) |
| `junc_specific_prior` | True | Per-junction Beta prior (vs. global) |
| `log_wandb` | False | Log metrics to Weights & Biases |

---

## Development roadmap
- [x] Implement Beta-Dirichlet factor model (LeafletFA)
- [x] Support for cross-species transfer learning
- [x] GPU-accelerated mini-batch training
- [x] pip-installable package
- [ ] ReadTheDocs documentation
- [ ] PyPI release
