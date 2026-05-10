# SplicingDataset format

SplicingDataset is the shared h5ad format consumed by LeafletFA and SpliceVI. It is produced by [ATSEmapper](atsemapper.md) from per-cell junction BED files.

## Structure

A SplicingDataset is an [AnnData](https://anndata.readthedocs.io/) object where:

- **rows** (`obs`) = cells
- **columns** (`var`) = junctions (one column per junction)

Two sparse layers are required:

| Layer | Shape | dtype | Contents |
|-------|-------|-------|----------|
| `cell_by_junction_matrix` | cells × junctions | float32, CSR/COO | Read counts per junction per cell |
| `cell_by_cluster_matrix` | cells × junctions | float32, CSR/COO | Total ATSE counts per cell (beta-binomial denominator) |

CSR is the preferred sparse format. COO is also accepted.

## What is an ATSE?

An **Alternative Transcript Structure Event (ATSE)** is a group of mutually exclusive or competing junctions that together describe a single splicing choice (e.g. a cassette exon skip produces a skipping junction and two inclusion junctions).

Each junction in `adata.var` belongs to exactly one ATSE. Multiple columns of both matrices map to the same ATSE.

## Understanding `cell_by_cluster_matrix`

`cell_by_cluster_matrix[c, j]` is the **total reads across all junctions in the same ATSE as junction j**, for cell c. It is the denominator in the beta-binomial likelihood and ensures PSI estimates are compositionally consistent within each splicing event.

Key invariants:

- `junction[c,j] > 0` ⟹ `cluster[c,j] > 0` (can't have junction reads without an ATSE total)
- `junction[c,j] = 0` does **not** imply `cluster[c,j] = 0` (other junctions in the same ATSE may have reads)
- For all junctions j1, j2 in the same ATSE: `cluster[c, j1] == cluster[c, j2]`

```
ATSE 0 (junctions 0,1,2)              ATSE 1 (junctions 3,4)
          j0   j1   j2                       j3   j4
cell 0 [   3    0    1  | cluster=4 ] [   0    2  | cluster=2 ]
cell 1 [   0    2    0  | cluster=2 ] [   1    0  | cluster=1 ]
cell 2 [   1    1    0  | cluster=2 ] [   0    0  | cluster=0 ]
```

## Sparsity

Both matrices are typically **70–95 % zeros** for Smart-Seq2 data. This sparsity is expected and handled correctly by the model; do not impute or fill zeros before passing to LeafletFA.

## Constructing from scratch

If you are not using ATSEmapper and want to assemble the AnnData manually:

```python
import numpy as np
import scipy.sparse as sp
import anndata as ad

# 5 cells, 6 junctions grouped into 2 ATSEs (3 junctions each)
atse_ids = np.array([0, 0, 0, 1, 1, 1])  # which ATSE each junction belongs to

junc = np.array([
    [3, 0, 1, 0, 2, 0],
    [0, 2, 0, 1, 0, 3],
    [1, 1, 0, 0, 0, 2],
    [0, 0, 4, 2, 1, 0],
    [2, 0, 0, 0, 3, 1],
], dtype=np.float32)

# cluster[c, j] = sum of all junctions in the same ATSE as j, for cell c
cluster = np.zeros_like(junc)
for atse in np.unique(atse_ids):
    cols = atse_ids == atse
    cluster[:, cols] = junc[:, cols].sum(axis=1, keepdims=True)

adata = ad.AnnData(X=sp.csr_matrix(junc))
adata.layers["cell_by_junction_matrix"] = sp.csr_matrix(junc)
adata.layers["cell_by_cluster_matrix"] = sp.csr_matrix(cluster)
```

`adata.obs` (one row per cell) and `adata.var` (one row per junction) can hold any additional metadata. Cell annotations such as `cell_type` or `tissue` added to `adata.obs` are preserved through training and useful for interpreting learned splicing programs.

## Zenodo dataset

The mouse splicing atlas used in the paper is available at **[DOI: 10.5281/zenodo.18158125](https://doi.org/10.5281/zenodo.18158125)**:

| File | Description |
|------|-------------|
| `model_ready_aligned_splicing_data.h5ad` | Raw junction counts ready to pass to LeafletFA (~1 GB) |
| `splice_adata_for_figures_mouse_foundation.h5ad` | Trained atlas with fitted SPs and cell activities (~93 MB) |
