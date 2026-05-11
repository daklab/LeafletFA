# Cell Activities

Each cell is assigned a continuous **activity score** for every Splicing Program. These scores live in `model.assign_post` (or `adata.obsm["X_PHI"]`) and can be used for visualization, clustering, and downstream analysis exactly like a standard dimensionality reduction.

## Accessing cell activities

```python
import numpy as np

# (cells × K) — rows are cells, columns are splicing programs
phi = np.asarray(model.assign_post)

# Also written into adata:
phi = adata.obsm["X_PHI"]   # (cells × K)
```

## UMAP / visualization

The `X_PHI` slot integrates directly with scanpy:

```python
import scanpy as sc

adata.obsm["X_PHI"] = phi

sc.pp.neighbors(adata, use_rep="X_PHI")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["cell_ontology_class", "tissue"])
```

## Plotting activity per cell type

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(phi, columns=[f"SP_{k+1}" for k in range(phi.shape[1])])
df["cell_type"] = adata.obs["cell_ontology_class"].values

df_long = df.melt(id_vars="cell_type", var_name="SP", value_name="activity")
sns.boxplot(data=df_long, x="SP", y="activity", hue="cell_type")
plt.xticks(rotation=45)
```

## Relationship to PSI loadings

A high activity score for SP `k` in a cell means that cell's splicing landscape is well explained by the PSI vector `psi[k, :]`. It does not mean the cell exclusively uses SP `k` — activities are soft assignments summing to approximately 1 across programs.
