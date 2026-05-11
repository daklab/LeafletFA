# Splicing Programs

A **Splicing Program (SP)** is one factor learned by LeafletFA. It is a vector of PSI values — one per junction — describing a coordinated pattern of splicing across the transcriptome. Cells express each SP to varying degrees (see [Cell Activities](cell-activities.md)).

## Accessing PSI loadings

After `model.get_all_variables()`:

```python
import numpy as np

# (K × junctions) — rows are programs, columns are junctions
psi = np.asarray(model.psi)

# Also written into adata:
# adata.varm["psi_learned"] is (junctions × K) — note transposed
psi_T = adata.varm["psi_learned"]
```

`psi[k, j]` is the expected proportion of reads supporting junction `j` under program `k`. Values are in (0, 1).

## Interpreting loadings

Junctions with extreme PSI values (near 0 or 1) across programs are the most informative. A junction with `psi[k, j] ≈ 0.9` and `psi[k', j] ≈ 0.1` for all other programs `k'` is a marker junction for SP `k`.

```python
# Top 20 marker junctions for SP 0
sp_idx = 0
top_junctions = np.argsort(np.abs(psi[sp_idx] - 0.5))[::-1][:20]
print(adata.var.iloc[top_junctions])
```

## Factor prevalences

`model.pi` (shape `K`) gives the mixing proportions — how prevalent each SP is across the dataset.

```python
import matplotlib.pyplot as plt
plt.bar(range(len(model.pi)), np.asarray(model.pi))
plt.xlabel("Splicing Program")
plt.ylabel("Prevalence (π)")
```
