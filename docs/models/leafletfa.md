# LeafletFA model

LeafletFA fits a Beta-Dirichlet factor model to single-cell splicing data. It learns K **Splicing Programs** — each a vector of per-junction PSI values — and assigns each cell a continuous activity score per program.

## Quickstart

```python
import anndata as ad
from leafletfa import LeafletFA

adata = ad.read_h5ad("splicing_dataset.h5ad")

model = LeafletFA(adata, K=20)
model.from_anndata()   # validate input, build tensors
model.train()          # SVI with multiple random initializations
model.get_all_variables()

# Results
model.psi          # ndarray (K × junctions) — splicing program PSI loadings
model.assign_post  # ndarray (cells × K)     — cell factor activities
model.pi           # ndarray (K,)            — factor prevalences
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 10 | Number of splicing programs |
| `num_epochs` | 500 | Training epochs per initialization |
| `lr` | 0.01 | Learning rate (ClippedAdam) |
| `waypoints_use` | `True` | PCA-based waypoint initialization — improves convergence, recommended |
| `junc_specific_prior` | `True` | Learn a per-junction Beta prior instead of a global one |
| `num_initializations` | 3 | Random restarts; best ELBO is kept |
| `patience` | 5 | Early stopping patience (epochs without `min_delta` improvement) |
| `log_wandb` | `False` | Log training metrics to Weights & Biases |
| `loss_plot` | `True` | Plot ELBO curve after training |

## Accessing results

After `get_all_variables()`, results are stored on the model object and also written back into `adata`:

| Attribute | `adata` location | Shape | Description |
|-----------|-----------------|-------|-------------|
| `model.psi` | `adata.varm["psi_learned"]` | K × junctions | Per-program PSI loadings |
| `model.assign_post` | `adata.obsm["X_PHI"]` | cells × K | Cell factor activities |
| `model.pi` | — | K | Factor prevalences |

## Saving and loading

```python
# save the trained adata (includes fitted results)
adata.write_h5ad("results.h5ad")

# reload later
import anndata as ad
adata = ad.read_h5ad("results.h5ad")
psi = adata.varm["psi_learned"]    # (junctions × K) — note transposed vs model.psi
phi = adata.obsm["X_PHI"]          # (cells × K)
```
