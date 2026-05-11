# Ecosystem interop

LeafletFA stores all results in standard AnnData slots, so the output integrates directly with the broader single-cell Python ecosystem.

## scanpy

Cell activities in `adata.obsm["X_PHI"]` work as a drop-in replacement for any PCA/scVI embedding:

```python
import scanpy as sc

sc.pp.neighbors(adata, use_rep="X_PHI")
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color=["leiden", "tissue", "age"])
```

## scvi-tools

SpliceVI — the sister model — is built on scvi-tools and shares the same SplicingDataset input format. If you want a joint splicing + gene expression latent space, use SpliceVI; the `adata.obsm["X_PHI"]` slot name is compatible.

## muon

For multimodal objects combining splicing with chromatin or gene expression modalities, [muon](https://muon.readthedocs.io/) can hold the SplicingDataset as one modality:

```python
import muon as mu

mdata = mu.MuData({"splicing": adata_splicing, "rna": adata_rna})
```

## pyroe / tximeta

If starting from alevin-fry or salmon quantification, pyroe can produce per-cell count matrices that ATSEmapper can consume after junction extraction with regtools.
