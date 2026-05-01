"""
Smoke tests for the leafletfa package.
These tests verify that the package can be imported and that the core model
runs end-to-end on a tiny synthetic dataset (no GPU required).
"""
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pytest
from pathlib import Path

TEST_H5AD = Path(__file__).parent / "data" / "test_splicing.h5ad"


def make_synthetic_adata(n_cells=30, n_junctions=60, seed=42, fmt="csr"):
    """Build a minimal synthetic AnnData with the two layers LeafletFA requires.

    Both layers must share the same sparsity pattern: if a cell has zero reads
    at a junction it also has zero reads at that ATSE (the real-data invariant).
    """
    rng = np.random.default_rng(seed)
    density = rng.poisson(lam=2, size=(n_cells, n_junctions)).astype(np.float32)
    density[rng.random((n_cells, n_junctions)) < 0.7] = 0
    # Cluster counts = junction counts + extra reads, but only where junction > 0
    extra = rng.poisson(lam=1, size=(n_cells, n_junctions)).astype(np.float32)
    cluster = density + np.where(density > 0, extra, 0)
    convert = {"csr": sp.csr_matrix, "coo": sp.coo_matrix}[fmt]
    Y = convert(density)
    T = convert(cluster)
    adata = ad.AnnData(X=sp.csr_matrix(density))
    adata.layers["cell_by_junction_matrix"] = Y
    adata.layers["cell_by_cluster_matrix"] = T
    return adata


# ---------------------------------------------------------------------------
# Import / version checks
# ---------------------------------------------------------------------------

def test_import():
    import leafletfa
    assert hasattr(leafletfa, "LeafletFA")
    assert leafletfa.__version__ == "0.1.0"


def test_submodule_imports():
    from leafletfa import waypoints, differential_splicing, estimate_bayesian_fdr
    assert callable(getattr(waypoints, "max_min_sampling", None))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_junction_layer():
    from leafletfa import LeafletFA
    adata = ad.AnnData(X=sp.eye(10, format="csr"))
    adata.layers["cell_by_cluster_matrix"] = sp.eye(10, format="csr")
    with pytest.raises(ValueError, match="cell_by_junction_matrix"):
        LeafletFA(adata, K=2, waypoints_use=False, log_wandb=False, loss_plot=False).from_anndata()


def test_missing_cluster_layer():
    from leafletfa import LeafletFA
    adata = ad.AnnData(X=sp.eye(10, format="csr"))
    adata.layers["cell_by_junction_matrix"] = sp.eye(10, format="csr")
    with pytest.raises(ValueError, match="cell_by_cluster_matrix"):
        LeafletFA(adata, K=2, waypoints_use=False, log_wandb=False, loss_plot=False).from_anndata()


def test_missing_both_layers():
    from leafletfa import LeafletFA
    adata = ad.AnnData(X=sp.eye(10, format="csr"))
    with pytest.raises(ValueError):
        LeafletFA(adata, K=2, waypoints_use=False, log_wandb=False, loss_plot=False).from_anndata()


# ---------------------------------------------------------------------------
# Sparse format compatibility
# ---------------------------------------------------------------------------

def test_csr_input():
    from leafletfa import LeafletFA
    adata = make_synthetic_adata(fmt="csr")
    model = LeafletFA(adata, K=2, num_epochs=3, waypoints_use=False, log_wandb=False, loss_plot=False)
    model.from_anndata()  # should not raise


def test_coo_input():
    from leafletfa import LeafletFA
    adata = make_synthetic_adata(fmt="coo")
    model = LeafletFA(adata, K=2, num_epochs=3, waypoints_use=False, log_wandb=False, loss_plot=False)
    model.from_anndata()  # should not raise


# ---------------------------------------------------------------------------
# End-to-end on synthetic data
# ---------------------------------------------------------------------------

def test_synthetic_run():
    from leafletfa import LeafletFA

    adata = make_synthetic_adata(n_cells=30, n_junctions=60)
    model = LeafletFA(adata, K=2, num_epochs=5, waypoints_use=False, log_wandb=False, loss_plot=False)
    model.from_anndata()
    model.train(num_initializations=1)
    model.get_all_variables()

    assert np.asarray(model.psi).shape == (2, 60)
    assert np.asarray(model.assign_post).shape == (30, 2)
    assert np.asarray(model.pi).shape == (2,)
    assert not np.any(np.isnan(np.asarray(model.psi)))


# ---------------------------------------------------------------------------
# End-to-end on real test fixture
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST_H5AD.exists(), reason="test fixture not found")
def test_real_data_run():
    from leafletfa import LeafletFA

    adata = ad.read_h5ad(TEST_H5AD)
    assert "cell_by_junction_matrix" in adata.layers
    assert "cell_by_cluster_matrix" in adata.layers

    # Subset to a tiny slice so the test stays fast
    idx = np.random.default_rng(0).choice(adata.n_obs, 200, replace=False)
    junc_counts = np.asarray(adata[idx].layers["cell_by_junction_matrix"].sum(axis=0)).flatten()
    top_j = np.argsort(junc_counts)[::-1][:500]
    adata_sub = adata[idx, top_j].copy()

    model = LeafletFA(adata_sub, K=3, num_epochs=5, waypoints_use=False, log_wandb=False, loss_plot=False)
    model.from_anndata()
    model.train(num_initializations=1)
    model.get_all_variables()

    assert np.asarray(model.psi).shape == (3, 500)
    assert np.asarray(model.assign_post).shape == (200, 3)
    assert "cell_ontology_class" in adata_sub.obs.columns
