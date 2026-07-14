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


def simulate_from_generative_model(n_cells=300, K=3, n_atses=30, juncs_per_atse=3, seed=0):
    """Simulate junction counts from LeafletFA's own generative model, with known psi/phi.

    phi ~ Dirichlet, psi ~ Beta, y ~ Binomial(T, phi @ psi). Returns (adata, psi_true, phi_true)
    so tests can check the model actually recovers the structure it claims to model.
    """
    rng = np.random.default_rng(seed)
    n_junctions = n_atses * juncs_per_atse
    atse_of = np.repeat(np.arange(n_atses), juncs_per_atse)

    psi_true = rng.beta(0.5, 0.5, size=(K, n_junctions))
    phi_true = rng.dirichlet(np.ones(K) * 0.4, size=n_cells)
    rho = phi_true @ psi_true

    atse_totals = rng.poisson(5, size=(n_cells, n_atses))
    cluster = atse_totals[:, atse_of].astype(np.float32)
    junc = rng.binomial(cluster.astype(int), np.clip(rho, 0, 1)).astype(np.float32)

    adata = ad.AnnData(X=sp.csr_matrix(junc))
    adata.layers["cell_by_junction_matrix"] = sp.csr_matrix(junc)
    adata.layers["cell_by_cluster_matrix"] = sp.csr_matrix(cluster)
    return adata, psi_true, phi_true


def best_match_corr(psi_hat, psi_true):
    """Mean correlation of each learned factor with its best-matching true factor.

    Factors are identifiable only up to permutation, so match before scoring.
    """
    K = psi_hat.shape[0]
    C = np.corrcoef(np.vstack([psi_hat, psi_true]))[:K, K:]
    return float(np.nanmax(C, axis=1).mean())


def make_synthetic_adata(n_cells=30, n_junctions=60, n_atses=15, seed=42, fmt="csr"):
    """Build a minimal synthetic AnnData with the two layers LeafletFA requires.

    Invariants (matching real ATSEmapper output):
    - junction[c,j] > 0  =>  cluster[c,j] > 0  (can't have junction reads without ATSE total)
    - junction[c,j] = 0 does NOT imply cluster[c,j] = 0  (other junctions in same ATSE may have reads)
    - sum of junction[c, all j in ATSE k] == cluster[c, j] for all j in ATSE k
    """
    rng = np.random.default_rng(seed)

    # Assign junctions to ATSEs (multiple junctions per ATSE)
    atse_ids = np.arange(n_junctions) % n_atses  # junction j belongs to ATSE atse_ids[j]

    # Generate per-junction counts with ~70% dropout
    junc_dense = rng.poisson(lam=2, size=(n_cells, n_junctions)).astype(np.float32)
    junc_dense[rng.random((n_cells, n_junctions)) < 0.7] = 0

    # Cluster count for each (cell, junction) = sum of all junctions in the same ATSE
    cluster_dense = np.zeros_like(junc_dense)
    for atse in range(n_atses):
        mask = atse_ids == atse
        atse_total = junc_dense[:, mask].sum(axis=1, keepdims=True)  # (n_cells, 1)
        cluster_dense[:, mask] = atse_total  # broadcast to all junctions in ATSE

    convert = {"csr": sp.csr_matrix, "coo": sp.coo_matrix}[fmt]
    Y = convert(junc_dense)
    T = convert(cluster_dense)
    adata = ad.AnnData(X=sp.csr_matrix(junc_dense))
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
# Gradient flow — the model must actually learn psi and assign
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dense", [False, True])
def test_likelihood_gradient_reaches_psi_and_assign(dense):
    """The likelihood must produce a non-zero gradient on the psi and assign guide params.

    Regression test for a bug where `sparse_dot_cpu` ran its index_select inside
    `torch.no_grad()` and then called `pred.requires_grad_(True)`. That detached psi/assign from
    the likelihood, so they were trained by the prior/entropy terms alone and the model returned
    (a shrunk version of) its initialization. Shape-only assertions cannot catch this.
    """
    import torch, pyro
    from pyro.infer import Trace_ELBO
    from pyro.infer.autoguide import AutoNormal
    from leafletfa import LeafletFA

    adata, _, _ = simulate_from_generative_model(n_cells=80, K=3, n_atses=15)
    model = LeafletFA(adata, K=3, waypoints_use=False, log_wandb=False, loss_plot=False,
                      use_dense_mode=dense)
    model.from_anndata()

    pyro.clear_param_store()
    pyro.set_rng_seed(0)
    guide = AutoNormal(model.model)
    guide()  # materialize the variational params
    # loss_and_grads runs backward but, unlike SVI.step, does not zero the grads afterwards
    Trace_ELBO().loss_and_grads(model.model, guide)

    store = pyro.get_param_store()
    for name in ("AutoNormal.locs.psi", "AutoNormal.locs.assign"):
        grad = store[name].grad
        assert grad is not None, f"{name} received no gradient at all"
        assert torch.isfinite(grad).all(), f"{name} has non-finite gradient"
        assert grad.abs().sum() > 0, (
            f"{name} has an all-zero gradient — the likelihood is detached from it"
        )


def test_psi_moves_away_from_its_initialization():
    """Training must actually move psi, driven by the data.

    The end-to-end counterpart to the gradient test. When the likelihood was detached, psi only
    relaxed toward the prior, so the fitted model was essentially its own initialization — and
    every shape/NaN assertion still passed.

    Deliberately NOT asserting recovery of the true psi: with random init that is a lottery
    (best-match correlation swings between ~0.12 and ~0.56 across settings) because the objective
    is non-convex and the Dirichlet prior on pi kills under-used factors. That is the very problem
    waypoint initialization exists to solve, not something a unit test should pretend is stable.
    """
    import torch
    from leafletfa import LeafletFA

    adata, _, _ = simulate_from_generative_model(n_cells=300, K=3, n_atses=30, seed=0)
    psi_start = torch.full((3, adata.shape[1]), 0.5)

    model = LeafletFA(adata, K=3, num_epochs=200, lr=0.05, patience=10**6, waypoints_use=False,
                      log_wandb=False, loss_plot=False, use_dense_mode=False)
    model.from_anndata()
    model.train(num_initializations=1, seeds=[0], psi_init=psi_start)
    model.get_all_variables()

    psi_end = np.asarray(model.psi)
    moved = np.abs(psi_end - psi_start.numpy()).mean()
    assert moved > 0.05, (
        f"psi barely left its initialization (mean |delta| = {moved:.4f}) — the likelihood is "
        f"probably not reaching it"
    )
    # and the factors must differentiate from each other, not collapse to one program
    assert psi_end.std(axis=0).mean() > 0.01, "all factors collapsed to the same psi"


# ---------------------------------------------------------------------------
# obs_scale — the N/batch rescaling used when training on subsamples
# ---------------------------------------------------------------------------

def _site_scales(callable_, scale, sites):
    """Effective ELBO weight (msg['scale']) of each site under _ScaleSites."""
    import pyro
    from pyro import poutine
    from leafletfa.model import _ScaleSites

    wrapped = _ScaleSites(scale, sites)(callable_) if scale != 1.0 else callable_
    pyro.set_rng_seed(0)
    tr = poutine.trace(wrapped).get_trace()
    return {n: float(node["scale"]) for n, node in tr.nodes.items()
            if node["type"] in ("sample", "factor")}


def test_obs_scale_scales_local_terms_on_BOTH_model_and_guide():
    """obs_scale must weight the LOCAL terms identically in the model AND in the guide.

    Regression test for a real bug: obs_scale was originally applied with a `poutine.scale` context
    *inside* model(). That cannot reach the guide -- the guide is a separate callable -- so
    log p(assign) and log p(y) were scaled by N/B while log q(assign) stayed at weight 1. The
    entropy of q(assign) is what stops the per-cell posteriors collapsing, so under-weighting it by
    N/B drives them toward point masses. A model-trace-only test passes while this is broken, which
    is exactly what happened, so this test checks the GUIDE side too.

    Globals (psi, pi, a, b) must stay at weight 1 on both sides.
    """
    import torch, pyro
    from pyro.infer.autoguide import AutoNormal
    from leafletfa import LeafletFA

    adata, _, _ = simulate_from_generative_model(n_cells=60, K=2, n_atses=10)
    S = 4.0

    m = LeafletFA(adata, K=2, waypoints_use=False, log_wandb=False, loss_plot=False,
                  obs_scale=S, use_dense_mode=False)
    m.from_anndata()
    pyro.clear_param_store()

    # --- model side: "assign" (local prior) and "obs" (likelihood) scale; globals do not
    model_scales = _site_scales(m.model, S, {"assign", "obs"})
    for local in ("obs", "assign"):
        assert model_scales[local] == pytest.approx(S), (
            f"model: local term '{local}' should have scale {S}, got {model_scales[local]}"
        )
    for glob in ("psi", "pi", "a", "b"):
        assert model_scales[glob] == pytest.approx(1.0), (
            f"model: global term '{glob}' must stay at weight 1, got {model_scales[glob]}"
        )

    # --- guide side: q("assign") must carry the SAME scale, or its entropy is under-weighted
    guide = AutoNormal(m.model)
    guide()  # materialize params
    guide_scales = _site_scales(guide, S, {"assign"})
    assert guide_scales["assign"] == pytest.approx(S), (
        f"guide: q(assign) has scale {guide_scales['assign']}, expected {S}. The entropy of "
        f"q(assign) is under-weighted by N/batch and the per-cell posteriors will collapse."
    )
    for glob in ("psi", "pi", "a", "b"):
        assert guide_scales[glob] == pytest.approx(1.0), (
            f"guide: global term '{glob}' must stay at weight 1, got {guide_scales[glob]}"
        )


# ---------------------------------------------------------------------------
# Transfer: projecting new cells onto a trained model
# ---------------------------------------------------------------------------

def test_pi_fixed_is_actually_fixed():
    """pi_fixed must freeze pi, not merely initialize it.

    pi is the prior over phi, so re-learning it on a target dataset puts the inferred phi on a
    different footing than the source phi and breaks cross-dataset comparison. `pi_init` only
    seeds a still-learned pi; `pi_fixed` is the one to use when projecting.
    """
    import torch
    from leafletfa import LeafletFA

    adata, _, _ = simulate_from_generative_model(n_cells=100, K=3, n_atses=15)
    psi_src = torch.rand(3, adata.shape[1], dtype=torch.float32).clamp(0.05, 0.95)
    pi_src = torch.tensor([0.6, 0.3, 0.1])

    model = LeafletFA(adata, K=3, num_epochs=10, waypoints_use=False, log_wandb=False,
                      loss_plot=False, fixed_psi=psi_src, pi_fixed=pi_src,
                      alpha_pi_fixed=torch.tensor(1.0), delta_fixed=1.0,
                      input_conc_prior=torch.tensor(float("inf")))
    model.from_anndata()
    model.train(num_initializations=1, seeds=[0])
    model.get_all_variables()

    np.testing.assert_allclose(np.asarray(model.pi), pi_src.numpy(), rtol=1e-6)
    # phi is still inferred, and is a valid simplex
    phi = np.asarray(model.assign_post)
    assert phi.shape == (100, 3)
    np.testing.assert_allclose(phi.sum(axis=1), 1.0, atol=1e-4)


def test_fixed_psi_junction_axis_mismatch_raises():
    """A fixed_psi whose junction axis does not match the target must fail loudly, not silently."""
    import torch
    from leafletfa import LeafletFA

    adata, _, _ = simulate_from_generative_model(n_cells=50, K=3, n_atses=10)
    wrong = torch.rand(3, adata.shape[1] - 5)  # junctions do not line up

    with pytest.raises(ValueError, match="junction axis"):
        LeafletFA(adata, K=3, fixed_psi=wrong, waypoints_use=False,
                  log_wandb=False, loss_plot=False)


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


# ---------------------------------------------------------------------------
# Recovery vs ground truth — the gate that catches symmetric collapse
# ---------------------------------------------------------------------------

def _hungarian_match(psi_hat, psi_true):
    """Programs are identifiable only up to permutation, so match before scoring."""
    from scipy.optimize import linear_sum_assignment
    K = psi_hat.shape[0]
    C = np.array([[np.corrcoef(psi_hat[i], psi_true[j])[0, 1] for j in range(K)] for i in range(K)])
    r, c = linear_sum_assignment(-np.abs(np.nan_to_num(C)))
    return r, c, np.array([C[i, j] for i, j in zip(r, c)])


def test_recovers_known_programs():
    """Fit data generated by the model itself and check psi/phi are actually recovered.

    This is the gate. A model stuck on the symmetric saddle scores ~0 here while every other
    diagnostic ("0 dead programs", "0 never dominant", a smooth loss curve) reports it healthy.
    That is exactly how the K=5 heart simulation was passing while recovering NOTHING
    (psi-vs-truth r = 0.008, dominant-program agreement BELOW chance).

    Schedule matters more than epochs: at lr=0.001 the programs never separate no matter how long
    you run. See METHODS.md sec 11.2.
    """
    from leafletfa import LeafletFA

    adata, psi_true, phi_true = simulate_from_generative_model(
        n_cells=600, K=4, n_atses=40, seed=0)

    model = LeafletFA(adata, K=4, waypoints_use=False, log_wandb=False, loss_plot=False,
                      num_epochs=800, patience=10**6, lr=0.01, gamma=0.5,
                      ELBO_num_particles=3, num_samples=30, use_dense_mode=False)
    model.from_anndata()
    model.train(num_initializations=1, seeds=[0])
    model.get_all_variables()

    psi_hat = np.asarray(model.psi)
    r, c, matched = _hungarian_match(psi_hat, psi_true)

    assert model.health["healthy"], f"health check failed: {model.health['problems']}"
    assert np.median(matched) > 0.4, (
        f"psi barely resembles the truth (median matched r = {np.median(matched):.3f}). "
        f"psi_contrast = {model.health['psi_contrast']:.4f} — near zero means the K programs "
        f"collapsed onto each other."
    )


def test_health_check_catches_symmetric_collapse():
    """health_check must FAIL a model trained with the schedule that cannot escape the saddle.

    lr=0.001 + gamma=0.01 is what the atlas pipeline shipped, and it recovers nothing. The point of
    health_check is that this failure is otherwise invisible.
    """
    from leafletfa import LeafletFA

    adata, _, _ = simulate_from_generative_model(n_cells=400, K=4, n_atses=30, seed=0)
    model = LeafletFA(adata, K=4, waypoints_use=False, log_wandb=False, loss_plot=False,
                      num_epochs=120, patience=10**6, lr=1e-4, gamma=0.01,
                      ELBO_num_particles=1, num_samples=10, use_dense_mode=False)
    model.from_anndata()
    model.train(num_initializations=1, seeds=[0])
    model.get_all_variables()

    assert not model.health["healthy"], (
        "health_check passed a model trained at lr=1e-4 that cannot have escaped the symmetric "
        "saddle — the check is not doing its job"
    )
    assert model.health["psi_contrast"] < 0.02


# ---------------------------------------------------------------------------
# Bayesian FDR — the max-detections rule
# ---------------------------------------------------------------------------

def test_bayesian_fdr_is_max_detections_and_fdp_is_monotone():
    """The threshold is the MOST PERMISSIVE tau with FDP <= alpha (== max detections).

    NB: FDP(n) is a running mean of (1 - P) over DESCENDING P, i.e. a running mean of an ASCENDING
    sequence, so it is monotone non-decreasing in n *by construction*. That means
    `len(where(fdp <= alpha))` and `argmax + 1` always coincide -- an earlier review claimed they
    could differ, and that was wrong. This test pins the monotonicity down so nobody "fixes" it again.

    What the Methods gets wrong is the direction: it says "the LARGEST tau with FDP(tau) <= alpha".
    FDP is non-increasing in tau, so the largest such tau is trivially 1.0 and selects nothing. The
    rule is the SMALLEST tau (== most detections).
    """
    from leafletfa.differential_splicing_groups import bayesian_fdr

    rng = np.random.default_rng(0)
    for _ in range(200):
        probs = rng.random(rng.integers(5, 80))
        sig, fdp, n_sig = bayesian_fdr(probs, alpha=0.05)
        ps = np.sort(probs)[::-1]
        curve = np.cumsum(1 - ps) / np.arange(1, len(ps) + 1)
        assert np.all(np.diff(curve) >= -1e-12), "FDP must be monotone non-decreasing in rank"
        # max detections: the n we keep is the LAST rank whose FDP is still within alpha
        ok = np.flatnonzero(curve <= 0.05)
        assert n_sig == (int(ok.max()) + 1 if ok.size else 0)
        assert sig.sum() == n_sig

    # pure null: every P = 0 -> nothing called
    sig0, _, n0 = bayesian_fdr(np.zeros(100), alpha=0.05)
    assert n0 == 0 and not sig0.any()
    # perfect signal: every P = 1 -> everything called, FDP 0
    sig1, fdp1, n1 = bayesian_fdr(np.ones(50), alpha=0.05)
    assert n1 == 50 and np.allclose(fdp1, 0.0)


def test_atse_total_variation_reduces_to_delta_psi_for_two_junctions():
    """For a 2-junction ATSE, TV must equal |DeltaPSI| — the ATSE and junction tests must agree."""
    import torch, anndata as ad, scipy.sparse as sp
    from leafletfa.differential_splicing_groups import group_delta_psi

    S, K, C = 20, 3, 40
    rng = np.random.default_rng(0)
    psi_s = torch.tensor(rng.random((S, K, 2)), dtype=torch.float32)
    phi_s = torch.tensor(rng.dirichlet(np.ones(K), size=(S, C)), dtype=torch.float32)
    c1, c2 = np.arange(0, 20), np.arange(20, 40)

    d = group_delta_psi(psi_s, phi_s, c1, c2)          # [S, 2]
    tv = 0.5 * d.abs().sum(dim=1)                      # TV over the 2-junction event

    # if the two junctions were a true composition (psi_2 = 1 - psi_1) then d[:,1] = -d[:,0]
    psi_comp = psi_s.clone()
    psi_comp[:, :, 1] = 1.0 - psi_comp[:, :, 0]
    d2 = group_delta_psi(psi_comp, phi_s, c1, c2)
    tv2 = 0.5 * d2.abs().sum(dim=1)
    np.testing.assert_allclose(tv2.numpy(), d2[:, 0].abs().numpy(), rtol=1e-3, atol=1e-6,
                               err_msg="TV must reduce to |DeltaPSI| for a 2-junction composition")


# ---------------------------------------------------------------------------
# Chunked (multi-pass minibatch) training must agree with full-batch
# ---------------------------------------------------------------------------

def _fit(adata, seed=0, obs_scale=1.0, psi_init=None, epochs=250, K=3, **kw):
    """One LeafletFA fit, quietly."""
    import io, contextlib, torch
    from leafletfa import LeafletFA
    m = LeafletFA(adata, K=K, waypoints_use=False, log_wandb=False, loss_plot=False,
                  num_epochs=epochs, patience=10**6, lr=0.05, gamma=0.5,
                  obs_scale=obs_scale, use_dense_mode=False, num_samples=20, **kw)
    with contextlib.redirect_stdout(io.StringIO()):
        m.from_anndata()
        m.train(num_initializations=1, seeds=[seed],
                psi_init=None if psi_init is None else torch.tensor(psi_init))
        m.get_all_variables()
    return m


def _npy(x):
    import torch
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def test_chunked_training_recovers_the_same_psi_as_full_batch():
    """The multi-pass chunked scheme must learn the SAME psi as a single full-batch fit.

    The atlas cannot be trained full-batch (13.8% dense, 1.46e9 nonzeros -> ~305 GB), so training is
    chunked: each chunk builds a fresh LeafletFA, warm-started from the previous chunk's psi, with
    obs_scale = N/batch so the objective stays an unbiased estimate of the FULL-data ELBO.

    That scheme is only trustworthy if it lands where full-batch lands. This test pins that down on a
    small simulation where full-batch IS feasible. It is the guard for the whole training pipeline.
    """
    adata, psi_true, _ = simulate_from_generative_model(n_cells=900, K=3, n_atses=30, seed=0)
    N, B = 900, 300

    full = _fit(adata, epochs=400)
    psi_full = _npy(full.psi)

    rng = np.random.default_rng(0)
    g_psi = None
    for _pass in range(3):
        perm = rng.permutation(N)
        for i in range(0, N, B):
            idx = np.sort(perm[i:i + B])
            m = _fit(adata[idx].copy(), obs_scale=N / len(idx), psi_init=g_psi, epochs=150)
            g_psi = _npy(m.psi)

    # both must recover the truth, and must agree with EACH OTHER (up to factor permutation)
    r_full = np.median(_hungarian_match(psi_full, psi_true)[2])
    r_chunk = np.median(_hungarian_match(g_psi, psi_true)[2])
    r_agree = np.median(_hungarian_match(g_psi, psi_full)[2])

    assert r_full > 0.5, f"full-batch failed to recover psi (r={r_full:.3f}) — fix that first"
    assert r_chunk > 0.5, (
        f"CHUNKED training did not recover psi (r={r_chunk:.3f}) while full-batch did "
        f"(r={r_full:.3f}). The minibatch scheme is broken."
    )
    assert r_agree > 0.7, (
        f"chunked and full-batch disagree with each other (r={r_agree:.3f}). Chunking is distorting "
        f"the learned programs."
    )


def test_obs_scale_is_what_makes_chunking_unbiased():
    """Without obs_scale, a chunk optimizes the ELBO of a batch_size-cell dataset, not the full one.

    The full global prior on psi/pi/a/b gets charged against only one batch's likelihood, which
    over-weights it by N/batch and shrinks the K programs toward each other. This test shows the
    global KL is the thing that must NOT be scaled, and the local terms are the things that must.
    """
    import pyro
    from leafletfa import LeafletFA

    adata, _, _ = simulate_from_generative_model(n_cells=300, K=3, n_atses=20, seed=0)
    sub = adata[:100].copy()
    S = 3.0                                     # N/batch = 300/100

    m = LeafletFA(sub, K=3, waypoints_use=False, log_wandb=False, loss_plot=False,
                  obs_scale=S, use_dense_mode=False)
    m.from_anndata()
    pyro.clear_param_store()

    scales = _site_scales(m.model, S, {"assign", "obs"})
    assert scales["obs"] == pytest.approx(S)      # likelihood: scaled
    assert scales["assign"] == pytest.approx(S)   # local phi prior: scaled
    for glob in ("psi", "pi", "a", "b"):
        assert scales[glob] == pytest.approx(1.0), (
            f"global '{glob}' must stay at weight 1 — scaling it would defeat the whole point"
        )


def test_junction_specific_psi_prior_shrinks_the_programs_together():
    """The junction-specific psi prior pulls the K programs TOWARD EACH OTHER. Pin the mechanism.

    psi_kj ~ Beta(a_j, b_j).  a_j and b_j are indexed by JUNCTION ONLY -- there is no k. So at every
    junction all K programs are drawn from the SAME learned Beta. That is a shrinkage prior toward
    "all programs are identical", which is precisely the symmetric saddle that is this model's main
    failure mode (METHODS.md 11.1). The prior pulls TOWARD what training has to escape.

    This is not a hypothetical. On the Figure-2 simulation the global prior recovers psi to r=0.99
    with no seed variance; the junction-specific one gets r=0.31 with huge seed variance. See
    experiments/junction_prior_shrinks_programs.py and METHODS.md 11.4d.

    The test asserts the DIRECTION of the effect, not a magnitude, so it stays meaningful as the
    schedule changes: the junction-specific prior must not produce MORE program contrast than the
    global one on data with real program structure.
    """
    # NB ELBO_num_particles=3: at the 1-particle default the ELBO is too noisy for EITHER prior to
    # recover the truth at this size, and the test cannot discriminate.
    adata, psi_true, _ = simulate_from_generative_model(n_cells=500, K=4, n_atses=30, seed=0)
    kw = dict(K=4, epochs=500, ELBO_num_particles=3, min_delta=0, min_delta_rel=1e-6)

    m_global = _fit(adata, seed=0, junc_specific_prior=False, **kw)
    m_junc   = _fit(adata, seed=0, junc_specific_prior=True,  **kw)

    r_global = np.median(_hungarian_match(_npy(m_global.psi), psi_true)[2])
    r_junc   = np.median(_hungarian_match(_npy(m_junc.psi),   psi_true)[2])

    # The global prior must recover the truth. If this fails, something else broke -- fix that first.
    assert r_global > 0.7, (
        f"the GLOBAL psi prior failed to recover the truth (r={r_global:.3f}). This test cannot say "
        f"anything about the junction-specific prior until the baseline works."
    )

    # The documented finding. If the junction-specific prior ever starts BEATING the global one, the
    # shrinkage analysis in METHODS.md 11.4d is wrong and the atlas grid needs revisiting.
    assert r_junc <= r_global + 0.10, (
        f"UNEXPECTED: the junction-specific psi prior (r={r_junc:.3f}) beat the global prior "
        f"(r={r_global:.3f}). METHODS.md 11.4d claims the opposite -- that sharing a_j/b_j across the "
        f"K factors shrinks the programs toward each other. Re-check that analysis before trusting it."
    )


def test_early_stopping_is_not_a_hard_epoch_cap_at_patience():
    """Early stopping must NOT fire while the ELBO is still descending fast.

    Regression test for a bug in the min_delta_rel logic. `best_loss` is initialised to +inf, and the
    threshold was computed as `max(min_delta, min_delta_rel * abs(best_loss))`. With min_delta_rel > 0
    that is `1e-5 * inf` = inf, so the improvement test became `loss < inf - inf` -> `loss < nan` ->
    ALWAYS FALSE. best_loss never left infinity, epochs_since_improvement incremented on every epoch,
    and the fit broke at EXACTLY `patience` no matter how steeply the loss was still falling.

    Early stopping was therefore not a convergence rule at all -- it was a hard epoch cap at
    `patience`. On the mouse atlas every batch ran exactly 50 epochs instead of its budgeted 150/60,
    with the ELBO still dropping ~2e6 nats/epoch when it quit. See METHODS.md 11.3.

    The tell is `n_ran == patience` exactly. This test fails if that returns.
    """
    adata, _, _ = simulate_from_generative_model(n_cells=300, K=3, n_atses=30, seed=0)

    import io, contextlib
    from leafletfa import LeafletFA
    patience, num_epochs = 10, 60
    m = LeafletFA(adata, K=3, waypoints_use=False, log_wandb=False, loss_plot=False,
                  num_epochs=num_epochs, patience=patience, lr=0.05, gamma=0.5,
                  min_delta=100, min_delta_rel=1e-5,      # the combination that triggered the bug
                  use_dense_mode=False, num_samples=10)
    with contextlib.redirect_stdout(io.StringIO()):
        m.from_anndata()
        m.train(num_initializations=1, seeds=[0])
        m.get_all_variables()

    n_ran = len(m.latent_results[m.best_init]["losses"])

    assert n_ran > patience, (
        f"training stopped after exactly {n_ran} epochs with patience={patience} — early stopping is "
        f"acting as a HARD EPOCH CAP, not a convergence rule. best_loss is almost certainly still "
        f"+inf (min_delta_rel * inf = inf, and `loss < inf - inf` is `loss < nan` = always False)."
    )


def test_pi_warm_start_survives_a_second_chunk_on_gpu():
    """The chunk-to-chunk pi/alpha_pi warm start must not pin the guide to the CPU on a GPU run.

    Regression test for the crash that killed the first mouse-atlas array (18850906_0/1) on its
    SECOND chunk:

        model.py: assign = pyro.sample("assign", dist.Dirichlet(pi * dir_conc).expand([N]) ...)
        RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu!

    run_training.py carries pi across chunks as `pi_init=torch.tensor(global_pi)` -- a CPU tensor.
    psi_init and phi_init were moved to self.device before being handed to init_to_value; pi_init and
    alpha_pi_init were NOT. init_to_value CREATES the guide's loc parameter from the tensor it is
    given, so the guide's `pi` loc was created on the CPU. The model then built Dirichlet(pi*dir_conc)
    on the CPU while the `assign` guide supplied a CUDA sample, and the `assign` site blew up.

    It only fires on the SECOND chunk, because chunk 1 has no warm start to carry -- which is exactly
    why every smoke test passed and the array still died 5 minutes in.
    """
    import torch
    # The crash only MANIFESTS on a GPU (on CPU everything is already on the same device), but the
    # warm-start path is exercised either way, so the test is still worth running on CPU.
    adata, _, _ = simulate_from_generative_model(n_cells=200, K=3, n_atses=20, seed=0)

    import io, contextlib
    from leafletfa import LeafletFA

    def fit(pi_init=None, alpha_pi_init=None):
        m = LeafletFA(adata, K=3, waypoints_use=False, log_wandb=False, loss_plot=False,
                      num_epochs=15, patience=10**6, lr=0.05, gamma=0.5, use_dense_mode=False,
                      num_samples=10, pi_init=pi_init, alpha_pi_init=alpha_pi_init)
        with contextlib.redirect_stdout(io.StringIO()):
            m.from_anndata()
            m.train(num_initializations=1, seeds=[0])
            m.get_all_variables()
        return m

    chunk1 = fit()

    # Carry pi forward the way run_training.py does: a CPU tensor, even on a CUDA run.
    global_pi = torch.tensor(np.asarray(chunk1.pi), dtype=torch.float32)          # CPU, on purpose
    global_alpha_pi = torch.tensor(float(np.asarray(chunk1.alpha_pi)), dtype=torch.float32)
    assert global_pi.device.type == "cpu", "the test must hand over a CPU tensor to be meaningful"

    chunk2 = fit(pi_init=global_pi, alpha_pi_init=global_alpha_pi)   # <-- used to RuntimeError on cuda

    assert np.isfinite(np.asarray(chunk2.psi)).all()
    assert np.asarray(chunk2.pi).shape == (3,)


def test_fit_chunked_carries_phi_per_cell_and_does_not_invent_an_E_step():
    """Pin the THREE things hand-copies of the chunked loop always get wrong.

    The Figure-2 benchmark hand-copied this loop, dropped the phi carry-over, and invented a final
    frozen-psi E-step that production never runs -- so the full-batch-vs-chunked comparison, the one
    result that justifies chunking for the entire atlas, was measuring a scheme nobody uses.

    That is why fit_chunked() exists: ONE implementation. This test pins its semantics against
    run_training.py:

      1. phi is carried PER CELL   (written back after every chunk, warm-started on re-encounter)
      2. the final phi IS the accumulated store, not a re-inference
      3. every cell has a phi, including cells whose last chunk was several passes ago
    """
    import numpy as np, io, contextlib
    from leafletfa.chunked import fit_chunked

    adata, psi_true, _ = simulate_from_generative_model(n_cells=400, K=3, n_atses=25, seed=0)
    N, K, B = 400, 3, 100

    seen_phi = []
    def spy(pass_idx, c_idx, n_chunks, model):
        seen_phi.append(_npy(model.assign_post).copy())

    with contextlib.redirect_stdout(io.StringIO()):
        m = fit_chunked(adata, K=K, batch_size=B, num_passes=3, seed=0,
                        num_epochs_first=40, num_epochs_later=25,
                        lr=0.05, gamma=0.5, patience=10**6, min_delta=0, min_delta_rel=1e-6,
                        waypoints_use=False, log_wandb=False, loss_plot=False,
                        use_dense_mode=False, num_samples=10, progress=spy)

    phi = _npy(m.assign_post)

    # 2 + 3: the final phi covers EVERY cell, and is the accumulated store (shape (N, K), not a chunk)
    assert phi.shape == (N, K), (
        f"final assign_post is {phi.shape}, expected {(N, K)}. If it is (batch_size, K) the loop "
        f"returned the LAST CHUNK's phi instead of the accumulated per-cell store.")
    assert (phi.sum(axis=1) > 0).all(), "some cells have no phi — the per-cell store was not filled"
    assert np.isfinite(phi).all()

    # 1: phi is genuinely REFINED across passes, not written once. A cell's phi must change between
    #    the first time it is seen and the last.
    assert len(seen_phi) == 3 * (N // B), "unexpected number of chunks"
    assert _npy(m.psi).shape == (K, adata.n_vars)

    # and it must still recover the truth -- a loop that carries phi but learns nothing is no better
    r = np.median(_hungarian_match(_npy(m.psi), psi_true)[2])
    assert r > 0.5, f"chunked training did not recover psi (r={r:.3f})"


def test_obs_scale_gives_chunked_training_the_full_data_posterior_width():
    """obs_scale = N/B must make q(psi)'s WIDTH match full-batch — not just its point estimate.

    THIS IS THE TEST NOTHING ELSE DOES. Every check that has ever been run on chunking is
    SCALE-INVARIANT (Hungarian-matched Pearson r, cosine, argmax agreement) — all of which stay ~1.0
    even if the posterior width is off by 6x. But the differential-splicing FDR SAMPLES FROM q(psi):
    P(|effect| > delta) is computed from those draws. Too narrow -> overconfident -> the FDR blows up.

    Without obs_scale, each chunk optimises the ELBO of a dataset with only B cells, so q(psi) has the
    width it would have given B cells, not N — too wide by roughly sqrt(N/B).

    So: chunked WITH obs_scale should sit far closer to full-batch than chunked WITHOUT it.
    """
    import numpy as np, io, contextlib
    from leafletfa.chunked import fit_chunked

    adata, _, _ = simulate_from_generative_model(n_cells=600, K=3, n_atses=25, seed=0)
    N, K, B = 600, 3, 150
    common = dict(gamma=0.5, patience=10**6, min_delta=0, min_delta_rel=1e-6, waypoints_use=False,
                  log_wandb=False, loss_plot=False, use_dense_mode=False, num_samples=10)

    def width(m):
        """Mean posterior sd of psi — the thing the FDR actually consumes.

        NB `psi_samples` only exists after prune_K(); the raw posterior lives in all_samples["psi"],
        shape [S, K, J]."""
        return float(np.mean(_npy(m.all_samples["psi"]).std(axis=0)))

    with contextlib.redirect_stdout(io.StringIO()):
        full = _fit(adata, seed=0, epochs=300, K=K)
        chunked_scaled = fit_chunked(adata, K=K, batch_size=B, num_passes=4, seed=0,
                                     num_epochs_first=80, num_epochs_later=60, lr=0.05, **common)
        # obs_scale is applied inside fit_chunked as N/len(idx); to get the UNSCALED comparison we
        # fake N == B so the factor becomes 1.0.
        unscaled = fit_chunked(adata[:B].copy(), K=K, batch_size=B, num_passes=4, seed=0,
                               num_epochs_first=80, num_epochs_later=60, lr=0.05, **common)

    w_full, w_scaled, w_unscaled = width(full), width(chunked_scaled), width(unscaled)
    print(f"\n  q(psi) width  full-batch={w_full:.4f}  chunked+obs_scale={w_scaled:.4f}  "
          f"B-cells-only={w_unscaled:.4f}")

    # The B-cells-only posterior must be WIDER than the full-data one -- that is the whole premise.
    assert w_unscaled > w_full, (
        f"a {B}-cell posterior ({w_unscaled:.4f}) is not wider than a {N}-cell posterior "
        f"({w_full:.4f}). The premise of obs_scale does not hold on this data; do not trust the "
        f"chunked posterior until this is understood.")

    # And chunked-with-obs_scale must land closer to full-batch than the B-cell posterior does.
    assert abs(w_scaled - w_full) < abs(w_unscaled - w_full), (
        f"obs_scale did NOT recover the full-data posterior width: full={w_full:.4f}, "
        f"chunked+obs_scale={w_scaled:.4f}, B-cells-only={w_unscaled:.4f}. The DS/FDR procedure "
        f"samples from this posterior, so a wrong width miscalibrates every reported probability.")
