"""The sparse observed-event Dirichlet-multinomial equals the dense one, and float64 phi changes
nothing but precision.

    python -m pytest -q tests/test_sparse_dm.py
"""
import numpy as np
import scipy.sparse as sp
import torch

from leafletfa.dirichlet_multinomial import (build_atse_index, dm_log_prob, dm_log_prob_sparse,
                                             SparseObs, segment_softmax)


def _data(N=60, n_atse=25, K=4, seed=0, density=0.3):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 6, n_atse)
    ev = np.repeat(np.arange(n_atse), sizes)
    ev = rng.permutation(ev)                      # events not contiguous in the column order
    J = len(ev)
    y = np.zeros((N, J), np.float32)
    for c in range(N):
        for e in np.flatnonzero(rng.random(n_atse) < density):
            cols = np.flatnonzero(ev == e)
            y[c, cols] = rng.multinomial(rng.integers(1, 30), rng.dirichlet(np.ones(len(cols))))
    y[3] = 0                                      # a cell with no reads at all
    return y, ev, K


def _globals(J, n_atse, K, atse_idx, seed=1):
    g = torch.Generator().manual_seed(seed)
    psi = segment_softmax(torch.randn(K, J, generator=g), atse_idx, n_atse)
    phi = torch.distributions.Dirichlet(torch.ones(K)).sample((1,)).squeeze(0)
    return psi, phi


def test_sparse_equals_dense_logprob_and_grads():
    y, ev, K = _data()
    atse_idx, _, sizes = build_atse_index(ev)
    n_atse, J, N = len(sizes), len(ev), y.shape[0]
    psi0, _ = _globals(J, n_atse, K, atse_idx)
    phi0 = torch.distributions.Dirichlet(torch.full((K,), 0.7)).sample((N,))
    conc = (2.0 + 10 * torch.rand(n_atse))[atse_idx]
    w = torch.where(sizes > 1, sizes / (sizes - 1.0), torch.ones_like(sizes, dtype=torch.float32))
    for weight in (None, w.float()):
        phi_d = phi0.clone().requires_grad_(); psi_d = psi0.clone().requires_grad_()
        mu = (phi_d @ psi_d).clamp(1e-4, 1 - 1e-4)
        lp_d = dm_log_prob(torch.as_tensor(y), mu, conc, atse_idx, n_atse, event_weight=weight)
        lp_d.sum().backward()
        so = SparseObs(sp.csr_matrix(y), atse_idx)
        phi_s = phi0.clone().requires_grad_(); psi_s = psi0.clone().requires_grad_()
        lp_s = dm_log_prob_sparse(so.batch(torch.arange(N)), phi_s, psi_s, conc, atse_idx,
                                  event_weight=weight)
        lp_s.sum().backward()
        assert torch.allclose(lp_d, lp_s, rtol=1e-5, atol=1e-3), (lp_d - lp_s).abs().max()
        assert torch.allclose(phi_d.grad, phi_s.grad, rtol=1e-4, atol=1e-3)
        assert torch.allclose(psi_d.grad, psi_s.grad, rtol=1e-4, atol=1e-3)
        assert float(lp_s[3]) == 0.0


def test_sparse_batch_subset_matches_rows():
    y, ev, K = _data(seed=2)
    atse_idx, _, sizes = build_atse_index(ev)
    n_atse, J, N = len(sizes), len(ev), y.shape[0]
    psi, _ = _globals(J, n_atse, K, atse_idx)
    phi = torch.distributions.Dirichlet(torch.ones(K)).sample((N,))
    so = SparseObs(sp.csr_matrix(y), atse_idx)
    full = dm_log_prob_sparse(so.batch(torch.arange(N)), phi, psi, 3.0, atse_idx)
    idx = torch.tensor([7, 0, 31, 3, 59, 7])
    sub = dm_log_prob_sparse(so.batch(idx), phi[idx], psi, 3.0, atse_idx)
    assert torch.allclose(full[idx], sub, rtol=1e-6, atol=1e-4)


def test_minibatch_sparse_matches_dense_loss():
    """Same seed, same steps: the sparse model's ELBO trajectory equals the dense one."""
    import pyro
    from leafletfa.minibatch import LeafletFAMinibatch
    y, ev, K = _data(N=80, seed=3)
    clus = np.zeros_like(y)
    for e in np.unique(ev):
        c = ev == e
        clus[:, c] = y[:, c].sum(1, keepdims=True)
    runs = {}
    for name, kw in (("dense", {}), ("sparse", {"sparse": True}),
                     ("dense64", {"phi_dtype": "float64"}),
                     ("sparse64", {"sparse": True, "phi_dtype": "float64"})):
        pyro.clear_param_store()
        m = LeafletFAMinibatch.dm(sp.csr_matrix(y) if kw.get("sparse") else y, clus, K=K,
                                  event_id=ev, device="cpu", **kw)
        runs[name] = np.array(m.fit(num_steps=30, subsample_size=32, lr=0.05, seed=0,
                                    early_stop=False, verbose=False))
        runs[name + "_phi"] = m.phi_mean().numpy()
    assert np.allclose(runs["dense"], runs["sparse"], rtol=1e-4), (runs["dense"] - runs["sparse"])
    assert np.allclose(runs["dense64"], runs["sparse64"], rtol=1e-4)
    # float64 draws a different random stream for q(phi), so trajectories differ step to step;
    # the first step and the end point must still agree
    assert abs(runs["dense"][0] - runs["dense64"][0]) / runs["dense"][0] < 1e-3
    assert abs(runs["dense"][-5:].mean() - runs["dense64"][-5:].mean()) / runs["dense"][-5:].mean() < 0.05
    assert np.allclose(runs["dense_phi"], runs["sparse_phi"], atol=1e-3)
    assert runs["sparse64_phi"].dtype == np.float32


def test_float64_phi_survives_corner_prior():
    """The E-step failure in miniature: pi * dir_conc << 1 drives q(phi) to the simplex edge.
    float32 phi goes non-finite; float64 phi does not (same data, seed and settings)."""
    import pyro
    from leafletfa.minibatch import LeafletFAMinibatch
    y, ev, K = _data(N=200, n_atse=40, K=10, seed=4, density=0.6)
    y = y * 20
    atse_idx, _, sizes = build_atse_index(ev)
    psi, _ = _globals(len(ev), len(sizes), K, atse_idx, seed=5)
    pi = np.full(K, 0.004, np.float32); pi[0] = 1 - pi[1:].sum()    # alpha_k = 0.04 for 9 of 10
    out = {}
    for dt in ("float32", "float64"):
        pyro.clear_param_store()
        m = LeafletFAMinibatch.dm(sp.csr_matrix(y), None, K=K, event_id=ev, device="cpu",
                                  sparse=True, phi_dtype=dt)
        m.freeze_globals(psi=psi.numpy(), pi=pi, alpha_pi=1.0, dir_conc=10.0,
                         bb_conc=np.full(len(sizes), 5.0, np.float32))
        m.infer_phi(num_steps=1500, subsample_size=None, lr=0.3, gamma=1.0, seed=0,
                    early_stop=False, max_nan_restarts=1000)
        out[dt] = m.n_nan_restarts
    assert out["float64"] == 0, out
    print("non-finite restarts:", out)
