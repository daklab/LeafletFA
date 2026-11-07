"""leafletfa.catalog.celltype_psi equals the per-cell definition of the catalog estimator.

    python -m pytest -q tests/test_catalog.py
The full-size reproduction of the saved check 10 estimates is scripts/validation/catalog_port_check.py (sbatch).
"""
import numpy as np
import scipy.sparse as sp

from leafletfa.catalog import celltype_psi


def _toy(N=80, n_atse=15, K=3, seed=0):
    rng = np.random.default_rng(seed)
    ev = rng.permutation(np.repeat(np.arange(n_atse), rng.integers(2, 5, n_atse)))
    J = len(ev)
    psi = np.zeros((K, J))
    for e in range(n_atse):
        c = np.flatnonzero(ev == e)
        psi[:, c] = rng.dirichlet(np.ones(len(c)), K)
    phi = rng.dirichlet(np.ones(K), N)
    y = np.zeros((N, J))
    for c in range(N):
        for e in np.flatnonzero(rng.random(n_atse) < 0.4):
            cols = np.flatnonzero(ev == e)
            y[c, cols] = rng.multinomial(rng.integers(1, 40), (phi[c] @ psi[:, cols]))
    gam = rng.uniform(0.5, 20, n_atse)
    labels = rng.choice(["a", "b", "c", "x"], N)
    mice = rng.choice(["m1", "m2", "m3"], N)
    return y, dict(phi=phi, psi=psi, gamma=gam, inv=ev), labels, mice


def test_matches_per_cell_definition():
    y, fit, labels, mice = _toy()
    out = celltype_psi(sp.csr_matrix(y), fit, labels, types=["a", "b", "c"], mice=mice)
    inv, gam = fit["inv"], fit["gamma"]
    rho = fit["phi"] @ fit["psi"]
    Nce = np.stack([y[:, inv == e].sum(1) for e in range(len(gam))], 1)        # cells x ATSE
    rt = (gam[inv] * rho + y) / (gam[inv] + Nce[:, inv])                      # per-cell rho~
    for k, t in enumerate(["a", "b", "c"]):
        m = labels == t
        np.testing.assert_allclose(out["psi"][k], rt[m].mean(0), rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(out["atse_reads"][k], Nce[m].sum(0))
        np.testing.assert_array_equal(out["cells_covering"][k], (Nce[m] > 0).sum(0))
        tot = Nce[m].sum(0)[inv]
        np.testing.assert_allclose(out["raw_psi"][k][tot > 0], (y[m].sum(0) / tot)[tot > 0])
        assert np.isnan(out["raw_psi"][k][tot == 0]).all()
        nm = np.array([len(np.unique(mice[m][Nce[m][:, e] > 0])) for e in range(len(gam))])
        np.testing.assert_array_equal(out["mice"][k], nm)
    # psi sums to 1 within each ATSE for every type
    for e in range(len(gam)):
        np.testing.assert_allclose(out["psi"][:, inv == e].sum(1), 1, atol=1e-10)


def test_other_labels_do_not_change_a_type():
    y, fit, labels, _ = _toy(seed=3)
    a = celltype_psi(y, fit, labels, types=["a"])["psi"]
    m = labels != "x"
    b = celltype_psi(y[m], dict(fit, phi=fit["phi"][m]), labels[m], types=["a"])["psi"]
    np.testing.assert_allclose(a, b, rtol=1e-12)
