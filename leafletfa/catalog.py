"""Cell-type splicing estimates from a fitted LeafletFA model (the catalog estimator).

Ported from ``scripts/validation/c10_catalog.py`` (check 10); the point estimate is the one shipped in the catalog and the
browser. Reproduction against the saved check 10 output: ``scripts/validation/catalog_port_check.py``.

Estimator. For cell c and junction j in ATSE e, the posterior-mean usage is

    rho~_cj = (gamma_e rho_cj + y_cj) / (gamma_e + N_ce),    rho_c = phi_c psi,

with y_cj the cell's junction reads, N_ce its reads at the ATSE and gamma_e the fitted concentration (``bb_conc_vec``).
A cell without reads at e contributes its program prior rho_cj. The cell-type psi is the mean of rho~ over the type's
cells. Raw pseudobulk psi (sum y / sum N), reads, cells with reads and mice are returned beside it.

Intervals are not computed here, and a user's own fit has none. The atlas catalog's interval is the check 12 (a)+(b) 95 %
predictive interval for a cell type's held-out pseudobulk psi at its read depth (``scripts/validation/c12_intervals.py``,
``c12_export.py``). Its model-error term is calibrated on held-out cells of those atlases, so it is not a property of the fit.

Example::

    from leafletfa.catalog import load_fit, celltype_psi
    fit = load_fit("path/to/fit")                       # directory with fit.npz from run_leafletfa_baseline.py
    out = celltype_psi(Y, fit, labels)                  # Y: cells x junctions counts, rows in fit["cell_id"] order
    out["psi"][out["types"].index("Lung_B CELL")]       # junctions, in fit["junction_id"] order
"""
import os

import numpy as np
import scipy.sparse as sp

__all__ = ["load_fit", "celltype_psi"]


def load_fit(fit_dir):
    """Read fit.npz written by the production driver. Returns phi (cells x K), psi (K x J), gamma (ATSE,), inv
    (J,, junction -> ATSE index), cell_id, junction_id, event_id. phi and psi are cast to float64."""
    z = np.load(os.path.join(fit_dir, "fit.npz") if os.path.isdir(fit_dir) else fit_dir, allow_pickle=True)
    return dict(phi=np.asarray(z["phi"], np.float64), psi=np.asarray(z["psi_mean"], np.float64),
                gamma=np.asarray(z["bb_conc_vec"], np.float64), inv=np.asarray(z["inv_ev"], np.int64),
                cell_id=z["cell_id"].astype(str), junction_id=z["junction_id"].astype(str), event_id=z["event_id"].astype(str))


def celltype_psi(Y, fit, labels, types=None, mice=None, min_cells=1):
    """Cell-type psi from a fit.

    Y        cells x junctions counts (scipy sparse or dense), rows aligned with ``fit["phi"]``, columns with ``fit["inv"]``.
    fit      dict from ``load_fit`` (needs phi, psi, gamma, inv).
    labels   (cells,) group label per cell.
    types    labels to estimate, in output order; default: sorted labels with >= ``min_cells`` cells. Cells with other
             labels are ignored (each type's estimate uses only its own cells).
    mice     optional (cells,) mouse / donor id; adds ``mice`` (types x ATSE): mice with reads at the ATSE.

    Returns dict: types (list), cells (T,), psi (T x J), raw_psi (T x J, NaN where the type has no reads at the ATSE),
    junction_reads (T x J), atse_reads (T x ATSE), cells_covering (T x ATSE), and mice if given.
    """
    phi, psi, gam, inv = fit["phi"], fit["psi"], np.asarray(fit["gamma"], np.float64), np.asarray(fit["inv"], np.int64)
    Y = sp.csr_matrix(Y, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    n_cells, J = Y.shape
    K, n_atse = phi.shape[1], len(gam)
    if phi.shape[0] != n_cells or psi.shape != (K, J) or len(inv) != J or len(labels) != n_cells:
        raise ValueError(f"shapes do not match: Y {Y.shape}, phi {phi.shape}, psi {psi.shape}, inv {inv.shape}, labels {labels.shape}")
    if types is None:
        u, c = np.unique(labels, return_counts=True)
        types = [t for t, n in zip(u, c) if n >= min_cells]
    types = list(types)
    code = {t: i for i, t in enumerate(types)}
    g = np.array([code.get(t, -1) for t in labels]); keep = np.flatnonzero(g >= 0); g = g[keep]
    T = len(types)
    Y = Y[keep]; phi = np.asarray(phi, np.float64)[keep]
    E = sp.csr_matrix((np.ones(J), (np.arange(J), inv)), shape=(J, n_atse))
    N = (Y @ E).tocsr(); N.eliminate_zeros()                                  # cells x ATSE
    Nc = N.tocoo()
    Vw = sp.csr_matrix((Nc.data / (gam[Nc.col] + Nc.data), (Nc.row, Nc.col)), shape=N.shape)   # N / (gamma + N)
    Yc = Y.tocoo()
    Ncj = np.asarray(N[Yc.row, inv[Yc.col]]).ravel()
    R = sp.csr_matrix((Yc.data / (gam[inv[Yc.col]] + Ncj), (Yc.row, Yc.col)), shape=Y.shape)    # y / (gamma + N)
    O = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(T, len(g)))
    n = np.bincount(g, minlength=T).astype(np.float64)
    Rs = np.asarray((O @ R).todense())
    Phis = O @ phi                                                             # T x K
    out = np.empty((T, J))
    for k in range(T):
        i = np.flatnonzero(g == k)
        A = Phis[k][None, :] - Vw[i].T.tocsr() @ phi[i]                        # ATSE x K: sum_c (1 - v_ce) phi_c
        out[k] = (np.einsum("jk,kj->j", A[inv], psi) + Rs[k]) / n[k]
    jr = np.asarray((O @ Y).todense()); er = np.asarray((O @ N).todense())
    cov = np.asarray((O @ (N > 0).astype(np.float64)).todense())
    tot = er[:, inv]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(tot > 0, jr / tot, np.nan)
    res = dict(types=types, cells=n, psi=out, raw_psi=raw, junction_reads=jr, atse_reads=er, cells_covering=cov)
    if mice is not None:
        _, mi = np.unique(np.asarray(mice)[keep], return_inverse=True)
        M = mi.max() + 1
        key = np.unique(g[Nc.row].astype(np.int64) * n_atse * M + Nc.col.astype(np.int64) * M + mi[Nc.row])
        tc = key // M
        m = np.zeros((T, n_atse), np.int64); np.add.at(m, (tc // n_atse, tc % n_atse), 1)
        res["mice"] = m
    return res
