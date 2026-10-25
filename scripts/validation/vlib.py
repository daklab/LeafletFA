"""Shared pieces for the LeafletFA validation (see LeafletFA_validation/VALIDATION_PREREG.md).

- data access for the SpliceVI TMS / Allen files (junction counts, labels, mice)
- the within-cell event split
- the full Dirichlet-multinomial log-pmf (with the multinomial coefficient), per cell, on CPU
- TRAIN-fitted nulls: population psi, per-label psi, per-ATSE maximum-likelihood concentration
- the simulator for check 4
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "splicevi_baseline"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from run_leafletfa_baseline import read_counts, read_cells, read_var, _strings, _splicing_root  # noqa: E402,F401

D = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/"
B = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_splicevi_baseline/"
V = "/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/"
DS_CODE = {"TMS": "TMS", "Allen": "AB"}
CLIP, EPS = 1e-4, 1e-2                       # the model's psi clip and alpha floor
SPLIT_SEED = 20261025


def paths(ds):
    s = f"{DS_CODE[ds]}_ONLY_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu"
    return dict(train=D + "train_70_30_" + s, test=D + "test_30_70_" + s)


def obs_labels(path):
    """(cell ids, label = tissue | broad_cell_type, mouse id) from an h5mu's obs."""
    import h5py
    with h5py.File(path, "r") as f:
        o = f["obs"]
        cells = _strings(o[o.attrs["_index"]])
        lab = np.char.add(np.char.add(_strings(o["tissue"]).astype(str), " | "),
                          _strings(o["broad_cell_type"]).astype(str))
        mouse = _strings(o["mouse.id"]).astype(str)
    return cells, lab, mouse


def event_index(ev):
    u, inv = np.unique(ev, return_inverse=True)
    return u, inv.astype(np.int64)


# ── event split ───────────────────────────────────────────────────────────────────────────────
def _mix(x):
    """splitmix64 on uint64 arrays -- a reproducible per-(cell, event) coin."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return x ^ (x >> np.uint64(31))


def event_split(Y, inv, cell_keys, seed=SPLIT_SEED):
    """Split each cell's observed events 50/50. Returns (Y_A, Y_B), complementary, same shape.
    cell_keys: integer key per row (stable across inputs, e.g. hash of the cell id)."""
    Y = sp.csr_matrix(Y)
    rows = np.repeat(np.arange(Y.shape[0]), np.diff(Y.indptr))
    n_atse = int(inv.max()) + 1
    key = cell_keys[rows].astype(np.uint64) * np.uint64(n_atse) + inv[Y.indices].astype(np.uint64)
    coin = (_mix(key ^ np.uint64(seed)) & np.uint64(1)).astype(bool)
    # copies: eliminate_zeros() compacts indices/indptr IN PLACE, and shared arrays would corrupt Y
    A = sp.csr_matrix((np.where(coin, 0, Y.data), Y.indices.copy(), Y.indptr.copy()), shape=Y.shape)
    Bm = sp.csr_matrix((np.where(coin, Y.data, 0), Y.indices.copy(), Y.indptr.copy()), shape=Y.shape)
    A.eliminate_zeros(); Bm.eliminate_zeros()
    return A, Bm


def cell_keys(cells):
    import hashlib
    return np.array([int(hashlib.md5(c.encode()).hexdigest()[:15], 16) for c in cells], dtype=np.uint64)


# ── full DM log-pmf per cell ─────────────────────────────────────────────────────────────────
def score(Y, inv, mu_fn, conc_e, chunk=2048, device="cpu"):
    """Per-cell DM log-pmf (incl. multinomial coefficient) and reads, summed over observed events.

    Y      : CSR counts (cells x J)
    inv    : [J] event index
    mu_fn  : f(row_slice) -> dense [rows, J] torch composition (sums to 1 within each event)
    conc_e : [n_atse] concentration
    Returns (logL [N] float64, reads [N] float64, n_obs_events [N]).
    """
    from leafletfa.dirichlet_multinomial import SparseObs
    Y = sp.csr_matrix(Y, dtype=np.float32)
    N = Y.shape[0]
    inv_t = torch.as_tensor(inv)
    conc = torch.as_tensor(np.asarray(conc_e, np.float64))
    out_l, out_r, out_e = np.zeros(N), np.zeros(N), np.zeros(N, np.int64)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        so = SparseObs(Y[a:b], inv_t)
        rows, cols, y, seg, seg_row, n_seg = so.batch(torch.arange(b - a))
        y = y.double()
        mu = mu_fn(slice(a, b)).double()
        m = mu.reshape(-1).index_select(0, rows * mu.shape[1] + cols).clamp(CLIP, 1 - CLIP)
        al = (conc[inv_t[cols]] * m).clamp_min(EPS)
        jt = torch.lgamma(y + al) - torch.lgamma(al) - torch.lgamma(y + 1)
        A = torch.zeros(n_seg, dtype=torch.float64).index_add(0, seg, al)
        Ne = torch.zeros(n_seg, dtype=torch.float64).index_add(0, seg, y)
        et = torch.lgamma(A) - torch.lgamma(Ne + A) + torch.lgamma(Ne + 1)
        lp = torch.zeros(b - a, dtype=torch.float64).index_add(0, rows, jt).index_add(0, seg_row, et)
        out_l[a:b] = lp.numpy()
        out_r[a:b] = torch.zeros(b - a, dtype=torch.float64).index_add(0, rows, y).numpy()
        out_e[a:b] = torch.bincount(seg_row, minlength=b - a).numpy()
    return out_l, out_r, out_e


def mu_from(phi, psi):
    phi = torch.as_tensor(np.asarray(phi, np.float64)); psi = torch.as_tensor(np.asarray(psi, np.float64))
    return lambda s: phi[s] @ psi


def mu_label(lab_idx, psi_lab):
    li = torch.as_tensor(lab_idx); P = torch.as_tensor(np.asarray(psi_lab, np.float64))
    return lambda s: P[li[s]]


# ── nulls ────────────────────────────────────────────────────────────────────────────────────
def usage(Y, inv, groups=None, n_groups=1):
    """Read-weighted within-event usage per group: [n_groups, J]; events with no reads -> NaN."""
    Y = sp.csr_matrix(Y, dtype=np.float64)
    g = np.zeros(Y.shape[0], np.int64) if groups is None else np.asarray(groups)
    G = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(n_groups, Y.shape[0]))
    tot = np.asarray((G @ Y).todense())                                       # [G, J]
    S = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)))
    ev_tot = tot @ S                                                          # [G, n_atse]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = tot / np.asarray(ev_tot)[:, inv]
    return u


def fit_conc(Y, inv, mu_fn, grid=None, chunk=2048, floor=2.0):
    """Per-ATSE ML concentration on Y given a fixed composition: coarse log-grid, then a finer one.
    Only conc-dependent terms are evaluated. Returns [n_atse]."""
    from leafletfa.dirichlet_multinomial import SparseObs
    n_atse = int(inv.max()) + 1
    inv_t = torch.as_tensor(inv)
    Y = sp.csr_matrix(Y, dtype=np.float32)

    def ll_grid(cgrid):                                  # cgrid [n_atse, G] -> [n_atse, G]
        Gn = cgrid.shape[1]
        tot = torch.zeros(n_atse, Gn, dtype=torch.float64)
        cg = torch.as_tensor(cgrid)
        for a in range(0, Y.shape[0], chunk):
            b = min(a + chunk, Y.shape[0])
            so = SparseObs(Y[a:b], inv_t)
            rows, cols, y, seg, seg_row, n_seg = so.batch(torch.arange(b - a))
            y = y.double()
            mu = mu_fn(slice(a, b)).double()
            m = mu.reshape(-1).index_select(0, rows * mu.shape[1] + cols).clamp(CLIP, 1 - CLIP)
            e_of_ent = inv_t[cols]
            Smu = torch.zeros(n_seg, dtype=torch.float64).index_add(0, seg, m)
            Ne = torch.zeros(n_seg, dtype=torch.float64).index_add(0, seg, y)
            e_of_seg = torch.zeros(n_seg, dtype=torch.long).index_copy(0, seg, e_of_ent)
            pos = y > 0
            yp, mp, ep = y[pos], m[pos], e_of_ent[pos]
            for gi in range(Gn):
                c_ent = cg[ep, gi]
                al = (c_ent * mp).clamp_min(EPS)
                jt = torch.lgamma(yp + al) - torch.lgamma(al)
                cs = cg[e_of_seg, gi]
                A = (cs * Smu).clamp_min(EPS)
                et = torch.lgamma(A) - torch.lgamma(Ne + A)
                tot[:, gi] += torch.zeros(n_atse, dtype=torch.float64).index_add(0, ep, jt).index_add(0, e_of_seg, et)
        return tot.numpy()
    g1 = np.exp(np.linspace(np.log(floor), np.log(1e4), 41))
    L1 = ll_grid(np.tile(g1, (n_atse, 1)))
    best = L1.argmax(1)
    lo = g1[np.clip(best - 1, 0, 40)]; hi = g1[np.clip(best + 1, 0, 40)]
    g2 = np.exp(np.linspace(np.log(lo), np.log(hi), 11).T)                   # [n_atse, 11]
    L2 = ll_grid(g2)
    c = g2[np.arange(n_atse), L2.argmax(1)]
    return np.maximum(c, floor)


# ── simulator ────────────────────────────────────────────────────────────────────────────────
def simulate(Y_template, inv, phi, psi, conc_e, thin=None, seed=0):
    """Counts with the template's observed (cell, event) totals (optionally binomially thinned),
    drawn y ~ DirMult(N_e, conc_e * (phi_c psi)_e). Returns CSR float32."""
    rng = np.random.default_rng(seed)
    Y = sp.csr_matrix(Y_template, dtype=np.float64)
    n_atse = int(inv.max()) + 1
    S = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)))
    E = (Y @ S).tocsr(); E.eliminate_zeros()                                  # [N, n_atse] totals
    order = np.argsort(inv, kind="stable"); bounds = np.searchsorted(inv[order], np.arange(n_atse + 1))
    R, C, Vv = [], [], []
    for c in range(Y.shape[0]):
        es = E.indices[E.indptr[c]:E.indptr[c + 1]]
        ns = E.data[E.indptr[c]:E.indptr[c + 1]].astype(np.int64)
        if thin is not None:
            ns = rng.binomial(ns, thin)
        mu_c = phi[c] @ psi
        for e, n in zip(es, ns):
            if n <= 0:
                continue
            js = order[bounds[e]:bounds[e + 1]]
            al = np.clip(conc_e[e] * np.clip(mu_c[js], CLIP, 1 - CLIP), EPS, None)
            p = rng.dirichlet(al)
            R.append(np.full(len(js), c)); C.append(js); Vv.append(rng.multinomial(n, p))
    out = sp.csr_matrix((np.concatenate(Vv).astype(np.float32), (np.concatenate(R), np.concatenate(C))),
                        shape=Y.shape)
    out.eliminate_zeros()
    return out


def simulate_fast(Y_template, inv, phi, psi, conc_e, thin=None, seed=0, chunk=4000):
    """Vectorised simulate(): same model, drawn per block of cells.

    Dirichlet via normalised Gamma draws; the multinomial via sequential conditional binomials over
    each event's junctions (event sizes <= 43). Returns CSR float32 with the template's shape."""
    from leafletfa.dirichlet_multinomial import SparseObs
    rng = np.random.default_rng(seed)
    Yt = sp.csr_matrix(Y_template, dtype=np.float32)
    inv_t = torch.as_tensor(inv)
    out = []
    for a in range(0, Yt.shape[0], chunk):
        b = min(a + chunk, Yt.shape[0])
        so = SparseObs(Yt[a:b], inv_t)
        rows, cols, y, seg, seg_row, n_seg = (x.numpy() if torch.is_tensor(x) else x for x in so.batch(torch.arange(b - a)))
        N = np.bincount(seg, weights=y, minlength=n_seg).round().astype(np.int64)
        if thin is not None:
            N = rng.binomial(N, thin)
        mu = np.einsum("nk,nk->n", phi[a:b][rows], psi.T[cols])
        al = np.clip(conc_e[inv[cols]] * np.clip(mu, CLIP, 1 - CLIP), EPS, None)
        g = rng.gamma(al)
        gs = np.bincount(seg, weights=g, minlength=n_seg)
        p = g / np.where(gs[seg] > 0, gs[seg], 1.0)
        # position of each entry within its segment (entries of a segment are contiguous)
        first = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
        start = np.repeat(first, np.diff(np.r_[first, len(seg)]))
        pos = np.arange(len(seg)) - start
        size = np.bincount(seg, minlength=n_seg)
        rem_n = N.copy(); rem_p = np.ones(n_seg)
        x = np.zeros(len(seg), np.int64)
        for k in range(int(size.max())):
            e = np.flatnonzero(pos == k)
            s_ = seg[e]
            last = pos[e] == size[s_] - 1
            q = np.where(last, 1.0, np.clip(p[e] / np.where(rem_p[s_] > 1e-12, rem_p[s_], 1.0), 0, 1))
            draw = rng.binomial(rem_n[s_], q)
            x[e] = draw; rem_n[s_] -= draw; rem_p[s_] -= p[e]
        keep = x > 0
        out.append(sp.csr_matrix((x[keep].astype(np.float32), (rows[keep], cols[keep])), shape=(b - a, Yt.shape[1])))
    return sp.vstack(out).tocsr()
