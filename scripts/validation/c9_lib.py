"""Check 9 shared pieces (pre-registration amendment 5): the shipped posterior DS test (A51 + A67 gate) run from a fit's
checkpoint, on the TMS-only training cells.

The test is A51's (Leaflet-analysis GenomeBiology_Submission2026/pipelines/05_leafletfa/A51_group_ds_posterior.py):
Delta_j = dphi . psi_:,j per draw (200 draws of psi and of the group-mean phi difference), TV = 0.5 sum_{j in e} |Delta_j|,
P(TV > delta), Bayesian FDR, call at q <= 0.05; an ATSE is tested only if >= 25 cells per arm have a read on it (A67).
fit.npz keeps only point estimates, so q(psi) and q(phi) come from checkpoint.pt (pyro param store state, unconstrained:
the scales go through their constraint's bijection, as pyro does when they are read back).
"""
import os, sys
import numpy as np, pandas as pd, torch
from torch.distributions import biject_to, constraints

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from leafletfa.dirichlet_multinomial import segment_softmax  # noqa: E402

DELTA, S, ALPHA, MIN_COV = 0.15, 200, 0.05, 25
C8 = os.path.join(vlib.V, "c8_consensus")


def fit_dir(seed):
    if seed <= 2:
        return os.path.join(vlib.B, "TMS", f"full_K20_s{seed}_f64", "fit")
    return os.path.join(C8, "fits", f"full_K20_s{seed}_f64", "fit")


def load_posterior(seed):
    """dict with psi_loc/psi_scale [K, J], phi_loc/phi_scale [N, K-1], phi and psi_mean (point), inv_ev, cell_id"""
    d = fit_dir(seed)
    z = np.load(os.path.join(d, "fit.npz"), allow_pickle=True)
    st = torch.load(os.path.join(d, "checkpoint.pt"), map_location="cpu", weights_only=False)["state"]
    p, c = st["params"], st["constraints"]

    def get(name):
        k = f"AutoNormal.{name}"
        return biject_to(c[k])(p[k]).detach()
    inv = torch.as_tensor(np.asarray(z["inv_ev"], np.int64))
    out = dict(psi_loc=get("locs.psi_logit").float(), psi_scale=get("scales.psi_logit").float(),
               phi_loc=get("locs.phi").double(), phi_scale=get("scales.phi").double(), inv=inv,
               n_atse=len(z["event_id"]), event_id=z["event_id"].astype(str), junction_id=z["junction_id"].astype(str),
               cell_id=z["cell_id"].astype(str), phi=z["phi"].astype(np.float64), psi_mean=z["psi_mean"].astype(np.float64))
    # the checkpoint is the fit's final state: its psi median must be the saved psi_mean
    med = segment_softmax(out["psi_loc"], inv, out["n_atse"]).numpy()
    assert np.abs(med - out["psi_mean"]).max() < 1e-3, f"seed {seed}: checkpoint psi does not match fit.npz"
    return out


def draw_psi(post, seed=0):
    g = torch.Generator().manual_seed(seed)
    loc, sc = post["psi_loc"], post["psi_scale"]
    return torch.stack([segment_softmax(loc + sc * torch.randn(loc.shape, generator=g), post["inv"], post["n_atse"])
                        for _ in range(S)])                                    # [S, K, J]


def draw_dphi(post, rows1, rows2, seed=0):
    sb = biject_to(constraints.simplex)
    g = torch.Generator().manual_seed(seed)
    rows = np.concatenate([rows1, rows2]); n1 = len(rows1)
    loc, sc = post["phi_loc"][rows], post["phi_scale"][rows]
    K = loc.shape[1] + 1
    out = torch.empty(S, K, dtype=torch.float32)
    for s in range(S):
        ph = sb(loc + sc * torch.randn(loc.shape, generator=g, dtype=loc.dtype))
        out[s] = (ph[:n1].mean(0) - ph[n1:].mean(0)).float()
    return out


def bayesian_fdr(prob):
    lfdr = 1.0 - np.asarray(prob, float)
    o = np.argsort(lfdr)
    run = np.cumsum(lfdr[o]) / np.arange(1, len(lfdr) + 1)
    out = np.empty_like(run)
    out[o] = np.minimum.accumulate(run[::-1])[::-1]
    return np.clip(out, 0, 1)


def coverage(Y, inv, n_atse):
    """cells x ATSE boolean: >= 1 read on the ATSE"""
    import scipy.sparse as sp
    E = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), np.asarray(inv))), shape=(len(inv), n_atse))
    return (sp.csr_matrix(Y) @ E) > 0


def test(post, psi_s, rows1, rows2, cov, draw_seed=0):
    """shipped test on one contrast -> DataFrame over gated ATSEs: atse, tv, prob, q, called"""
    n1 = np.asarray(cov[rows1].sum(0)).ravel(); n2 = np.asarray(cov[rows2].sum(0)).ravel()
    gate = (n1 >= MIN_COV) & (n2 >= MIN_COV)
    dphi = draw_dphi(post, rows1, rows2, seed=draw_seed)
    d = torch.einsum("sk,skj->sj", dphi, psi_s)
    tv = 0.5 * torch.zeros(S, post["n_atse"]).index_add(1, post["inv"], d.abs())
    prob = (tv > DELTA).float().mean(0).numpy()
    idx = np.flatnonzero(gate)
    q = bayesian_fdr(prob[idx])
    return pd.DataFrame(dict(atse=post["event_id"][idx], tv=tv.mean(0).numpy()[idx], prob=prob[idx], q=q,
                             called=q <= ALPHA, ncov1=n1[idx], ncov2=n2[idx]))


def tms_train():
    """(cells, tissue_cell_type, mouse, Y) for the TMS-only training split, in fit row order"""
    import h5py
    P = vlib.paths("TMS")
    cells, _, mouse = vlib.obs_labels(P["train"])
    with h5py.File(P["train"], "r") as f:
        tct = vlib._strings(f["obs"]["tissue_cell_type"]).astype(str)
    return cells.astype(str), tct, mouse.astype(str), vlib.read_counts(P["train"])
