"""Reproduce and diagnose the non-finite E-step restarts of job 22372199 (STAGE5 section 170) on a
subset of TMS TEST cells, then show the fix.

    python diag_estep_nan.py --n-cells 2000 --out <dir>

Same dictionary and settings as 22372199: TMS70 K20 seed 0 (psi_mean, pi = trace_pi[-1], delta = 20,
per-ATSE concentration), E-step 3,000 steps, lr 0.1, gamma 0.1, batch 8,192 (= all cells here),
max 40 restarts, no early stopping. Arms:
  dense32   the original code path, timing only (--dense-steps steps)
  sparse32  same numerics as dense32 (verified equal) -> does the failure reproduce?
  lr025     sparse32 at lr 0.025 (the training rate): is the rate the cause?
  sparse64  the fix: phi site in float64
At every non-finite step the cells whose q(phi) sample sits on the simplex edge are counted.
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from run_leafletfa_baseline import read_cells, read_counts, read_var, log, peak_gb  # noqa: E402

D = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
TEST = f"{D}/MODEL_INPUT/20260526/test_30_70_TMS_ONLY_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu"
REF = f"{D}/ATLAS_DM/runs_TMS70/dm_simplex_K20_seed0_b8192/fit.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dense-steps", type=int, default=100)
    ap.add_argument("--arms", default="dense32,sparse32,lr025,sparse64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch, pyro
    from torch.distributions import biject_to, constraints
    from leafletfa.minibatch import LeafletFAMinibatch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))

    z = np.load(REF, allow_pickle=True)
    junc_id, ev = read_var(TEST)
    assert np.array_equal(np.asarray(z["event_id"]).astype(str)[z["inv_ev"]], ev), "junction order"
    pi = np.asarray(z["trace_pi"][-1], np.float32); pi /= pi.sum()
    psi, conc = np.asarray(z["psi_mean"], np.float32), np.asarray(z["bb_conc_vec"], np.float32)
    cells = read_cells(TEST)
    rows = np.sort(np.random.default_rng(0).choice(len(cells), a.n_cells, replace=False))
    Y = read_counts(TEST, rows)
    reads = np.asarray(Y.sum(1)).ravel()
    log(f"{a.n_cells} TEST cells, nnz {Y.nnz:,}; cells with no reads {int((reads == 0).sum())}; "
        f"reads/cell median {np.median(reads):.0f}; alpha = 20 pi: {int((20 * pi < 1).sum())} of 20 below 1, "
        f"min {20 * pi.min():.3f}")
    summary = dict(n_cells=a.n_cells, zero_read_cells=int((reads == 0).sum()),
                   alpha_below_1=int((20 * pi < 1).sum()), alpha_min=float(20 * pi.min()), arms={})

    def build(arm):
        pyro.clear_param_store()
        kw = dict(conc_min=2.0, device=a.device)
        if arm == "dense32":
            Dn = Y.toarray()
            _, inv = np.unique(ev, return_inverse=True)
            import scipy.sparse as sp
            tot = Y @ sp.csr_matrix((np.ones(len(ev)), (np.arange(len(ev)), inv)))
            clus = np.asarray(tot[:, inv].todense(), np.float32)
            m = LeafletFAMinibatch.dm(Dn, clus, K=20, event_id=ev, **kw)
        else:
            m = LeafletFAMinibatch.dm(Y, None, K=20, event_id=ev, sparse=True,
                                      phi_dtype="float64" if arm == "sparse64" else "float32", **kw)
        m.freeze_globals(psi=psi, pi=pi, alpha_pi=1.0, dir_conc=20.0, bb_conc=conc)
        return m

    for arm in a.arms.split(","):
        m = build(arm)
        steps = a.dense_steps if arm == "dense32" else a.steps
        lr = 0.025 if arm == "lr025" else 0.1
        edge = []

        # count cells on the simplex edge whenever the loss goes non-finite: one q(phi) draw
        def on_nan_probe():
            st = pyro.get_param_store()
            loc, sc = st["AutoNormal.locs.phi"].detach(), st["AutoNormal.scales.phi"].detach()
            g = torch.Generator().manual_seed(len(edge))
            x = loc + sc * torch.randn(loc.shape, generator=g, dtype=loc.dtype).to(loc.device)
            ph = biject_to(constraints.simplex)(x)
            edge.append(dict(n_cells_phi_exact_zero=int((ph == 0).any(-1).sum()),
                             min_phi=float(ph.min()), max_abs_loc=float(loc.abs().max()),
                             max_scale=float(sc.max())))

        orig = m._restore_snapshot
        def restore(snap, _orig=orig):
            on_nan_probe(); _orig(snap)
        m._restore_snapshot = restore
        t0 = time.time()
        phi = m.infer_phi(num_steps=steps, subsample_size=8192, lr=lr, gamma=0.1, seed=0,
                          early_stop=False, max_nan_restarts=40, verbose=True, log_every=250)
        dt = time.time() - t0
        # where q(phi) ended up, regardless of restarts
        st = pyro.get_param_store()
        loc = st["AutoNormal.locs.phi"].detach().cpu()
        res = dict(steps=steps, lr=lr, n_nan_restarts=int(m.n_nan_restarts),
                   nan_steps=list(map(int, m.nan_steps)), s_per_step=round(dt / steps, 4),
                   minutes=round(dt / 60, 2), finite_phi=bool(np.isfinite(phi).all()),
                   final_max_abs_loc=float(loc.abs().max()), edge_at_nan=edge,
                   loss_last=float(np.nanmean(m.losses[-100:])), peak_rss_gb=round(peak_gb(), 2))
        if arm != "dense32":
            np.save(os.path.join(a.out, f"phi_{arm}.npy"), phi)
        summary["arms"][arm] = res
        log(f"{arm}: {res['n_nan_restarts']} restarts at {res['nan_steps'][:10]}, "
            f"{res['s_per_step']} s/step, edge cells at NaN {[e['n_cells_phi_exact_zero'] for e in edge][:10]}")
        with open(os.path.join(a.out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        del m
    if "sparse32" in summary["arms"] and "sparse64" in summary["arms"]:
        p32, p64 = np.load(os.path.join(a.out, "phi_sparse32.npy")), np.load(os.path.join(a.out, "phi_sparse64.npy"))
        summary["phi_32_vs_64"] = dict(mean_abs_diff=float(np.abs(p32 - p64).mean()),
                                       argmax_agree=float((p32.argmax(1) == p64.argmax(1)).mean()))
        with open(os.path.join(a.out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
    log(json.dumps(summary.get("phi_32_vs_64", {})))


if __name__ == "__main__":
    main()
