"""Run the plate-based minibatch model on the REAL 70% atlas training data.

The decisive check is ψ LOCATION: the shipped chunked models land at 0.670-0.720 against an
observed read-weighted splicing rate of 0.2951. If the plate model lands near 0.295, the
optimisation failure is fixed. Everything else (recovery, contrast, perplexity) is secondary --
without the location being right, nothing downstream is usable.

Usage:
    run_real_70pct.py <out_dir> [K=20] [steps=3000] [batch=2048] [lr=0.05]
                      [n_cells=0 (all)] [n_junctions=0 (all)] [learned_ab=0] [junc_prior=0]
"""
import sys, os, json, time
import numpy as np
import h5py
import scipy.sparse as sp

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
TRAIN = ("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/"
         "MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad")


def load(n_cells=0, n_junctions=0):
    def rows(g, n):
        ip = g["indptr"][:n + 1] if n else g["indptr"][:]
        a, b = int(ip[0]), int(ip[-1])
        return sp.csr_matrix((g["data"][a:b], g["indices"][a:b], ip - ip[0]),
                             shape=((n if n else len(ip) - 1), int(g.attrs["shape"][1])))
    with h5py.File(TRAIN, "r") as h:
        N_all = h["obs/_index"].shape[0]
        n = min(n_cells, N_all) if n_cells else N_all
        junc = rows(h["layers/cell_by_junction_matrix"], n).toarray().astype(np.float32)
        clus = rows(h["layers/cell_by_cluster_matrix"], n).toarray().astype(np.float32)
    if n_junctions and n_junctions < junc.shape[1]:
        # keep the best-covered junctions so the subset is informative, not arbitrary
        keep = np.argsort(-(clus > 0).sum(0))[:n_junctions]
        keep.sort()
        junc, clus = junc[:, keep], clus[:, keep]
    return junc, clus


def main():
    out_dir = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    n_cells = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    n_junc = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    learned_ab = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
    junc_prior = bool(int(sys.argv[9])) if len(sys.argv) > 9 else False
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading real 70% training data (cells={n_cells or 'all'}, junctions={n_junc or 'all'})",
          flush=True)
    junc, clus = load(n_cells, n_junc)
    cov = clus > 0
    obs_rate = float(junc.sum() / max(clus.sum(), 1))
    psi_obs = float((junc[cov] / np.maximum(clus[cov], 1)).mean())
    print(f"  shape={junc.shape}  coverage={cov.mean()*100:.2f}%", flush=True)
    print(f"  READ-WEIGHTED observed rate = {obs_rate:.4f}   <-- psi must land HERE", flush=True)
    print(f"  unweighted mean PSI (covered) = {psi_obs:.4f}", flush=True)
    print(f"  frac of covered PSI exactly 0 = "
          f"{float((junc[cov] == 0).mean())*100:.1f}%", flush=True)

    from leafletfa.minibatch import LeafletFAMinibatch
    m = LeafletFAMinibatch(junc, clus, K=K, learn_ab_hyperprior=learned_ab,
                           junc_specific_prior=junc_prior)
    print(f"\ndevice={m.device}  K={K}  steps={steps}  batch={batch}  lr={lr}  "
          f"learned_ab={learned_ab}  junc_prior={junc_prior}", flush=True)
    t0 = time.time()
    losses = m.fit(num_steps=steps, subsample_size=batch, lr=lr, seed=0, log_every=100)
    dt = time.time() - t0

    psi = m.psi_mean().cpu().numpy()
    phi = m.phi_mean().cpu().numpy()
    p = np.clip(phi, 1e-12, None)
    perp = float(np.median(np.exp(-(p * np.log(p)).sum(1))))
    g = m.globals_()
    res = {
        "K": K, "steps": steps, "batch": batch, "lr": lr,
        "n_cells": int(junc.shape[0]), "n_junctions": int(junc.shape[1]),
        "observed_rate": obs_rate,
        "psi_mean": float(psi.mean()), "psi_median": float(np.median(psi)),
        "psi_p1": float(np.percentile(psi, 1)), "psi_p99": float(np.percentile(psi, 99)),
        "psi_frac_below_005": float((psi < 0.05).mean()),
        "psi_frac_above_095": float((psi > 0.95).mean()),
        "psi_location_err": float(abs(psi.mean() - obs_rate)),
        "psi_contrast": float(psi.std(axis=0).mean()),
        "pred_mean": float((phi @ psi).mean()),
        "phi_perplexity": perp, "K_max_perplexity": K,
        "final_loss": float(losses[-1]), "seconds": round(dt, 1),
        "a": float(np.mean(g.get("a", np.nan))), "b": float(np.mean(g.get("b", np.nan))),
        "bb_conc": float(np.mean(g.get("bb_conc", np.nan))),
        "dir_conc": float(np.mean(g.get("dir_conc", np.nan))),
    }
    np.savez_compressed(os.path.join(out_dir, "fit.npz"), psi=psi, phi=phi,
                        losses=np.asarray(losses), **{k: v for k, v in g.items()})
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("\n================ REAL-DATA RESULT ================", flush=True)
    for k in ("observed_rate", "psi_mean", "psi_median", "psi_location_err", "pred_mean",
              "psi_contrast", "phi_perplexity", "psi_frac_below_005", "final_loss", "seconds"):
        print(f"  {k:<22} {res[k]}", flush=True)
    verdict = ("FIXED -- psi is at the observed rate" if res["psi_location_err"] < 0.05 else
               "STILL MISLOCATED" if res["psi_location_err"] > 0.15 else "PARTIAL")
    print(f"\n  VERDICT: {verdict}   (shipped chunked models: psi 0.670 vs rate 0.295, err 0.375)",
          flush=True)
    print(f"\nwrote {out_dir}/result.json", flush=True)


if __name__ == "__main__":
    main()
