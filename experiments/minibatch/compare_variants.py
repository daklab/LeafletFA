"""Compare candidate fixes for the psi-mislocation / non-convergence failure, on simulated data
where the truth is known.

THE FAILURE BEING FIXED (measured on the 2026-07-18 atlas grid, all 39 models):
  psi initialises correctly at the coverage-aware mean (~0.305), jumps to ~0.71 inside pass 1, and
  stays at 0.63-0.72 for all 8 passes -- at every K, under waypoint AND random init. The likelihood
  prefers psi ~ 0.31 by ~1.09 nats/observed-entry (~1e9 nats over the training set) while the psi
  prior can supply at most ~1e7 nats of counterforce, so the objective wants low psi and the
  optimiser never gets there.

VARIANTS
  chunked_baseline   the shipped approach: rebuild model+guide per chunk, AutoDiagonalNormal,
                     learned a/b hyperprior. Expected to REPRODUCE the bug.
  plate_fixed_ab     pyro.plate + AutoNormal, a,b ~ Gamma(1,1) as METHODS.md specifies.
  plate_learned_ab   as above but WITH the learned a/b hyperprior level -> isolates whether that
                     level still matters once the optimiser is fixed.
  plate_junc_prior   as plate_fixed_ab but with per-junction a_j, b_j.
  full_batch         no subsampling (subsample_size = N). The reference ceiling.

SANITY CHECKS (the intuitions accumulated while debugging this)
  psi_location    |fitted psi mean - TRUE psi mean|. The bug: 0.67 vs 0.29. Must be ~0.
  psi_recovery    Hungarian-matched mean corr(psi_hat_k, psi_true_k). Is the structure learned?
  beta_slope      regress the fitted effect size on the true one. Scale-INVARIANT metrics (Pearson
                  r, cosine, argmax) cannot see effect-size attenuation: shrink every factor toward
                  a common per-junction mean and they all stay ~1 while beta collapses. Slope ~1 is
                  the only thing that certifies the DS effect sizes. (audit plan, section 2)
  psi_contrast    psi.std(axis=0).mean() -- do the K programs actually differ?
  phi_perplexity  median effective programs per cell. Near K = cells hedge = diffuse phi, which
                  dilutes the likelihood gradient on psi by ~K and lets the prior win.
  phi_recovery    Hungarian-matched corr on the cell loadings.
"""

from __future__ import annotations
import sys, time, json
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")


def simulate(N=3000, J=400, K=5, seed=0, coverage=0.135, mean_reads=12, psi_a=0.30, psi_b=0.72):
    """Simulate with the real atlas's characteristics.

    The mouse training data has ~13.5% coverage, a read-weighted splicing rate of 0.295, and among
    COVERED entries 58.7% of PSI values are exactly 0 and 19.8% exactly 1. Beta(0.30, 0.72) gives a
    similarly U-shaped psi with mass at 0 -- the regime the shipped model fails in.
    """
    rng = np.random.default_rng(seed)
    psi = rng.beta(psi_a, psi_b, size=(K, J)).astype(np.float32)
    phi = rng.dirichlet(np.ones(K) * 0.3, size=N).astype(np.float32)
    mu = np.clip(phi @ psi, 1e-6, 1 - 1e-6)
    covered = rng.random((N, J)) < coverage
    clus = (covered * (1 + rng.poisson(mean_reads, size=(N, J)))).astype(np.int64)
    junc = rng.binomial(clus, mu).astype(np.int64)
    return dict(junc=junc, clus=clus, psi=psi, phi=phi,
                observed_rate=float(junc.sum() / max(clus.sum(), 1)))


def _match(A, B):
    """Hungarian-match rows of A to rows of B by |corr|; return (mean corr, column order)."""
    K = A.shape[0]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            a, b = A[i], B[j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                C[i, j] = 0.0
            else:
                C[i, j] = np.corrcoef(a, b)[0, 1]
    r, c = linear_sum_assignment(-np.abs(C))
    return float(np.mean([C[i, j] for i, j in zip(r, c)])), c


def _beta(psi):
    """DS effect size: psi_kj - mean over the OTHER programs at junction j."""
    K = psi.shape[0]
    return psi - (psi.sum(0, keepdims=True) - psi) / max(K - 1, 1)


def evaluate(psi_hat, phi_hat, sim, elapsed, name, extra=None):
    psi_t, phi_t = sim["psi"], sim["phi"]
    r_psi, order = _match(psi_hat, psi_t)
    # align fitted -> true ordering before any scale-sensitive comparison
    inv = np.argsort(order)
    psi_al = psi_hat[np.argsort(np.argsort(order))] if False else psi_hat
    # build alignment explicitly: order[i] = true index matched to fitted i
    aligned = np.zeros_like(psi_t)
    for i, j in enumerate(order):
        aligned[j] = psi_hat[i]
    b_hat, b_true = _beta(aligned).ravel(), _beta(psi_t).ravel()
    slope = float(np.polyfit(b_true, b_hat, 1)[0]) if b_true.std() > 1e-12 else np.nan
    p = np.clip(phi_hat, 1e-12, None)
    perp = float(np.median(np.exp(-(p * np.log(p)).sum(1))))
    r_phi, _ = _match(phi_hat.T, phi_t.T)
    out = {
        "variant": name,
        "psi_mean_fit": float(psi_hat.mean()),
        "psi_mean_true": float(psi_t.mean()),
        "observed_rate": sim["observed_rate"],
        "psi_location_err": float(abs(psi_hat.mean() - psi_t.mean())),
        "psi_recovery_r": r_psi,
        "beta_slope": slope,
        "psi_contrast": float(psi_hat.std(axis=0).mean()),
        "psi_contrast_true": float(psi_t.std(axis=0).mean()),
        "phi_perplexity": perp,
        "phi_perplexity_max": float(phi_hat.shape[1]),
        "phi_recovery_r": r_phi,
        "seconds": round(elapsed, 1),
    }
    if extra:
        out.update(extra)
    return out


# ── variants ────────────────────────────────────────────────────────────────────
def run_plate(sim, K, *, learned_ab=False, junc_prior=False, full_batch=False,
              steps=1500, batch=512, lr=0.05, seed=0, device="cpu"):
    from leafletfa.minibatch import LeafletFAMinibatch
    N = sim["junc"].shape[0]
    m = LeafletFAMinibatch(sim["junc"], sim["clus"], K=K, device=device,
                           junc_specific_prior=junc_prior, learn_ab_hyperprior=learned_ab)
    t0 = time.time()
    losses = m.fit(num_steps=steps, subsample_size=(N if full_batch else batch),
                   lr=lr, seed=seed, verbose=False)
    dt = time.time() - t0
    g = m.globals_()
    extra = {"final_loss": float(losses[-1]),
             "a": float(np.mean(g.get("a", np.nan))), "b": float(np.mean(g.get("b", np.nan)))}
    extra["prior_mean_ab"] = extra["a"] / (extra["a"] + extra["b"]) if np.isfinite(extra["a"]) else np.nan
    return m.psi_mean().cpu().numpy(), m.phi_mean().cpu().numpy(), dt, extra


def run_chunked_baseline(sim, K, *, chunks=6, passes=8, epochs=60, lr=0.05, seed=0, device="cpu"):
    """The shipped approach: a FRESH LeafletFA (and guide) per chunk, warm-starting psi only."""
    from leafletfa import LeafletFA
    import anndata as ad
    from scipy.sparse import csr_matrix
    import torch
    N = sim["junc"].shape[0]
    idx_all = np.arange(N)
    rng = np.random.default_rng(seed)
    psi_carry = None
    phi_store = np.zeros((N, K), dtype=np.float32)
    t0 = time.time()
    bs = int(np.ceil(N / chunks))
    for p in range(passes):
        rng.shuffle(idx_all)
        for c in range(chunks):
            sel = idx_all[c * bs:(c + 1) * bs]
            if len(sel) == 0:
                continue
            a = ad.AnnData(X=csr_matrix(sim["junc"][sel].astype(np.float32)))
            a.layers["cell_by_junction_matrix"] = csr_matrix(sim["junc"][sel].astype(np.float32))
            a.layers["cell_by_cluster_matrix"] = csr_matrix(sim["clus"][sel].astype(np.float32))
            jr = np.where(sim["clus"][sel] > 0, sim["junc"][sel] / np.maximum(sim["clus"][sel], 1), 0)
            a.layers["junc_ratio"] = csr_matrix(jr.astype(np.float32))
            a.layers["psi_mask"] = csr_matrix((sim["clus"][sel] > 0).astype(np.float32))
            mdl = LeafletFA(adata=a, K=K, waypoints_use=False, num_epochs=epochs,
                            print_epochs=epochs + 1, lr=lr, gamma=0.5, patience=10,
                            min_delta=100, min_delta_rel=1e-5, ELBO_num_particles=1,
                            log_wandb=False, loss_plot=False, output_dir=None,
                            enable_cpu_optimization=False,
                            obs_scale=float(N) / max(len(sel), 1),
                            **({"fixed_psi": None} if psi_carry is None else {}))
            mdl.from_anndata()
            if psi_carry is not None:
                mdl.psi_init = torch.tensor(psi_carry, dtype=torch.float32, device=mdl.device)
            mdl.train(num_initializations=1)
            mdl.get_all_variables()
            psi_carry = np.asarray(mdl.psi.detach().cpu() if hasattr(mdl.psi, "detach") else mdl.psi)
            ph = mdl.assign_post
            phi_store[sel] = np.asarray(ph.detach().cpu() if hasattr(ph, "detach") else ph)
    dt = time.time() - t0
    return psi_carry, phi_store, dt, {"final_loss": np.nan}


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1"])]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    rows = []
    for seed in seeds:
        sim = simulate(K=K, seed=seed)
        print(f"\n===== seed {seed}: true psi mean={sim['psi'].mean():.4f}  "
              f"observed rate={sim['observed_rate']:.4f} =====", flush=True)
        variants = [
            ("plate_fixed_ab",   dict(learned_ab=False, junc_prior=False, full_batch=False)),
            ("plate_learned_ab", dict(learned_ab=True,  junc_prior=False, full_batch=False)),
            ("plate_junc_prior", dict(learned_ab=False, junc_prior=True,  full_batch=False)),
            ("full_batch",       dict(learned_ab=False, junc_prior=False, full_batch=True)),
        ]
        for name, kw in variants:
            try:
                psi_h, phi_h, dt, extra = run_plate(sim, K, steps=steps, seed=seed,
                                                    device=device, **kw)
                r = evaluate(psi_h, phi_h, sim, dt, name, extra)
            except Exception as e:
                r = {"variant": name, "error": str(e)[:200]}
            r["seed"] = seed
            rows.append(r)
            print(f"  {name:<18} " + (r.get("error") or
                  f"psi {r['psi_mean_fit']:.4f} (true {r['psi_mean_true']:.4f}, "
                  f"err {r['psi_location_err']:.4f})  r_psi={r['psi_recovery_r']:.3f}  "
                  f"slope={r['beta_slope']:.3f}  perp={r['phi_perplexity']:.1f}/{K}  "
                  f"{r['seconds']}s"), flush=True)
        try:
            psi_h, phi_h, dt, extra = run_chunked_baseline(sim, K, seed=seed, device=device)
            r = evaluate(psi_h, phi_h, sim, dt, "chunked_baseline", extra)
        except Exception as e:
            r = {"variant": "chunked_baseline", "error": str(e)[:300]}
        r["seed"] = seed
        rows.append(r)
        print(f"  {'chunked_baseline':<18} " + (r.get("error") or
              f"psi {r['psi_mean_fit']:.4f} (true {r['psi_mean_true']:.4f}, "
              f"err {r['psi_location_err']:.4f})  r_psi={r['psi_recovery_r']:.3f}  "
              f"slope={r['beta_slope']:.3f}  perp={r['phi_perplexity']:.1f}/{K}  "
              f"{r['seconds']}s"), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = "/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/variants.csv"
    df.to_csv(out, index=False)
    print("\n\n================ SUMMARY (mean over seeds) ================", flush=True)
    if "psi_location_err" in df:
        agg = df.groupby("variant")[["psi_location_err", "psi_recovery_r", "beta_slope",
                                     "psi_contrast", "phi_perplexity", "seconds"]].mean()
        print(agg.to_string(), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
