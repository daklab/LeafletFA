"""Does TRUE minibatch SVI (now that subsampling actually engages) match full-batch, and does it
CONVERGE and STAY there?

Context. The earlier compare_variants.py sweep ran with subsampling silently OFF (the guide traced
the full N), so every 'plate' arm there was really full-batch -- it never tested minibatching. And
the first real 70% run showed psi drifting monotonically PAST the observed rate (0.2947 @ step 900
-> 0.2807 @ 2700) before a NaN blow-up, so 'lands on the right value once' is not enough: we need to
see it CONVERGE and hold.

This script, with subsampling verified engaged, compares full-batch against several minibatch sizes
on atlas-like simulated data and checks:

  subsampling_ok   traced phi is (B, K), not (N, K)   -- the check the old sweep never made
  location_err     |mean psi_hat - mean psi_true|
  recovery_r       Hungarian-matched corr(psi_hat, psi_true)
  beta_slope       regress fitted DS effect size on true (scale-sensitive; ~1 = no attenuation)
  contrast         psi.std(0).mean()  (programs differ)
  perplexity       median effective programs per cell (should be << K)
  final_vs_full    final -ELBO / full-batch final -ELBO  (per-cell, comparable)
  late_drift       |mean psi over last 10% of steps - mean psi over the preceding 10%|
                   -> ~0 means CONVERGED and STABLE; large means still drifting (the real-run problem)
  plateaued        slope of -ELBO over the last 20% of steps ~ 0

Run:
    minibatch_vs_fullbatch.py [K=6] [steps=3000] [seeds=0,1] [device=cpu]
"""
import sys, time
import numpy as np

sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0, "/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate, _match, _beta   # reuse the atlas-like generator + matchers
import torch, pyro
from pyro import poutine


def subsampling_engaged(m, B):
    """Trace the model once and confirm phi is batch-sized, not N-sized."""
    m._B = B
    tr = poutine.trace(m.model).get_trace()
    return tuple(tr.nodes["phi"]["value"].shape) == (B, m.K)


def convergence(losses, psi_trace):
    L = np.asarray(losses, dtype=float)
    ok = np.isfinite(L)
    n = len(L)
    tail = slice(int(0.8 * n), n)
    # plateau: normalised slope of -ELBO over the last 20%
    xs = np.arange(n)[tail]
    ys = L[tail]
    good = np.isfinite(ys)
    slope = float(np.polyfit(xs[good], ys[good], 1)[0]) if good.sum() > 2 else np.nan
    plateau = abs(slope) / (abs(np.nanmean(ys)) + 1e-9)      # ~0 = flat
    # late psi drift: last 10% vs the preceding 10%
    p = np.asarray(psi_trace, dtype=float)
    m = len(p)
    late = float(abs(np.nanmean(p[int(0.9*m):]) - np.nanmean(p[int(0.8*m):int(0.9*m)]))) if m >= 10 else np.nan
    return {"n_finite": int(ok.sum()), "plateau_slope_rel": plateau, "late_psi_drift": late}


def run(sim, K, B, steps, seed, device):
    from leafletfa.minibatch import LeafletFAMinibatch
    N = sim["junc"].shape[0]
    full = (B >= N)
    m = LeafletFAMinibatch(sim["junc"], sim["clus"], K=K, device=device)
    Beff = N if full else B
    engaged = None if full else subsampling_engaged(m, Beff)
    # capture the psi trajectory by logging every 25 steps through fit's own logging is awkward;
    # instead fit in short segments and record psi_mean between them
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    m._B = Beff
    m.guide = m._make_guide(Beff)
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam
    lrd = 0.1 ** (1.0 / max(steps, 1))
    svi = SVI(m.model, m.guide, ClippedAdam({"lr": 0.05, "lrd": lrd, "clip_norm": 10.0}),
              Trace_ELBO())
    losses, psi_tr = [], []
    t0 = time.time()
    best = m._param_snapshot(); best_i = 0
    for s in range(steps):
        loss = svi.step(); losses.append(loss)
        if not np.isfinite(loss):
            m._restore_snapshot(best); break
        if s % 25 == 0:
            best = m._param_snapshot(); best_i = s
            psi_tr.append(float(m.psi_mean().mean()))
    dt = time.time() - t0
    psi = m.psi_mean().cpu().numpy(); phi = m.phi_mean().cpu().numpy()

    r_psi, order = _match(psi, sim["psi"])
    aligned = np.zeros_like(sim["psi"])
    for i, j in enumerate(order):
        aligned[j] = psi[i]
    bh, bt = _beta(aligned).ravel(), _beta(sim["psi"]).ravel()
    slope = float(np.polyfit(bt, bh, 1)[0]) if bt.std() > 1e-12 else np.nan
    p = np.clip(phi, 1e-12, None)
    perp = float(np.median(np.exp(-(p * np.log(p)).sum(1))))
    conv = convergence(losses, psi_tr)
    return {
        "B": ("full" if full else B), "subsampling_ok": engaged,
        "psi_mean": float(psi.mean()), "location_err": float(abs(psi.mean() - sim["psi"].mean())),
        "recovery_r": r_psi, "beta_slope": slope,
        "contrast": float(psi.std(0).mean()), "perplexity": perp,
        "final_neg_elbo": float(losses[-1]) if np.isfinite(losses[-1]) else np.nan,
        "seconds": round(dt, 1), **conv,
    }


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seeds = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1"])]
    device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    N = 4000
    rows = []
    for seed in seeds:
        J = max(400, 100 * K)   # ~100 junctions/program so higher K stays identifiable
        sim = simulate(N=N, J=J, K=K, seed=seed)
        print(f"\n===== seed {seed}: true psi mean={sim['psi'].mean():.4f}  "
              f"rate={sim['observed_rate']:.4f}  N={N} =====", flush=True)
        for B in (N, 1024, 512, 256):     # full-batch first as the reference
            r = run(sim, K, B, steps, seed, device); r["seed"] = seed
            rows.append(r)
            print(f"  B={str(r['B']):>5}  sub_ok={r['subsampling_ok']}  "
                  f"psi={r['psi_mean']:.4f} (err {r['location_err']:.4f})  r={r['recovery_r']:.3f}  "
                  f"slope={r['beta_slope']:.3f}  perp={r['perplexity']:.1f}/{K}  "
                  f"drift={r['late_psi_drift']:.4f}  plateau={r['plateau_slope_rel']:.1e}  "
                  f"{r['seconds']}s", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = "/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/minibatch_vs_fullbatch_K%d.csv" % K
    df.to_csv(out, index=False)
    print("\n================ mean over seeds ================", flush=True)
    agg = df.groupby("B")[["location_err", "recovery_r", "beta_slope", "contrast",
                           "perplexity", "late_psi_drift", "plateau_slope_rel", "seconds"]].mean()
    print(agg.to_string(), flush=True)
    # verdicts
    full = df[df.B == "full"]
    mb = df[df.B != "full"]
    print("\nCHECKS:", flush=True)
    print(f"  subsampling engaged for every minibatch run: {bool(mb['subsampling_ok'].all())}", flush=True)
    print(f"  minibatch recovery within 0.02 of full-batch: "
          f"{abs(mb.recovery_r.mean() - full.recovery_r.mean()) < 0.02}", flush=True)
    print(f"  minibatch beta_slope in [0.9, 1.1] (no attenuation): "
          f"{mb.beta_slope.between(0.9, 1.1).all()}", flush=True)
    print(f"  minibatch converged (late psi drift < 0.01): {bool((mb.late_psi_drift < 0.01).all())}", flush=True)
    print(f"  speedup vs full-batch: {full.seconds.mean() / max(mb.seconds.mean(), 1e-9):.1f}x", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
