#!/usr/bin/env python
"""Does the JUNCTION-SPECIFIC psi prior pull the K programs toward the symmetric saddle?

    psi_kj ~ Beta(a_j, b_j)      with a_j, b_j SHARED ACROSS ALL K FACTORS

a_j/b_j are indexed by JUNCTION only, so this is literally a shrinkage prior pulling the K programs
TOWARD EACH OTHER at every junction -- i.e. toward the symmetric saddle that is the model's main
failure mode (METHODS.md 11.1). A GLOBAL Beta(a, b) has no such per-junction pull.

The paper uses junc_specific_prior=True. It was never tested.

Run on the REAL Figure-2 simulation. Scored against ground truth.
"""
import contextlib, io, sys, warnings
import numpy as np, torch, anndata as ad
warnings.filterwarnings("ignore")
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from leafletfa import LeafletFA

SIM = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/SIMULATION"
sim = ad.read_h5ad(f"{SIM}/sim_heart_K5_binom_eff1.0_seed0.h5ad")
K, E = 5, sys.stderr
tphi = np.asarray(sim.obsm["true_phi"])
ev = np.asarray(sim.uns["atse_ids"], dtype=object); ei = {x: i for i, x in enumerate(ev)}
cev = np.array([ei[x] for x in sim.var["event_id"].astype(str).values])
skp = np.where(sim.var["junction_role"].values == "J3")[0]; ordr = cev[skp]
tp = np.asarray(sim.uns["true_psi_skip"])[:, np.sort(np.unique(ordr))]
isd = np.array([sim.var["is_differential"].values[cev == i][0] for i in range(len(ev))])

def score(psi_hat, phi_hat):
    ph = psi_hat[:, skp][:, np.argsort(ordr)]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            m = np.isfinite(tp[j]) & np.isfinite(ph[i])
            C[i, j] = np.corrcoef(ph[i][m], tp[j][m])[0, 1] if m.sum() > 2 else 0
    r, c = linear_sum_assignment(-np.abs(np.nan_to_num(C)))
    d = ph.max(0) - ph.min(0); ok = np.isfinite(d)
    return (float(np.median([C[i, j] for i, j in zip(r, c)])),
            float((phi_hat[:, r].argmax(1) == tphi[:, c].argmax(1)).mean()),
            float(roc_auc_score(isd[ok], d[ok])))

print("REAL Figure-2 simulation (6,986 cells x 12,804 junctions, K=5)", file=E)
print("Beta-Binomial, lr=0.05, gamma=0.5, random init.  chance: agree 20%, AUROC 0.500\n", file=E)
print(f"{'junc_specific':>14} {'seed':>5} {'psi_r':>7} {'agree':>7} {'AUROC':>7} {'contrast':>9} {'loss':>12}", file=E)
res = {True: [], False: []}
for jsp in (True, False):
    for sd in (0, 1, 2):
        m = LeafletFA(adata=sim, K=K, junc_specific_prior=jsp, waypoints_use=False,
                      input_conc_prior=None, print_epochs=10**6, num_epochs=1200,
                      patience=10**6, min_delta=0, min_delta_rel=1e-6,
                      ELBO_num_particles=5, num_samples=50, lr=0.05, gamma=0.5,
                      log_wandb=False, loss_plot=False)
        with contextlib.redirect_stdout(io.StringIO()):
            m.from_anndata(); m.train(num_initializations=1, seeds=[sd]); m.get_all_variables()
        pr, ag, au = score(np.asarray(m.psi), np.asarray(m.assign_post))
        res[jsp].append((pr, ag, au, m.best_elbo))
        print(f"{str(jsp):>14} {sd:>5} {pr:>+7.3f} {ag:>6.1%} {au:>7.3f} "
              f"{m.health['psi_contrast']:>9.4f} {m.best_elbo:>12.4e}", file=E, flush=True)

t = np.array(res[True]); f = np.array(res[False])
print(f"\n  junction-specific : psi_r {t[:,0].mean():+.3f} +- {t[:,0].std():.3f}   "
      f"agree {t[:,1].mean():.1%}   AUROC {t[:,2].mean():.3f}", file=E)
print(f"  global            : psi_r {f[:,0].mean():+.3f} +- {f[:,0].std():.3f}   "
      f"agree {f[:,1].mean():.1%}   AUROC {f[:,2].mean():.3f}", file=E)
print(f"\n  SEED VARIANCE (sd of psi_r):  junction-specific {t[:,0].std():.3f}   "
      f"global {f[:,0].std():.3f}", file=E)
print(f"  Does the ELBO prefer the better one?  mean loss: "
      f"junc-specific {t[:,3].mean():.4e}   global {f[:,3].mean():.4e}   "
      f"-> ELBO picks {'GLOBAL' if f[:,3].mean() < t[:,3].mean() else 'JUNC-SPECIFIC'}", file=E)
