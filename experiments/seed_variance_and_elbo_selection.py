#!/usr/bin/env python
"""Two questions that decide whether the 3-restart protocol is worth anything.

 Q1. How much does the SEED matter at the chosen schedule?
     (The same config gave psi_r 0.758 on seed 0 and 0.223 on seed 5973.)

 Q2. Does the ELBO TRACK RECOVERY? The Methods says "fit three times, keep the one with the best
     ELBO". That only works if a better ELBO means a better fit. NOBODY HAS CHECKED. If the ELBO does
     not correlate with psi recovery, best-ELBO selection is worthless and model selection must be
     done on HELD-OUT metrics instead.

Run: python experiments/seed_variance_and_elbo_selection.py
"""
import contextlib, io, sys, warnings
import numpy as np, torch, anndata as ad
warnings.filterwarnings("ignore")
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
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

SEEDS = [0, 1, 2, 3, 4, 5]
LR, GAMMA = 0.05, 0.5

for use_wp in (False,):            # waypoints need adata.varm keys; random init is the hard case
    print(f"\nlr={LR} gamma={GAMMA}, Beta-Binomial, random init — {len(SEEDS)} seeds", file=E)
    print(f"{'seed':>5} {'ELBO (loss)':>14} {'psi_r':>7} {'agree':>7} {'AUROC':>7} {'contrast':>9}", file=E)
    res = []
    for sd in SEEDS:
        m = LeafletFA(adata=sim, K=K, junc_specific_prior=True, waypoints_use=False,
                      input_conc_prior=None,          # Beta-Binomial (the right choice)
                      print_epochs=10**6, num_epochs=1200, patience=10**6,
                      min_delta=0, min_delta_rel=1e-6, ELBO_num_particles=5, num_samples=50,
                      lr=LR, gamma=GAMMA, log_wandb=False, loss_plot=False)
        with contextlib.redirect_stdout(io.StringIO()):
            m.from_anndata(); m.train(num_initializations=1, seeds=[sd]); m.get_all_variables()
        pr, ag, au = score(np.asarray(m.psi), np.asarray(m.assign_post))
        res.append((m.best_elbo, pr, ag, au, m.health["psi_contrast"]))
        print(f"{sd:>5} {m.best_elbo:>14.5e} {pr:>+7.3f} {ag:>6.1%} {au:>7.3f} {res[-1][4]:>9.4f}",
              file=E, flush=True)

    r = np.array(res)
    elbo, psi_r = r[:, 0], r[:, 1]
    print(f"\n  Q1 SEED VARIANCE: psi_r ranges {psi_r.min():.3f} .. {psi_r.max():.3f} "
          f"(mean {psi_r.mean():.3f}, sd {psi_r.std():.3f})", file=E)

    # best ELBO = LOWEST loss (LeafletFA stores the negative ELBO / loss)
    pick = int(np.argmin(elbo))
    rho, pval = spearmanr(elbo, psi_r)
    print(f"\n  Q2 DOES THE ELBO TRACK RECOVERY?", file=E)
    print(f"     Spearman(loss, psi_r) = {rho:+.3f}  (p={pval:.3f})", file=E)
    print(f"     want STRONGLY NEGATIVE: lower loss  ->  better recovery", file=E)
    print(f"     best-ELBO seed = {SEEDS[pick]}  ->  psi_r = {psi_r[pick]:.3f}", file=E)
    print(f"     best POSSIBLE  ->  psi_r = {psi_r.max():.3f}   (oracle)", file=E)
    print(f"     mean over seeds ->  psi_r = {psi_r.mean():.3f}  (what 1 random seed gets you)", file=E)
    if psi_r[pick] >= psi_r.max() - 0.02:
        print(f"\n     VERDICT: best-ELBO selection PICKS THE BEST FIT. The 3-restart protocol works.", file=E)
    elif psi_r[pick] > psi_r.mean():
        print(f"\n     VERDICT: best-ELBO selection HELPS but is not optimal. Prefer held-out selection.", file=E)
    else:
        print(f"\n     VERDICT: *** best-ELBO selection DOES NOT WORK *** — the ELBO does not track", file=E)
        print(f"              recovery. MODEL SELECTION MUST USE HELD-OUT METRICS.", file=E)
