"""Mechanism test: does the Beta-Binomial failure at K=50 come from a runaway LEARNED dispersion?

At K=50 (random init) Beta-Binomial fails (r~0.14, no factor commits) while Binomial recovers (r~0.94).
Hypothesis: the learned bb_conc runs to high overdispersion, flattening the likelihood so no gradient
pushes factors apart. Test at K=50 full-batch:
  bb_learned  : learn bb_conc (baseline failure) -- REPORT the learned value (collapsed?)
  bb_fix10    : hold bb_conc at the TRUE value (overdisp_conc=10)
  bb_fix20    : hold at the prior mean (Gamma(2,0.1) mean = 20)
  bb_fix100   : hold high (near-Binomial, little overdispersion)
  binomial    : the working reference
If fixing bb_conc rescues recovery, the failure is a dispersion-collapse optimization pathology, and the
fix for the atlas is to constrain bb_conc (or use structured init / Binomial) at high K.
Run: mechanism_bbconc.py <K> <seeds> <device>"""
import sys, numpy as np, time
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate, _match, _beta
import torch, pyro
from leafletfa.minibatch import LeafletFAMinibatch

def fit_eval(sim, K, steps, seed, device, likelihood, fix_bb_conc):
    N=sim["junc"].shape[0]
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    m=LeafletFAMinibatch(sim["junc"],sim["clus"],K=K,device=device,
                         likelihood=likelihood,fix_bb_conc=fix_bb_conc)
    m._B=N; m.guide=m._make_guide(N)
    from pyro.infer import SVI,Trace_ELBO; from pyro.optim import ClippedAdam
    svi=SVI(m.model,m.guide,ClippedAdam({"lr":0.05,"lrd":0.1**(1/steps),"clip_norm":10.0}),Trace_ELBO())
    t0=time.time(); best=m._param_snapshot()
    for s in range(steps):
        loss=svi.step()
        if not np.isfinite(loss): m._restore_snapshot(best); break
        if s%25==0: best=m._param_snapshot()
    psi=m.psi_mean().cpu().numpy(); phi=m.phi_mean().cpu().numpy()
    r,order=_match(psi,sim["psi"]); aligned=np.zeros_like(sim["psi"])
    for i,j in enumerate(order): aligned[j]=psi[i]
    bh,bt=_beta(aligned).ravel(),_beta(sim["psi"]).ravel()
    slope=float(np.polyfit(bt,bh,1)[0]) if bt.std()>1e-12 else np.nan
    p=np.clip(phi,1e-12,None); perp=float(np.median(np.exp(-(p*np.log(p)).sum(1))))
    g=m.globals_(); learned_conc=float(g["bb_conc"]) if "bb_conc" in g else np.nan
    return dict(recovery_r=r,beta_slope=slope,perplexity=perp,learned_bb_conc=learned_conc,
                seconds=round(time.time()-t0,1))

K=int(sys.argv[1]) if len(sys.argv)>1 else 50
seeds=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1"])]
device=sys.argv[3] if len(sys.argv)>3 else "cpu"
N=40000; J=max(400,100*K); steps=3000
arms=[("bb_learned","betabinomial",None),("bb_fix10","betabinomial",10.0),
      ("bb_fix20","betabinomial",20.0),("bb_fix100","betabinomial",100.0),
      ("binomial","binomial",None)]
import pandas as pd; rows=[]
for seed in seeds:
    sim=simulate(N=N,J=J,K=K,seed=seed,overdisp_conc=10.0)
    print(f"\n===== K={K} seed {seed}: true psi={sim['psi'].mean():.4f} J={J} true_disp=10 =====",flush=True)
    for name,lik,fx in arms:
        r=fit_eval(sim,K,steps,seed,device,lik,fx); r.update(dict(K=K,arm=name,seed=seed))
        rows.append(r)
        lc=f"learned_conc={r['learned_bb_conc']:.2f}" if name=="bb_learned" else ""
        print(f"  {name:>10}  r={r['recovery_r']:.3f}  slope={r['beta_slope']:.3f}  "
              f"perp={r['perplexity']:.1f}/{K}  {lc}  {r['seconds']}s",flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/mechanism_bbconc_K{K}.csv"
df.to_csv(out,index=False)
print(f"\n===== K={K} mean over seeds =====",flush=True)
print(df.groupby("arm")[["recovery_r","beta_slope","perplexity","learned_bb_conc"]].mean().to_string(),flush=True)
print(f"wrote {out}",flush=True)
