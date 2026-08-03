"""Simulation figure factorial (K in {10,20,30}, random init, no waypoints).
Three ablation axes, each measured against recovery of known-truth psi:
  batch      : full-batch vs minibatch (B=8192)          -- is minibatch a valid approximation?
  likelihood : Beta-Binomial vs Binomial                 -- does modelling overdispersion matter?
  junc_prior : junction-specific Beta(a_j,b_j) psi prior vs shared scalar a,b
Data: N=40000, overdispersed (overdisp_conc=10), J=100*K, 3 seeds. Random init throughout.
Run: sim_factorial.py <K> <seeds> <device>"""
import sys, numpy as np, time
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate, _match, _beta
import torch, pyro
from leafletfa.minibatch import LeafletFAMinibatch

def run(sim,K,B,seed,device,likelihood,junc_prior,steps=3000):
    N=sim["junc"].shape[0]; full=(B>=N); Beff=N if full else B
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    m=LeafletFAMinibatch(sim["junc"],sim["clus"],K=K,junc_specific_prior=junc_prior,
                         device=device,likelihood=likelihood)
    m._B=Beff; m.guide=m._make_guide(Beff)
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
    return dict(recovery_r=r,beta_slope=slope,perplexity=perp,seconds=round(time.time()-t0,1))

K=int(sys.argv[1]) if len(sys.argv)>1 else 20
seeds=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1","2"])]
device=sys.argv[3] if len(sys.argv)>3 else "cpu"
N=40000; J=max(400,100*K)
batches=[("full",N),("mb8192",8192)]
liks=["betabinomial","binomial"]; jpriors=[True,False]
import itertools, pandas as pd; rows=[]
for seed in seeds:
    sim=simulate(N=N,J=J,K=K,seed=seed,overdisp_conc=10.0)
    print(f"\n===== K={K} seed {seed}: true psi={sim['psi'].mean():.4f} J={J} =====",flush=True)
    for (bname,B),lik,jp in itertools.product(batches,liks,jpriors):
        r=run(sim,K,B,seed,device,lik,jp)
        r.update(dict(K=K,batch=bname,likelihood=lik,junc_prior=jp,seed=seed))
        rows.append(r)
        print(f"  {bname:>6} {lik[:3]} jprior={str(jp):>5}  r={r['recovery_r']:.3f}  "
              f"slope={r['beta_slope']:.3f}  perp={r['perplexity']:.1f}/{K}  {r['seconds']}s",flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/sim_factorial_K{K}.csv"; df.to_csv(out,index=False)
print(f"\n===== K={K} recovery_r by (batch,likelihood,junc_prior), mean over seeds =====",flush=True)
print(df.groupby(["batch","likelihood","junc_prior"])["recovery_r"].mean().to_string(),flush=True)
print(f"wrote {out}",flush=True)
