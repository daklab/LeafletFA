"""Over-specification: fix K_true, fit K_model >= K_true. Does a big model recover the FEW real programs
and leave the extras unused (prunable)? Decouples K_true (many real programs = data question) from
K_model (big model = optimization question), which every earlier experiment fused (K_model==K_true).

For each (K_true, K_model): full-batch, 3000 steps, random vs kmeans init. Report:
  recovered_true  : mean over the K_true true programs of best corr to ANY fitted program (rectangular match)
  n_active        : # fitted programs with pop usage (phi.mean(0)) > 0.5/K_model   (the rest are prunable)
  perp            : median effective programs/cell (target = true progs/cell of the K_true sim)
Run: overspec.py <K_true> <seeds> <device>"""
import sys, numpy as np, time
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
import torch, pyro
from leafletfa.minibatch import LeafletFAMinibatch

def rect_recovery(psi_hat, psi_true):
    """Mean over TRUE programs of correlation to its best-matched fitted program (K_true x K_model)."""
    Kt, Km = psi_true.shape[0], psi_hat.shape[0]
    C = np.zeros((Kt, Km))
    for i in range(Kt):
        a = psi_true[i]
        for j in range(Km):
            b = psi_hat[j]
            C[i,j] = 0.0 if (a.std()<1e-9 or b.std()<1e-9) else abs(np.corrcoef(a,b)[0,1])
    ri, ci = linear_sum_assignment(-C)          # best assignment of true->fitted
    return float(np.mean([C[i,j] for i,j in zip(ri,ci)])), ci

def cell_psi(sim):
    junc,clus = sim["junc"].astype(np.float32), sim["clus"].astype(np.float32)
    with np.errstate(invalid="ignore",divide="ignore"):
        psi=np.where(clus>0, junc/np.maximum(clus,1), np.nan)
    cm=np.nanmean(np.where(clus>0,psi,np.nan),axis=0); cm=np.nan_to_num(cm,nan=float(np.nanmean(cm)))
    return np.where(np.isnan(psi),cm[None,:],psi).astype(np.float32)

def fit_eval(sim, K_model, steps, seed, device, init):
    N=sim["junc"].shape[0]
    pinit=None
    if init=="kmeans":
        km=MiniBatchKMeans(n_clusters=K_model,random_state=seed,n_init=3,batch_size=4096).fit(cell_psi(sim))
        pinit=np.clip(km.cluster_centers_.astype(np.float32),1e-4,1-1e-4)
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    m=LeafletFAMinibatch(sim["junc"],sim["clus"],K=K_model,device=device,likelihood="betabinomial",psi_init=pinit)
    m._B=N; m.guide=m._make_guide(N)
    from pyro.infer import SVI,Trace_ELBO; from pyro.optim import ClippedAdam
    svi=SVI(m.model,m.guide,ClippedAdam({"lr":0.05,"lrd":0.1**(1/steps),"clip_norm":10.0}),Trace_ELBO())
    t0=time.time(); best=m._param_snapshot()
    for s in range(steps):
        loss=svi.step()
        if not np.isfinite(loss): m._restore_snapshot(best); break
        if s%25==0: best=m._param_snapshot()
    psi=m.psi_mean().cpu().numpy(); phi=m.phi_mean().cpu().numpy()
    rec,_=rect_recovery(psi, sim["psi"])
    usage=phi.mean(0); n_active=int((usage > 0.5/K_model).sum())
    p=np.clip(phi,1e-12,None); perp=float(np.median(np.exp(-(p*np.log(p)).sum(1))))
    return dict(recovered_true=rec, n_active=n_active, perplexity=perp, seconds=round(time.time()-t0,1))

K_true=int(sys.argv[1]) if len(sys.argv)>1 else 10
seeds=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1"])]
device=sys.argv[3] if len(sys.argv)>3 else "cpu"
N=40000; J=max(400,100*K_true); steps=3000
K_models=sorted(set([K_true, 2*K_true, 50]))
import pandas as pd; rows=[]
for seed in seeds:
    sim=simulate(N=N,J=J,K=K_true,seed=seed,overdisp_conc=10.0)
    tp=np.clip(sim["phi"],1e-12,None); true_perp=float(np.median(np.exp(-(tp*np.log(tp)).sum(1))))
    print(f"\n===== K_true={K_true} seed {seed}: J={J} true_progs/cell={true_perp:.1f} =====",flush=True)
    for Km in K_models:
        for init in ("random","kmeans"):
            r=fit_eval(sim,Km,steps,seed,device,init); r.update(dict(K_true=K_true,K_model=Km,init=init,seed=seed))
            rows.append(r)
            print(f"  K_model={Km:>3} init={init:>7}  recovered_true={r['recovered_true']:.3f}  "
                  f"n_active={r['n_active']:>3}/{Km}  perp={r['perplexity']:.1f}  {r['seconds']}s",flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/overspec_Ktrue{K_true}.csv"
df.to_csv(out,index=False)
print(f"\n===== K_true={K_true}: recovered_true by (K_model,init) =====",flush=True)
print(df.pivot_table(index="K_model",columns="init",values="recovered_true",aggfunc="mean").to_string(),flush=True)
print(df.pivot_table(index="K_model",columns="init",values="n_active",aggfunc="mean").to_string(),flush=True)
print(f"wrote {out}",flush=True)
