"""Does STRUCTURED init rescue high-K recovery? (the plan's section 9: 'init is load-bearing at high K')

The sim uses RANDOM init and fails at K=50 (r~0.14 even full-batch, cells spread over all 50 programs).
The atlas uses WAYPOINT init precisely to avoid this. Here we test a faithful proxy: k-means on the
per-cell PSI matrix gives K distinct, data-supported anchors (the essence of waypoints), used as psi_init.
Compare random vs structured init at K in {20,50}, full-batch, matched steps. If structured init recovers
where random fails, that is direct evidence waypoints are load-bearing at high K -- and the retrain fix.

Run: init_highK.py <K> <seeds> <device>"""
import sys, numpy as np
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate, _match, _beta
from sklearn.cluster import MiniBatchKMeans
import torch, pyro, time
from leafletfa.minibatch import LeafletFAMinibatch

def cell_psi(sim):
    junc, clus = sim["junc"].astype(np.float32), sim["clus"].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.where(clus>0, junc/np.maximum(clus,1), np.nan)
    colmean = np.nanmean(np.where(clus>0, psi, np.nan), axis=0)   # per-junction mean over covered
    colmean = np.nan_to_num(colmean, nan=float(np.nanmean(colmean)))
    filled = np.where(np.isnan(psi), colmean[None,:], psi)
    return filled.astype(np.float32)

def kmeans_psi_init(sim, K, seed):
    X = cell_psi(sim)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3, batch_size=4096).fit(X)
    return np.clip(km.cluster_centers_.astype(np.float32), 1e-4, 1-1e-4)   # (K,J)

def fit_eval(sim, K, steps, seed, device, psi_init):
    N = sim["junc"].shape[0]
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    m = LeafletFAMinibatch(sim["junc"], sim["clus"], K=K, device=device,
                           likelihood="betabinomial", psi_init=psi_init)
    m._B = N; m.guide = m._make_guide(N)
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam
    svi = SVI(m.model, m.guide, ClippedAdam({"lr":0.05,"lrd":0.1**(1/steps),"clip_norm":10.0}), Trace_ELBO())
    t0=time.time(); best=m._param_snapshot()
    for s in range(steps):
        loss=svi.step()
        if not np.isfinite(loss): m._restore_snapshot(best); break
        if s%25==0: best=m._param_snapshot()
    psi=m.psi_mean().cpu().numpy(); phi=m.phi_mean().cpu().numpy()
    r,order=_match(psi,sim["psi"])
    aligned=np.zeros_like(sim["psi"])
    for i,j in enumerate(order): aligned[j]=psi[i]
    bh,bt=_beta(aligned).ravel(),_beta(sim["psi"]).ravel()
    slope=float(np.polyfit(bt,bh,1)[0]) if bt.std()>1e-12 else np.nan
    p=np.clip(phi,1e-12,None); perp=float(np.median(np.exp(-(p*np.log(p)).sum(1))))
    return dict(recovery_r=r, beta_slope=slope, perplexity=perp, seconds=round(time.time()-t0,1))

K=int(sys.argv[1]) if len(sys.argv)>1 else 50
seeds=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1"])]
device=sys.argv[3] if len(sys.argv)>3 else "cpu"
N=40000; J=max(400,100*K); steps=3000
import pandas as pd; rows=[]
for seed in seeds:
    sim=simulate(N=N,J=J,K=K,seed=seed,overdisp_conc=10.0)
    print(f"\n===== K={K} seed {seed}: true psi={sim['psi'].mean():.4f} J={J} =====",flush=True)
    pk=kmeans_psi_init(sim,K,seed)
    for name,pinit in [("random",None),("kmeans",pk)]:
        r=fit_eval(sim,K,steps,seed,device,pinit); r.update(dict(K=K,init=name,seed=seed,steps=steps))
        rows.append(r)
        print(f"  init={name:>7}  r={r['recovery_r']:.3f}  slope={r['beta_slope']:.3f}  "
              f"perp={r['perplexity']:.1f}/{K}  {r['seconds']}s",flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/init_highK_K{K}.csv"
df.to_csv(out,index=False)
print(f"\n===== K={K} recovery by init (mean over seeds) =====",flush=True)
print(df.groupby("init")[["recovery_r","beta_slope","perplexity"]].mean().to_string(),flush=True)
print(f"wrote {out}",flush=True)
