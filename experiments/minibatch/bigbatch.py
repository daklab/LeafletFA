"""Batch-ceiling at LARGER batches: how big must B be to recover r~0.99 at K=20/50?
The grid stopped at B=8192 (ceiling 0.89 at K=20). Here B in {8192,16384,24576,32768}, kmeans init
(removes the init confound; the atlas uses waypoints), 1200 epochs (grid: ceiling reached by ~1200).
Finds the batch the atlas retrain should use. Run: bigbatch.py <K> <seeds> <device>"""
import sys, numpy as np, time
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate, _match, _beta
from sklearn.cluster import MiniBatchKMeans
import torch, pyro
from leafletfa.minibatch import LeafletFAMinibatch

def cell_psi(sim):
    junc,clus=sim["junc"].astype(np.float32),sim["clus"].astype(np.float32)
    with np.errstate(invalid="ignore",divide="ignore"):
        psi=np.where(clus>0,junc/np.maximum(clus,1),np.nan)
    cm=np.nanmean(np.where(clus>0,psi,np.nan),axis=0); cm=np.nan_to_num(cm,nan=float(np.nanmean(cm)))
    return np.where(np.isnan(psi),cm[None,:],psi).astype(np.float32)

def run(sim,K,B,epochs,seed,device,pinit):
    N=sim["junc"].shape[0]; steps=max(1,round(epochs*N/B))
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    m=LeafletFAMinibatch(sim["junc"],sim["clus"],K=K,device=device,likelihood="betabinomial",psi_init=pinit)
    m._B=(N if B>=N else B); m.guide=m._make_guide(m._B)
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
    return dict(recovery_r=r,beta_slope=slope,steps=steps,seconds=round(time.time()-t0,1))

K=int(sys.argv[1]) if len(sys.argv)>1 else 20
seeds=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1"])]
device=sys.argv[3] if len(sys.argv)>3 else "cpu"
init_mode=sys.argv[4] if len(sys.argv)>4 else "kmeans"
N=40000; J=max(400,100*K); epochs=1200
Bs=[8192,16384,24576,32768,N]
import pandas as pd; rows=[]
for seed in seeds:
    sim=simulate(N=N,J=J,K=K,seed=seed,overdisp_conc=10.0)
    if init_mode=="kmeans":
        km=MiniBatchKMeans(n_clusters=K,random_state=seed,n_init=3,batch_size=4096).fit(cell_psi(sim))
        pinit=np.clip(km.cluster_centers_.astype(np.float32),1e-4,1-1e-4)
    else:
        pinit=None
    print(f"\n===== K={K} seed {seed} ({init_mode} init) J={J} =====",flush=True)
    for B in Bs:
        r=run(sim,K,B,epochs,seed,device,pinit); r.update(dict(K=K,batch=("full" if B>=N else B),seed=seed))
        rows.append(r)
        print(f"  B={str(r['batch']):>5}  r={r['recovery_r']:.3f}  slope={r['beta_slope']:.3f}  "
              f"steps={r['steps']}  {r['seconds']}s",flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/bigbatch_{init_mode}_K{K}.csv"; df.to_csv(out,index=False)
print(f"\n===== K={K} recovery/slope by batch =====",flush=True)
print(df.groupby("batch")[["recovery_r","beta_slope","steps"]].mean().to_string(),flush=True)
print(f"wrote {out}",flush=True)
