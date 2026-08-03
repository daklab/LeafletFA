"""Simulation figure: ONE dataset with K_true=10 real programs. Fit K_model in {10,20,30} and ask:
do the models recover the 10 real programs, or hallucinate extra ones? Does pruning (pi>0.01 +
renormalize) trim the over-specified models back toward 10?
Ablation axes: full-batch vs minibatch B=8192, Beta-Binomial vs Binomial, junction-specific vs global prior.
Random init, 3000 steps, N=40000, overdispersed (conc=10), 3 seeds.
Run: sim_recover10.py <K_model> <seeds> <device>"""
import sys, numpy as np, time, itertools
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate, _beta
from scipy.optimize import linear_sum_assignment
import torch, pyro
from leafletfa.minibatch import LeafletFAMinibatch

K_TRUE=10; PRUNE_TAU=0.01

def rect_match(psi_hat, psi_true):
    """Match each TRUE program to its best fitted program; return (mean corr, matched fitted indices)."""
    Kt,Km=psi_true.shape[0],psi_hat.shape[0]; C=np.zeros((Kt,Km))
    for i in range(Kt):
        a=psi_true[i]
        for j in range(Km):
            b=psi_hat[j]; C[i,j]=0.0 if (a.std()<1e-9 or b.std()<1e-9) else abs(np.corrcoef(a,b)[0,1])
    ri,ci=linear_sum_assignment(-C)
    return float(np.mean([C[i,j] for i,j in zip(ri,ci)])), ci

def run(sim,K_model,B,seed,device,likelihood,junc_prior,steps=5000):
    N=sim["junc"].shape[0]; full=(B>=N); Beff=N if full else B
    m=LeafletFAMinibatch(sim["junc"],sim["clus"],K=K_model,junc_specific_prior=junc_prior,
                         device=device,likelihood=likelihood)
    t0=time.time()
    # real trainer path: high step CAP + early stopping (models use what they need)
    m.fit(num_steps=steps, subsample_size=Beff, lr=0.05, gamma=0.1, seed=seed, verbose=False)
    used=int(getattr(m,"stopped_step",steps))
    psi=m.psi_mean().cpu().numpy(); phi=m.phi_mean().cpu().numpy()
    # recovery of the 10 true programs (rectangular match)
    rec,ci=rect_match(psi, sim["psi"])
    aligned=np.stack([psi[ci[i]] for i in range(K_TRUE)])
    bt,bh=_beta(sim["psi"]).ravel(),_beta(aligned).ravel()
    slope=float(np.polyfit(bt,bh,1)[0]) if bt.std()>1e-12 else np.nan
    # PRUNING: usage per program; keep pi>tau (pipeline rule) then renormalize the survivors
    g=m.globals_(); pi=np.asarray(g["pi"]).ravel() if "pi" in g else phi.mean(0)
    phi_usage=phi.mean(0)
    pruned_K_pi=int((pi>PRUNE_TAU).sum()); pruned_K_phi=int((phi_usage>PRUNE_TAU).sum())
    p=np.clip(phi,1e-12,None); perp=float(np.median(np.exp(-(p*np.log(p)).sum(1))))
    return dict(recovered_true=rec, beta_slope=slope, initial_K=K_model,
                pruned_K_pi=pruned_K_pi, pruned_K_phi=pruned_K_phi, perplexity=perp,
                steps_used=used, seconds=round(time.time()-t0,1))

K_model=int(sys.argv[1]) if len(sys.argv)>1 else 10
seeds=[int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1","2"])]
device=sys.argv[3] if len(sys.argv)>3 else "cpu"
N=40000; J=100*K_TRUE
batches=[("full",N),("mb8192",8192)]; liks=["betabinomial","binomial"]; jpriors=[True,False]
import pandas as pd; rows=[]
for seed in seeds:
    sim=simulate(N=N,J=J,K=K_TRUE,seed=seed,overdisp_conc=10.0)
    print(f"\n===== K_true={K_TRUE} K_model={K_model} seed {seed}: J={J} =====",flush=True)
    for (bn,B),lik,jp in itertools.product(batches,liks,jpriors):
        r=run(sim,K_model,B,seed,device,lik,jp)
        r.update(dict(K_true=K_TRUE,batch=bn,likelihood=lik,junc_prior=jp,seed=seed)); rows.append(r)
        print(f"  {bn:>6} {lik[:3]} jp={str(jp):>5}  recover10={r['recovered_true']:.3f}  "
              f"pruned_K(pi)={r['pruned_K_pi']:>2} pruned_K(phi)={r['pruned_K_phi']:>2}/{K_model}  "
              f"slope={r['beta_slope']:.3f}  {r['seconds']}s",flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/sim_recover10_Kmodel{K_model}.csv"; df.to_csv(out,index=False)
print(f"\n===== K_model={K_model}: recover10 & pruned_K by (batch,likelihood,junc_prior) =====",flush=True)
print(df.groupby(["batch","likelihood","junc_prior"])[["recovered_true","pruned_K_phi"]].mean().round(3).to_string(),flush=True)
print(f"wrote {out}",flush=True)
