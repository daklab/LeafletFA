"""Which term parks psi at 0.67 -- the likelihood, or the prior?

Sweeps a logit-space shift delta on the trained psi and evaluates BOTH ELBO terms:
  loglik(delta)  Beta-Binomial log p(counts | phi@psi_shifted), scaled to the FULL cell count
  logprior(delta) sum over K*J of Beta(a,b) log-density of psi_shifted

Training minimises -(loglik + logprior). So:
  argmin near delta=0     -> the objective really is optimal at psi=0.67. The MODEL is misspecified
                            (the psi prior is fighting the data). Fix the prior.
  argmin near delta=-2.5  -> the objective wants low psi and training never got there.
                            The OPTIMISER is at fault (guide reset / epochs / lr).
"""
import sys, importlib.util, numpy as np, h5py, scipy.sparse as sp
from scipy.special import betaln, gammaln
P="/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa"
spec=importlib.util.spec_from_file_location("h",f"{P}/07c_heldout_structure.py")
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
MD="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18"
TRAIN="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad"
run=sys.argv[1] if len(sys.argv)>1 else "run_30"
N=int(sys.argv[2]) if len(sys.argv)>2 else 1500
N_TOTAL=92894

m=h.load_model(f"{MD}/{run}")
psi=np.asarray(m["psi"],np.float64); phi=np.asarray(m["assign_post"],np.float64)
conc=float(np.asarray(m["bb_conc"])); phi=phi/phi.sum(1,keepdims=True)
a=np.asarray(m["a"],np.float64); b=np.asarray(m["b"],np.float64)
if a.ndim==0: a=a.reshape(1)
if b.ndim==0: b=b.reshape(1)
print(f"{run} K={psi.shape[0]} J={psi.shape[1]}  bb_conc={conc:.3f}",flush=True)
print(f"  a: shape={a.shape} mean={a.mean():.3f}   b: shape={b.shape} mean={b.mean():.3f}",flush=True)
print(f"  prior mean a/(a+b) = {(a/(a+b)).mean():.4f}   trained psi mean = {psi.mean():.4f}",flush=True)

def rows(g,n):
    ip=g["indptr"][:n+1]; s,e=int(ip[0]),int(ip[-1])
    return sp.csr_matrix((g["data"][s:e],g["indices"][s:e],ip-ip[0]),shape=(n,int(g.attrs["shape"][1])))
with h5py.File(TRAIN,"r") as hf:
    junc=rows(hf["layers/cell_by_junction_matrix"],N).toarray().astype(np.float64)
    clus=rows(hf["layers/cell_by_cluster_matrix"],N).toarray().astype(np.float64)
ph=phi[:N]; cov=clus>0; k=junc[cov]; n=clus[cov]
const=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
scale=N_TOTAL/float(N)          # extrapolate the subsample to the full training set
print(f"  cells={N} covered={cov.sum():,}  scaling likelihood by {scale:.1f}x to full N",flush=True)
lg=np.log(psi/(1-psi+1e-12)+1e-12)
aa=a if a.size>1 else a[0]; bb=b if b.size>1 else b[0]
print(f"\n{'delta':>7} {'psiMean':>8} {'logLik(full)':>15} {'logPrior':>15} {'TOTAL':>15}",flush=True)
print("-"*66,flush=True)
best=(None,-1e30)
for d in (0.5,0.0,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0,-3.5):
    ps=np.clip(1/(1+np.exp(-(lg+d))),1e-9,1-1e-9)
    mu=np.clip(ph@ps,1e-6,1-1e-6)[cov]
    al=mu*conc; be=(1-mu)*conc
    ll=float((const+betaln(k+al,n-k+be)-betaln(al,be)).sum())*scale
    lp=float((( aa-1)*np.log(ps)+(bb-1)*np.log(1-ps)-betaln(aa,bb)).sum())
    tot=ll+lp
    if tot>best[1]: best=(d,tot)
    print(f"{d:>7.2f} {ps.mean():>8.4f} {ll:>15.4e} {lp:>15.4e} {tot:>15.4e}",flush=True)
    del ps,mu,al,be
print(f"\nBEST delta by TOTAL objective = {best[0]:+.2f}",flush=True)
if best[0] is not None and abs(best[0])<=0.5:
    print("=> The OBJECTIVE is optimal near the trained psi. Training CONVERGED correctly to a",flush=True)
    print("   misspecified model: the psi prior is strong enough to overrule the data. FIX THE PRIOR.",flush=True)
else:
    print("=> The OBJECTIVE wants LOWER psi but training never got there. The prior does NOT",flush=True)
    print("   explain psi=0.67. FIX THE OPTIMISER (guide reset across chunks / epochs / lr).",flush=True)
