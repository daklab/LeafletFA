"""Is the trained psi at a likelihood optimum, or did training leave ELBO on the table?

Holds phi at its trained values and shifts psi in LOGIT space by delta, then evaluates the
Beta-Binomial log-likelihood of the ACTUAL counts. The argmax over delta decides the fork:

  best delta << 0  -> the likelihood prefers LOWER psi than training produced. Training did not
                      converge / is stuck. Fix is optimization-side (epochs, lr, init).
  best delta ~= 0  -> trained psi IS the likelihood optimum given phi. The high psi is then forced
                      by the model structure (diffuse phi + convex mixing), not by bad optimisation.
  best delta > 0   -> the likelihood genuinely prefers HIGH psi -> indexing/definition bug in how
                      junc/cluster counts feed the likelihood.
"""
import sys, importlib.util, numpy as np, h5py, scipy.sparse as sp
from scipy.special import betaln, gammaln
P="/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa"
sys.path.insert(0,P)
spec=importlib.util.spec_from_file_location("h",f"{P}/07c_heldout_structure.py")
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
MD="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFLETFA_MODELS/2026-07-18"
TRAIN="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/train_splice_model_ready_70pct_2026-06-17.h5ad"

def rows(g,n):
    ip=g["indptr"][:n+1]; a,b=int(ip[0]),int(ip[-1])
    return sp.csr_matrix((g["data"][a:b],g["indices"][a:b],ip-ip[0]),shape=(n,int(g.attrs["shape"][1])))

N=int(sys.argv[2]) if len(sys.argv)>2 else 1500
run=sys.argv[1] if len(sys.argv)>1 else "run_30"
m=h.load_model(f"{MD}/{run}")
psi=np.asarray(m["psi"],np.float64); phi=np.asarray(m["assign_post"],np.float64)
conc=float(np.asarray(m["bb_conc"]))
phi=phi/phi.sum(1,keepdims=True)
print(f"{run}  K={psi.shape[0]}  bb_conc={conc:.3f}  psi_mean={psi.mean():.4f}",flush=True)
with h5py.File(TRAIN,"r") as hf:
    junc=rows(hf["layers/cell_by_junction_matrix"],N).toarray().astype(np.float64)
    clus=rows(hf["layers/cell_by_cluster_matrix"],N).toarray().astype(np.float64)
ph=phi[:N]; cov=clus>0
k=junc[cov]; n=clus[cov]
const=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
print(f"cells={N}  covered={cov.sum():,}  observed rate={k.sum()/n.sum():.4f}",flush=True)
lg=np.log(psi/(1-psi+1e-12)+1e-12)
print(f"\n{'delta':>7} {'psi_mean':>9} {'pred_mean':>10} {'logLik/entry':>14}",flush=True)
print("-"*45,flush=True)
best=(None,-1e18)
for d in (1.0,0.5,0.0,-0.25,-0.5,-0.75,-1.0,-1.5,-2.0,-2.5,-3.0,-4.0):
    ps=1/(1+np.exp(-(lg+d)))
    mu=np.clip(ph@ps,1e-6,1-1e-6)[cov]
    a=mu*conc; b=(1-mu)*conc
    ll=float((const+betaln(k+a,n-k+b)-betaln(a,b)).mean())
    if ll>best[1]: best=(d,ll)
    print(f"{d:>7.2f} {ps.mean():>9.4f} {mu.mean():>10.4f} {ll:>14.5f}",flush=True)
    del ps,mu,a,b
print(f"\nBEST delta = {best[0]:+.2f}   (delta=0 = the trained model)",flush=True)
print(f"psi mean at best delta = {(1/(1+np.exp(-(lg+best[0])))).mean():.4f}",flush=True)
print(f"observed data rate     = {k.sum()/n.sum():.4f}",flush=True)
if best[0] is not None and best[0] < -0.2:
    print("\n=> VERDICT: likelihood prefers LOWER psi. Training left ELBO on the table -- NOT converged.",flush=True)
elif best[0] is not None and abs(best[0])<=0.2:
    print("\n=> VERDICT: trained psi IS the likelihood optimum given phi. High psi is structural.",flush=True)
else:
    print("\n=> VERDICT: likelihood prefers HIGHER psi -- suspect junc/cluster indexing bug.",flush=True)
