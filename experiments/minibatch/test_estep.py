"""Validate the frozen-global E-step AND confirm standard training is unaffected."""
import numpy as np, warnings, torch, pyro; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch
from pyro import poutine
dev = "cuda" if torch.cuda.is_available() else "cpu"
rng=np.random.default_rng(0); N,J,K=3000,400,4
psi_true=rng.beta(0.5,0.5,size=(K,J)).astype(np.float32)
phi_true=rng.dirichlet(np.full(K,0.3),size=N).astype(np.float32)
clus=((rng.random((N,J))<0.3)*(1+rng.poisson(12,(N,J)))).astype(np.int64)
junc=rng.binomial(clus, np.clip(phi_true@psi_true,1e-6,1-1e-6))

print("=== A. E-STEP: frozen globals, infer phi only ===", flush=True)
def estep(seed):
    pyro.clear_param_store()
    m=LeafletFAMinibatch(junc,clus,K=K,device=dev,likelihood="betabinomial")
    m.freeze_globals(psi=psi_true, pi=np.full(K,1.0/K), alpha_pi=1.0, dir_conc=1.0, bb_conc=20.0)
    return m, m.infer_phi(num_steps=1500, lr=0.1, gamma=0.1, seed=seed)
m0,p0=estep(0); m1,p1=estep(1)
r=np.mean([np.corrcoef(p0[:,k],phi_true[:,k])[0,1] for k in range(K)])
print(f"  recovery of true phi: mean per-program corr = {r:.4f}")
print(f"  seed stability: max|d|={np.abs(p0-p1).max():.5f}  mean|d|={np.abs(p0-p1).mean():.6f}")
sites=set(poutine.trace(m0.guide).get_trace().nodes)
leaked=[s for s in ("psi","pi","alpha_pi","dir_conc","bb_conc","a","b") if s in sites]
print(f"  globals leaking into the guide (must be []): {leaked}")
# psi must be bit-identical to what we froze
got=m0.psi_mean().cpu().numpy()
print(f"  psi unchanged after E-step: max|psi_hat - psi_frozen| = "
      f"{np.abs(got-np.clip(psi_true,1e-4,1-1e-4)).max():.2e}")

print("\n=== B. REGRESSION: standard training must be unaffected ===", flush=True)
def train(seed):
    pyro.clear_param_store()
    m=LeafletFAMinibatch(junc,clus,K=K,device=dev,likelihood="betabinomial")
    m.fit(num_steps=1200, subsample_size=1024, lr=0.05, gamma=0.1, seed=seed, verbose=False)
    return m
t=train(0)
sites=set(poutine.trace(t.guide).get_trace().nodes)
need=[s for s in ("psi","pi","alpha_pi","dir_conc","bb_conc","phi") if s not in sites]
print(f"  sites MISSING from a normal guide (must be []): {need}")
ps=t.psi_mean().cpu().numpy()
rr=[]
for k in range(K):
    rr.append(max(abs(np.corrcoef(ps[i],psi_true[k])[0,1]) for i in range(K)))
print(f"  training still recovers psi: mean best-match corr = {np.mean(rr):.4f}")
print(f"  globals learned: bb_conc={t.globals_().get('bb_conc')}, dir_conc={t.globals_().get('dir_conc')}")
print("\nDONE", flush=True)
