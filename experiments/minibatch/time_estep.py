"""Where does the E-step time actually go? One chunk at exactly 04e's dimensions."""
import numpy as np, torch, time, warnings, pyro; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch
dev="cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev}  {torch.cuda.get_device_name(0) if dev=='cuda' else ''}", flush=True)
N,J,K = 4096, 12197, 10                       # 04e: chunk_size=4096, 12,197 HVJ, K=10
rng=np.random.default_rng(0)
psi=rng.beta(.5,.5,size=(K,J)).astype(np.float32)
clus=((rng.random((N,J))<0.5)*(1+rng.poisson(12,(N,J)))).astype(np.float32)
junc=rng.binomial(clus.astype(np.int64), 0.3).astype(np.float32)
pyro.clear_param_store()
m=LeafletFAMinibatch(junc,clus,K=K,device=dev,likelihood="betabinomial")
m.freeze_globals(psi=psi, pi=np.full(K,1/K), alpha_pi=1.0, dir_conc=1.0, bb_conc=2.0)
t0=time.time(); m.infer_phi(num_steps=500, lr=0.1, gamma=0.1, seed=0, verbose=False)
dt=time.time()-t0
used=int(getattr(m,'stopped_step',500)); early=bool(getattr(m,'stopped_early',False))
print(f"500-step cap: took {dt:.1f}s  ({dt/max(used,1)*1000:.0f} ms/step)")
print(f"steps actually used = {used}   early_stop fired = {early}")
print(f"-> per level (10 chunks): {dt*10/60:.1f} min ;  per model (7 levels): {dt*70/60:.1f} min")
# would fewer steps suffice?
for ns in (100, 200):
    pyro.clear_param_store()
    m2=LeafletFAMinibatch(junc,clus,K=K,device=dev,likelihood="betabinomial")
    m2.freeze_globals(psi=psi, pi=np.full(K,1/K), alpha_pi=1.0, dir_conc=1.0, bb_conc=2.0)
    t=time.time(); p2=m2.infer_phi(num_steps=ns, lr=0.1, gamma=0.1, seed=0, verbose=False)
    print(f"  {ns:>3} steps: {time.time()-t:.1f}s")
