"""Measure peak GPU memory for one SVI step at ATLAS junction count (J=79768) across batch sizes.
Answers: what is the largest batch that fits on the L40S (44.5 GB) at atlas scale? Uses synthetic
data of the real J so memory is realistic; recovery is irrelevant here."""
import sys, numpy as np, torch, pyro
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
from leafletfa.minibatch import LeafletFAMinibatch
J=79768; K=30; device="cuda"
Bs=[int(x) for x in (sys.argv[1].split(",") if len(sys.argv)>1 else ["8192","12288","16384","20480","24576"])]
# small N; we only load a B-sized minibatch per step, data kept on CPU (as at atlas scale)
N=40000
rng=np.random.default_rng(0)
print(f"atlas-scale memory probe: J={J}, K={K}, card={torch.cuda.get_device_name(0)}",flush=True)
for B in Bs:
    try:
        pyro.clear_param_store(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        clus=(rng.random((N,J))<0.135)*(1+rng.poisson(12,(N,J)))
        junc=rng.binomial(clus.astype(np.int64), np.clip(rng.random((N,J)).astype(np.float32),0,1))
        m=LeafletFAMinibatch(junc,clus,K=K,device=device,likelihood="betabinomial",data_on_cpu=True)
        pyro.set_rng_seed(0); m._B=B; m.guide=m._make_guide(B)
        from pyro.infer import SVI,Trace_ELBO; from pyro.optim import ClippedAdam
        svi=SVI(m.model,m.guide,ClippedAdam({"lr":0.05,"clip_norm":10.0}),Trace_ELBO())
        loss=svi.step(); loss=svi.step()
        peak=torch.cuda.max_memory_allocated()/1e9
        print(f"  B={B:>6}  PEAK GPU={peak:5.1f} GB  loss_finite={np.isfinite(loss)}  -> {'FITS' if peak<44 else 'RISKY'}",flush=True)
        del m,clus,junc,svi
    except RuntimeError as e:
        print(f"  B={B:>6}  OOM / error: {str(e)[:80]}",flush=True)
        torch.cuda.empty_cache()
