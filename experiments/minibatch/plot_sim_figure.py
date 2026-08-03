"""SIMULATION figure (one dataset: N=40000, overdispersed Beta-Binomial, J=100*K).
A: Psi recovery vs batch (Beta-Binomial likelihood) per K -- minibatch matches full-batch.
B: DS effect-size slope (beta) vs batch per K -- 1.0 = no attenuation.
C: likelihood ablation at production batch B=8192 -- Beta-Binomial vs Binomial on overdispersed data.
Reads minibatch_vs_fullbatch_K{10,20,50}.csv (cols incl. B, likelihood, recovery_r, beta_slope, seed)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

S="/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch"
Ks=[10,20,50]; colors={10:"#0E6B63",20:"#A8641B",50:"#7C3B5E"}
def load():
    ds=[]
    for K in Ks:
        f=f"{S}/minibatch_vs_fullbatch_K{K}.csv"
        if os.path.exists(f):
            d=pd.read_csv(f); d["K"]=K
            if "likelihood" not in d: d["likelihood"]="betabinomial"
            ds.append(d)
    return pd.concat(ds) if ds else None
d=load()
if d is None: print("no CSVs yet"); raise SystemExit
def ms(sub,col):
    x=sub[col].dropna().values.astype(float)
    return (np.mean(x), np.std(x,ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0) if len(x) else (np.nan,0)

fig,ax=plt.subplots(1,3,figsize=(15,4.6))
bb=d[d.likelihood=="betabinomial"]
xorder=["full","8192","4096","2048"]; xpos=np.arange(len(xorder))
for panel,metric,ylab,ttl in [(0,"recovery_r","Psi recovery (corr to truth)","A · minibatch vs full-batch (recovery)"),
                               (1,"beta_slope","DS effect-size slope (beta)","B · effect-size fidelity (1.0=no attenuation)")]:
    for K in Ks:
        dk=bb[bb.K==K]; means=[];ses=[]
        for b in xorder:
            m,e=ms(dk[dk.B.astype(str)==b],metric); means.append(m);ses.append(e)
        ax[panel].errorbar(xpos,means,yerr=ses,fmt="o-",color=colors[K],capsize=3,ms=7,label=f"K={K}",lw=1.5)
    ax[panel].set_xticks(xpos);ax[panel].set_xticklabels(["full\nbatch","B=8192","B=4096","B=2048"])
    ax[panel].set_ylabel(ylab);ax[panel].set_title(ttl);ax[panel].grid(alpha=0.3);ax[panel].legend(fontsize=8)
ax[1].axhline(1.0,ls="--",c="k",lw=1)

# Panel C: Beta-Binomial vs Binomial likelihood at production batch B=8192 (overdispersed data)
prod=d[d.B.astype(str)=="8192"]; w=0.35; xk=np.arange(len(Ks))
for i,(lik,lab,c) in enumerate([("betabinomial","Beta-Binomial","#0E6B63"),("binomial","Binomial","#C0504D")]):
    rr=[ms(prod[(prod.K==K)&(prod.likelihood==lik)],"recovery_r")[0] for K in Ks]
    re=[ms(prod[(prod.K==K)&(prod.likelihood==lik)],"recovery_r")[1] for K in Ks]
    ax[2].bar(xk+(i-0.5)*w, rr, w, yerr=re, capsize=3, color=c, label=lab)
ax[2].set_xticks(xk);ax[2].set_xticklabels([f"K={K}" for K in Ks])
ax[2].set_ylabel("Psi recovery (corr to truth)")
ax[2].set_title("C · likelihood on overdispersed data (B=8192)\nBeta-Binomial vs Binomial")
ax[2].legend(fontsize=8);ax[2].grid(axis="y",alpha=0.3)

fig.suptitle("LeafletFA minibatch on simulated data (N=40k, overdispersed, known truth): matches full-batch; Beta-Binomial handles overdispersion",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95])
out=f"{S}/SIM_FIGURE.png";fig.savefig(out,dpi=140);print(f"wrote {out}")
for K in Ks:
    dk=d[d.K==K]; print(f"  K={K}: {len(dk)} rows, likelihoods={sorted(dk.likelihood.unique())}, seeds={sorted(dk.seed.unique())}")
