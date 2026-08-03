"""Sim figure: K_true=10 real programs, fit K_model in {10,20,30}. Do models recover the 10, and does
pruning trim over-specified models back? Reads sim_recover10_Kmodel{10,20,30}.csv."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch"; Kms=[10,20,30]
d=[]
for Km in Kms:
    f=f"{S}/sim_recover10_Kmodel{Km}.csv"
    if os.path.exists(f): d.append(pd.read_csv(f))
if not d: print("no recover10 CSVs yet"); raise SystemExit
d=pd.concat(d)
def ms(sub,col):
    x=sub[col].dropna().values.astype(float)
    return (np.mean(x), np.std(x,ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0) if len(x) else (np.nan,0)
def line(ax,sub,col,label,color, K=Kms):
    mn=[ms(sub[sub.initial_K==k],col)[0] for k in K]; se=[ms(sub[sub.initial_K==k],col)[1] for k in K]
    ax.errorbar(K,mn,yerr=se,fmt="o-",capsize=3,ms=7,label=label,color=color,lw=1.6)
fig,ax=plt.subplots(1,4,figsize=(20,4.8))
# A: batch ablation (recover the 10), ref BB + junc-specific
ref=d[(d.likelihood=="betabinomial")&(d.junc_prior==True)]
for v,lab,c in [("full","full-batch","#2E7D32"),("mb8192","minibatch B=8192","#4F81BD")]:
    line(ax[0],ref[ref.batch==v],"recovered_true",lab,c)
ax[0].set_title("A · recover 10 true programs\n(full vs minibatch)"); ax[0].legend(fontsize=8)
# B: likelihood ablation, ref full + junc-specific
ref=d[(d.batch=="full")&(d.junc_prior==True)]
for v,lab,c in [("betabinomial","Beta-Binomial","#0E6B63"),("binomial","Binomial","#C0504D")]:
    line(ax[1],ref[ref.likelihood==v],"recovered_true",lab,c)
ax[1].set_title("B · recover 10\n(likelihood)"); ax[1].legend(fontsize=8)
# C: junc prior ablation, ref full + BB
ref=d[(d.batch=="full")&(d.likelihood=="betabinomial")]
for v,lab,c in [(True,"junc-specific prior","#7C3B5E"),(False,"global prior","#E8A33D")]:
    line(ax[2],ref[ref.junc_prior==v],"recovered_true",lab,c)
ax[2].set_title("C · recover 10\n(junction prior)"); ax[2].legend(fontsize=8)
# D: pruning -- initial K vs pruned K (ref full BB junc-specific)
ref=d[(d.batch=="full")&(d.likelihood=="betabinomial")&(d.junc_prior==True)]
ax[3].plot(Kms,Kms,"k--",lw=1,label="initial K (fitted)")
line(ax[3],ref,"pruned_K_pi","pruned K (pi>0.01)","#B00020")
line(ax[3],ref,"pruned_K_phi","pruned K (phi usage)","#1565C0")
ax[3].axhline(10,ls=":",c="green",lw=1.5,label="true K=10")
ax[3].set_title("D · pruning: initial vs pruned K"); ax[3].legend(fontsize=8); ax[3].set_ylabel("K")
for a in ax[:3]:
    a.set_ylim(0,1.02); a.axhline(1.0,ls=":",c="grey",lw=0.8); a.set_ylabel("recovery of 10 true programs")
for a in ax: a.set_xlabel("K_model (fitted)"); a.set_xticks(Kms); a.grid(alpha=0.3)
fig.suptitle("Simulation: K_true=10 real programs. Over-specified models (K=20,30) still recover the 10; "
             "pruning behavior shown in D. (overdispersed, random init, early stopping)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93])
out=f"{S}/SIM_FIGURE_RECOVER10.png"; fig.savefig(out,dpi=140); print(f"wrote {out}")
print(d.groupby(["initial_K","batch","likelihood","junc_prior"])[["recovered_true","pruned_K_pi","pruned_K_phi"]].mean().round(2).to_string())
