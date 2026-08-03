"""Sim figure (K=10/20/30, random init, no waypoints). Three ablation panels, each isolating one axis
while holding the others at the reference config (full-batch, Beta-Binomial, junction-specific prior).
Reads sim_factorial_K{10,20,30}.csv. Plots recovery (top) and beta-slope (bottom) vs K."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S="/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch"; Ks=[10,20,30]
d=[]
for K in Ks:
    f=f"{S}/sim_factorial_K{K}.csv"
    if os.path.exists(f): d.append(pd.read_csv(f))
if not d: print("no factorial CSVs yet"); raise SystemExit
d=pd.concat(d)
def ms(sub,col):
    x=sub[col].dropna().values.astype(float)
    return (np.mean(x), np.std(x,ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0) if len(x) else (np.nan,0)
def line(ax,sub,metric,label,color):
    mn=[ms(sub[sub.K==K],metric)[0] for K in Ks]; se=[ms(sub[sub.K==K],metric)[1] for K in Ks]
    ax.errorbar(Ks,mn,yerr=se,fmt="o-",capsize=3,ms=7,label=label,color=color,lw=1.6)
# reference: full-batch, betabinomial, junc_prior=True
REF=dict(batch="full",likelihood="betabinomial",junc_prior=True)
panels=[
 ("batch", "A · full vs minibatch", [("full","full-batch","#2E7D32"),("mb8192","minibatch B=8192","#4F81BD")],
   dict(likelihood="betabinomial",junc_prior=True)),
 ("likelihood","B · likelihood", [("betabinomial","Beta-Binomial","#0E6B63"),("binomial","Binomial","#C0504D")],
   dict(batch="full",junc_prior=True)),
 ("junc_prior","C · junction-specific psi prior",[(True,"junc-specific prior","#7C3B5E"),(False,"shared a,b prior","#E8A33D")],
   dict(batch="full",likelihood="betabinomial")),
]
fig,ax=plt.subplots(2,3,figsize=(15,8),squeeze=False)
for ci,(axis,title,levels,hold) in enumerate(panels):
    for metric,row,ylab,ref in [("recovery_r",0,"Psi recovery (corr to truth)",None),
                                ("beta_slope",1,"DS effect-size slope",1.0)]:
        for val,lab,c in levels:
            sub=d.copy()
            for k,v in hold.items(): sub=sub[sub[k]==v]
            sub=sub[sub[axis]==val]
            line(ax[row][ci],sub,metric,lab,c)
        ax[row][ci].set_xticks(Ks); ax[row][ci].set_xlabel("K (programs)")
        ax[row][ci].set_ylabel(ylab); ax[row][ci].grid(alpha=0.3)
        if ref is not None: ax[row][ci].axhline(ref,ls="--",c="k",lw=1)
        if row==0: ax[row][ci].set_title(title); ax[row][ci].legend(fontsize=8)
fig.suptitle("LeafletFA on simulated data (N=40k, overdispersed, known truth; random init, no waypoints)\n"
             "Top: program recovery.  Bottom: DS effect-size fidelity (1.0=no attenuation).",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
out=f"{S}/SIM_FIGURE_FACTORIAL.png"; fig.savefig(out,dpi=140); print(f"wrote {out}")
print(d.groupby(["K","batch","likelihood","junc_prior"])["recovery_r"].mean().round(3).to_string())
