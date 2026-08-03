"""Convergence figure: recovery & effect-size-fidelity vs EPOCHS, one line per batch size, faceted by K.
If batch-size lines overlap -> epochs governs convergence, B only affects speed (use big B for throughput).
If small-B lines sit lower -> minibatch gradient noise itself costs recovery.
Reads grid_convergence_K{10,20,50}.csv."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

S="/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch"
Ks=[10,20]
bcolors={2048:"#C0504D",4096:"#E8A33D",8192:"#4F81BD","full":"#2E7D32"}
def ms(sub,col):
    x=sub[col].dropna().values.astype(float)
    return (np.mean(x), np.std(x,ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0) if len(x) else (np.nan,0)
dfs={}
for K in Ks:
    f=f"{S}/grid_convergence_K{K}.csv"
    if os.path.exists(f): dfs[K]=pd.read_csv(f)
if not dfs: print("no grid CSVs yet"); raise SystemExit

fig,ax=plt.subplots(2,len(Ks),figsize=(5*len(Ks),8.5),squeeze=False)
for ci,K in enumerate(Ks):
    if K not in dfs:
        for r in (0,1): ax[r][ci].set_title(f"K={K} (pending)")
        continue
    d=dfs[K]
    for metric,row,ylab,ref in [("recovery_r",0,"Psi recovery (corr to truth)",None),
                                ("beta_slope",1,"DS effect-size slope (beta)",1.0)]:
        for b in [2048,4096,8192,"full"]:
            db=d[d.batch.astype(str)==str(b)]
            if not len(db): continue
            eps=sorted(db.epochs.unique())
            mn=[ms(db[db.epochs==e],metric)[0] for e in eps]
            se=[ms(db[db.epochs==e],metric)[1] for e in eps]
            ax[row][ci].errorbar(eps,mn,yerr=se,fmt="o-",color=bcolors[b],capsize=2,ms=5,
                                 label=("full-batch" if b=="full" else f"B={b}"),lw=1.4)
        ax[row][ci].set_xscale("log")
        ax[row][ci].set_xlabel("epochs (steps x B / N)")
        ax[row][ci].set_ylabel(ylab); ax[row][ci].grid(alpha=0.3,which="both")
        if ref is not None: ax[row][ci].axhline(ref,ls="--",c="k",lw=1)
        ax[row][ci].set_title(f"K={K}  (J={max(400,100*K)})")
        if ci==0 and row==0: ax[row][ci].legend(fontsize=8,title="batch size")
fig.suptitle("Convergence vs epochs, batch size, and K (N=40k, overdispersed sim, known truth)\n"
             "Top: program recovery.  Bottom: DS effect-size fidelity (1.0=no attenuation).",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
out=f"{S}/CONVERGENCE_FIGURE.png"; fig.savefig(out,dpi=140); print(f"wrote {out}")
# text summary: does batch matter at fixed epochs?
for K in dfs:
    d=dfs[K]
    print(f"\nK={K}: recovery_r pivot (batch x epochs)")
    print(d.pivot_table(index="batch",columns="epochs",values="recovery_r",aggfunc="mean").round(3).to_string())
