"""Check 10.2 null, read-only follow-up to c10_null_diag.py: paired real-minus-null (ii) with a type bootstrap, and shares of null confirmations by per-cell variance. Exploratory; verdict unchanged."""
import pandas as pd,numpy as np
V="/gpfs/commons/groups/knowles_lab/Karin/LeafletFA_validation/c10_null_diag/"
for ds in ["TMS","Allen"]:
    A=pd.read_csv(V+ds+"_calls.tsv.gz",sep="\t")
    A=A.assign(d=A.conf_ii-A.null_ii)
    g=A.groupby("cell_type").agg(n=("d","size"),s=("d","sum"),ci=("conf_ii","sum"),ni=("null_ii","sum"))
    rng=np.random.default_rng(0);B=[]
    for _ in range(2000):
        x=g.iloc[rng.integers(0,len(g),len(g))];B.append(x.s.sum()/x.n.sum())
    print(ds,"types with calls",len(g),"real-null(ii) paired",round(A.d.mean(),4),np.round(np.percentile(B,[2.5,97.5]),4))
    print(" per-type excess median",round((g.s/g.n).median(),3),"types with excess<=0",int((g.s<=0).sum()))
    w=A.null_ii.sum()
    print(" cell_var quantiles",A.cell_var.quantile([.1,.5,.9]).round(3).tolist())
    for t in (.1,.2,.24):
        m=A.cell_var>=t;print(f"  cell_var>={t}: calls {m.mean():.3f} null-conf share {A.null_ii[m].sum()/w:.3f} null {A.null_ii[m].mean():.3f} real {A.conf_ii[m].mean():.3f}")
    print(" spearman null_ii~cell_var",round(A.null_ii.corr(A.cell_var,method='spearman'),3),"~reads",round(A.null_ii.corr(A.null_perm_reads,method='spearman'),3),"~retained",round(A.null_ii.corr(A.retained,method='spearman'),3))
    print(" retained/base median",round((A.retained/A.base_rate).median(),2))
