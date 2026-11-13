"""Display numbers only (Karin 2026-11-13): absolute error of check 14 comparator B (shrinkage to the across-type pooled
TRAIN psi) per 10.1 bin with the 10.1 bootstrap (cell types, 1,000, seed 0), added to c14_shrinkage/<ds>_abs_panel.tsv
as B, B_lo, B_hi (and LeafletFA - B with CI from c14 <ds>_accuracy.tsv). No new estimate: err_B is read from c14 units.
    python c14_abs_B.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
O = os.path.join(vlib.V, "c14_shrinkage")
for ds in ("TMS", "Allen"):
    U = pd.read_csv(os.path.join(O, f"{ds}_units.tsv.gz"), sep="\t", dtype={"size_bin": str, "read_bin": str})
    P = pd.read_csv(os.path.join(O, f"{ds}_abs_panel.tsv"), sep="\t", dtype={"bin": str})
    A14 = pd.read_csv(os.path.join(O, f"{ds}_accuracy.tsv"), sep="\t", dtype={"bin": str}).set_index(["by", "bin"])
    out = []
    for _, r in P.iterrows():
        sub = U[U[{"size": "size_bin", "reads": "read_bin"}[r.by]] == r.bin]
        rng = np.random.default_rng(0); per = sub.groupby("cell_type").err_B.agg(["sum", "size"])
        v = [(lambda x: x["sum"].sum() / x["size"].sum())(per.iloc[rng.integers(0, len(per), len(per))]) for _ in range(1000)]
        a = A14.loc[(r.by, r.bin)]
        assert len(sub) == r.units and abs(sub.err_B.mean() - a.err_B) < 1e-12
        out.append(dict(B=sub.err_B.mean(), B_lo=np.quantile(v, 0.025), B_hi=np.quantile(v, 0.975),
                        d_B=a.d_B, d_B_lo=a.d_B_lo, d_B_hi=a.d_B_hi))
    P = pd.concat([P.drop(columns=[c for c in ("B", "B_lo", "B_hi", "d_B", "d_B_lo", "d_B_hi") if c in P]), pd.DataFrame(out)], axis=1)
    P.to_csv(os.path.join(O, f"{ds}_abs_panel.tsv"), sep="\t", index=False)
    print(ds); print(P[["by", "bin", "raw", "B", "B_lo", "B_hi", "leafletfa", "d_B", "d_B_lo", "d_B_hi"]].round(4).to_string(index=False))
