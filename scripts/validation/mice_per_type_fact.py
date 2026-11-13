"""Fact only (Karin 2026-11-13, no new analysis): per atlas, cell types with TRAIN and TEST cells from >= 4 mice (Allen: donors),
and the share of catalog rows with a check 12 interval that those types cover. Reads obs labels and the shipped catalog TSV."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels  # noqa: E402
out = {}
for ds in ("TMS", "Allen"):
    P = vlib.paths(ds)
    m = {}
    for part in ("train", "test"):
        _, t, mouse = labels(ds, P[part]); m[part] = pd.DataFrame(dict(t=t, m=mouse)).groupby("t").m.nunique()
    C = pd.read_csv(os.path.join(vlib.V, "c12", "export", f"catalog_{ds}_s0.tsv.gz"), sep="\t", usecols=["cell_type", "interval_status"])
    types = C.cell_type.unique()
    tr = m["train"].reindex(types).fillna(0); te = m["test"].reindex(types).fillna(0)
    r = {}
    for k in (2, 3, 4, 5):
        ok = set(types[(tr >= k).values & (te >= k).values])
        iv = C.interval_status.str.startswith("check 12")
        r[f">={k}"] = dict(types=len(ok), of_types=len(types), rows_with_interval=int((iv & C.cell_type.isin(ok)).sum()),
                           of_rows_with_interval=int(iv.sum()), share_interval_rows=float((iv & C.cell_type.isin(ok)).sum() / iv.sum()),
                           share_all_rows=float(C.cell_type.isin(ok).mean()))
    r["mice_total_train"] = int(len(np.unique(labels(ds, P["train"])[2])))
    out[ds] = r; print(ds, json.dumps(r, indent=1), flush=True)
json.dump(out, open(os.path.join(vlib.V, "c14_shrinkage", "mice_per_type_fact.json"), "w"), indent=1)
