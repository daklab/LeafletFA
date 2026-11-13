"""Fact only (Karin 2026-11-13; no new analysis): how many shipped catalog rows a 20-99 TEST-read interval band would add,
and the share of rows with >= 20 TRAIN reads at the ATSE that already have an interval. Reads the shipped catalog TSVs.
test_reads / test_cells = TEST reads and TEST cells with reads at the row's ATSE (c12_export.py)."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
out = {}
for ds in ("TMS", "Allen"):
    C = pd.read_csv(os.path.join(vlib.V, "c12", "export", f"catalog_{ds}_s0.tsv.gz"), sep="\t",
                    usecols=["interval_status", "test_reads", "test_cells", "atse_reads"])
    n = len(C); iv = C.interval_status.str.startswith("check 12")
    band = C.test_reads.between(20, 99)
    r = dict(rows=n, rows_with_interval=int(iv.sum()), share_with_interval=iv.mean())
    for k in (10, 5):
        b = band & (C.test_cells >= k)
        r[f"band_20_99_cells_ge{k}"] = int(b.sum()); r[f"share_if_band_passes_cells_ge{k}"] = float((iv | b).mean())
    for t in (20, 100):
        m = C.atse_reads >= t
        r[f"rows_train_reads_ge{t}"] = int(m.sum()); r[f"share_of_those_with_interval_train_ge{t}"] = float(iv[m].mean())
        r[f"share_of_those_with_interval_if_band10_train_ge{t}"] = float((iv | (band & (C.test_cells >= 10)))[m].mean())
    out[ds] = r; print(ds, json.dumps(r, indent=1), flush=True)
json.dump(out, open(os.path.join(vlib.V, "c12", "band_20_99_fact.json"), "w"), indent=1)
