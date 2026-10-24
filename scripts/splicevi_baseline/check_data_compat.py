"""Check that every SpliceVI TMS / Allen file shares one junction and ATSE order, that it matches the
TMS70 LeafletFA fits, and that the TMS70 fits were trained on exactly SpliceVI's TMS TRAIN cells.

    python check_data_compat.py > data_compat.txt
"""
import os, sys
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_leafletfa_baseline import read_var, read_cells, _strings, _splicing_root

W = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"
D = f"{W}/MODEL_INPUT/20260526/"
S170 = "/gpfs/commons/home/kisaev/SpliceVI-utils/manuscript/age_analysis/audit_2026_09/screen_evals/s170/emb/inputs/"
stem = lambda ds: f"{ds}_ONLY_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu"
files = {}
for ds in ("TMS", "AB"):
    files[f"{ds} train"] = D + "train_70_30_" + stem(ds)
    files[f"{ds} test"] = D + "test_30_70_" + stem(ds)
    files[f"{ds} contentonly test"] = D + "SPLICINGCONTENTONLY_SEED20260926_test_30_70_" + stem(ds)
    files[f"{ds} s170 var.tsv"] = S170 + f"{ds}_var.tsv"
ref_j, ref_e = read_var(files["TMS train"])
print(f"reference = TMS train: {len(ref_j):,} junctions, {len(np.unique(ref_e)):,} ATSEs")
for k, p in files.items():
    if not os.path.exists(p):
        print(f"  {k:<22} MISSING {p}"); continue
    j, e = read_var(p)
    print(f"  {k:<22} junction_id equal {np.array_equal(j, ref_j)}  event_id equal {np.array_equal(e, ref_e)}")
# ATSE sizes: singletons?
_, cnt = np.unique(ref_e, return_counts=True)
print(f"junctions per ATSE: min {cnt.min()}, median {np.median(cnt):.0f}, max {cnt.max()}; singletons {int((cnt == 1).sum())}")
with h5py.File(files["TMS test"], "r") as f:
    lay = _splicing_root(f)["layers"]
    print("layers:", list(lay.keys()), " dtype", lay["cell_by_junction_matrix"]["data"].dtype)
# junction counts sum to the ATSE total (the DM derives N_e from y) on a sample of rows
import scipy.sparse as sp
from run_leafletfa_baseline import read_counts
for k in ("TMS test", "AB test"):
    rows = np.arange(0, 2000)
    Y = read_counts(files[k], rows)
    with h5py.File(files[k], "r") as f:
        g = _splicing_root(f)["layers"]["cell_by_cluster_matrix"]; ip = g["indptr"][: rows[-1] + 2]
        C = sp.csr_matrix((g["data"][: ip[-1]].astype(np.float32), g["indices"][: ip[-1]], ip), shape=(len(rows), len(ref_j)))
    _, inv = np.unique(ref_e, return_inverse=True)
    S = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)))
    tot = (Y @ S)[:, inv].toarray(); Cd = C.toarray()
    print(f"  {k}: sum_j y == cell_by_cluster on observed entries: {np.allclose(tot[Cd > 0], Cd[Cd > 0])}; "
          f"cluster>0 where y-total 0: {int(((Cd > 0) & (tot == 0)).sum())}; integer counts {np.allclose(Y.data, np.round(Y.data))}")
# TMS70 fit
z = np.load(f"{W}/ATLAS_DM/runs_TMS70/dm_simplex_K20_seed0_b8192/fit.npz", allow_pickle=True)
fe = np.asarray(z["event_id"]).astype(str)[z["inv_ev"]]
print(f"TMS70 K20 s0 fit: junction->ATSE map equal to SpliceVI var: {np.array_equal(fe, ref_e)}")
H5 = f"{D}train_splice_model_ready_2026-06-17.h5ad"
with h5py.File(H5, "r") as f:
    atlas_cells = _strings(f["obs"][f["obs"].attrs["_index"]])
tr = set(atlas_cells[z["train_idx"]]); sv = set(read_cells(files["TMS train"])); te = set(read_cells(files["TMS test"]))
print(f"TMS70 training cells {len(tr):,}; SpliceVI TMS TRAIN {len(sv):,}; identical {tr == sv}; overlap with TMS TEST {len(tr & te)}")
for ds in ("AB",):
    a, b = set(read_cells(files[f"{ds} train"])), set(read_cells(files[f"{ds} test"]))
    print(f"{ds}: train {len(a):,}, test {len(b):,}, overlap {len(a & b)}")
