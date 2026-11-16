"""Check 18 (amendment 14, CPU): stratified mouse split of TMS and the half-A / half-B inputs.
    python c18_split.py   -> $V/c18/split.json, cells.parquet, halfA_counts.npz, halfA_cells.tsv, halfB_counts.npz, halfB_cells.tsv
All cells of the TMS-only TRAIN and TEST files are pooled; rows keep (file, row) provenance. Junction order = the TRAIN file's."""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels, log  # noqa: E402

DS = "TMS"; OUT = os.path.join(vlib.V, "c18"); N_SPLIT = 10000


def scores(pres, inA, keep):
    """pres: types x mice cell counts > 0; inA: bool over mice. (check16 types with >= 4 mice per half, types >= 1, types >= 2)"""
    a = (pres & inA[None, :]).sum(1); b = (pres & ~inA[None, :]).sum(1)
    return (int(((a >= 4) & (b >= 4))[keep].sum()), int(((a >= 1) & (b >= 1)).sum()), int(((a >= 2) & (b >= 2)).sum()))


def main():
    os.makedirs(OUT, exist_ok=True); P = vlib.paths(DS)
    jv, ev = vlib.read_var(P["train"]); jt, et = vlib.read_var(P["test"])
    assert np.array_equal(jv, jt) and np.array_equal(ev, et), "TRAIN and TEST junction order differ"
    pd.DataFrame(dict(junction_id=jv, event_id=ev)).to_csv(os.path.join(OUT, "var.tsv"), sep="\t", index=False)
    ctr, ttr, mtr = labels(DS, P["train"]); cte, tte, mte = labels(DS, P["test"])
    C = pd.DataFrame(dict(cell=np.r_[ctr, cte], cell_type=np.r_[ttr, tte], mouse=np.r_[mtr, mte],
                          source=["train"] * len(ctr) + ["test"] * len(cte), row=np.r_[np.arange(len(ctr)), np.arange(len(cte))]))
    assert C.cell.is_unique
    mice = np.array(sorted(C.mouse.unique())); log(f"{len(C):,} cells, {len(mice)} mice")
    n = C.cell_type.value_counts(); types = np.array(sorted(n[n >= 10].index))
    R = pd.read_csv(os.path.join(vlib.V, "c10_lc", f"{DS}_halfA", "types.tsv"), sep="\t"); keep = np.isin(types, R.type.values); assert keep.sum() == 100
    ct = pd.crosstab(C.cell_type, C.mouse).reindex(index=types, columns=mice).fillna(0).values > 0
    rng = np.random.default_rng(18); best, bi, bA = None, -1, None
    for i in range(N_SPLIT):
        A = np.isin(mice, rng.permutation(mice)[:9]); s = scores(ct, A, keep)
        if best is None or s > best:
            best, bi, bA = s, i, A
    trm = np.unique(mtr); old = set(np.random.default_rng(0).permutation(trm)[: len(trm) // 2])         # the 10.2a split
    s_old = scores(ct, np.isin(mice, list(old)), keep)
    mA = mice[bA]; C["half"] = np.where(C.mouse.isin(mA), "A", "B")
    rec = dict(n_mice=len(mice), draws=N_SPLIT, chosen_draw=bi, half_A=mA.tolist(), half_B=mice[~bA].tolist(),
               scores_chosen=dict(check16_types_4_mice_each=best[0], types_1_mouse_each=best[1], types_2_mice_each=best[2]),
               scores_10_2a_split=dict(check16_types_4_mice_each=s_old[0], types_1_mouse_each=s_old[1], types_2_mice_each=s_old[2]),
               n_types_ge10_cells=len(types), cells_A=int((C.half == "A").sum()), cells_B=int((C.half == "B").sum()))
    json.dump(rec, open(os.path.join(OUT, "split.json"), "w"), indent=1); log(json.dumps(rec))
    C.to_parquet(os.path.join(OUT, "cells.parquet"), index=False)
    Ytr = vlib.read_counts(P["train"]).tocsr(); Yte = vlib.read_counts(P["test"]).tocsr()
    for h in ("A", "B"):
        s = C[C.half == h]
        Y = sp.vstack([Ytr[s.row[s.source == "train"].values], Yte[s.row[s.source == "test"].values]]).tocsr()
        cells = np.r_[s.cell[s.source == "train"].values, s.cell[s.source == "test"].values]
        sp.save_npz(os.path.join(OUT, f"half{h}_counts.npz"), Y.astype(np.float32))
        pd.DataFrame(dict(cell=cells)).to_csv(os.path.join(OUT, f"half{h}_cells.tsv"), sep="\t", index=False)
        log(f"half {h}: {Y.shape[0]:,} cells x {Y.shape[1]:,}, nnz {Y.nnz:,}, empty {int((np.diff(Y.indptr) == 0).sum())}")


if __name__ == "__main__":
    main()
