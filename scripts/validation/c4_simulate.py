"""Check 4, step 1 (CPU): simulate TMS TRAIN from the fitted seed-0 model (truth = psi, per-ATSE
concentration, TRAIN cells' phi), keeping every cell's observed ATSE totals. Depth A = TMS, depth B =
EasySci-like (totals thinned Binomial(N, 0.011)).

    python c4_simulate.py   -> <V>/check4/sim_{A,B}_train.npz, cells.tsv, var.tsv, truth.npz
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

def main():
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    out = os.path.join(vlib.V, "check4"); os.makedirs(out, exist_ok=True)
    P = vlib.paths("TMS")
    jid, ev = vlib.read_var(P["train"]); _, inv = vlib.event_index(ev)
    cells = vlib.read_cells(P["train"])
    z = np.load(os.path.join(vlib.B, "TMS", "full_K20_s0_f64", "fit", "fit.npz"), allow_pickle=True)
    assert np.array_equal(z["cell_id"].astype(str), cells)
    phi, psi, conc = z["phi"].astype(np.float64), z["psi_mean"].astype(np.float64), z["bb_conc_vec"].astype(np.float64)
    Y = vlib.read_counts(P["train"])
    np.savez_compressed(os.path.join(out, "truth.npz"), phi=phi.astype(np.float32), psi=psi.astype(np.float32),
                        conc=conc.astype(np.float32), pi=z["pi"], cell_id=cells)
    pd.DataFrame({"cell_id": cells}).to_csv(os.path.join(out, "cells.tsv"), sep="\t", index=False)
    pd.DataFrame({"junction_id": jid, "event_id": ev}).to_csv(os.path.join(out, "var.tsv"), sep="\t", index=False)
    for name, thin, seed in (("A", None, 401), ("B", 0.011, 402)):
        p = os.path.join(out, f"sim_{name}_train.npz")
        if os.path.exists(p):
            continue
        S = vlib.simulate_fast(Y, inv, phi, psi, conc, thin=thin, seed=seed)
        sp.save_npz(p, S)
        r = np.asarray(S.sum(1)).ravel()
        print(f"sim {name}: reads {S.sum():,.0f} (real {Y.sum():,.0f}); cells with no reads {(r == 0).sum()}", flush=True)

if __name__ == "__main__":
    main()
