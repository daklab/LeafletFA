"""Does raising the E-step learning rate make phi converge in far fewer epochs?

The test-time phi E-step (04e/07c) uses lr=0.01, gamma=0.05 -> lr crawls from 0.01 to 0.0005 and
never early-stops within budget (phi still moving at 800+ epochs; 07d). Training, by contrast, uses
lr=0.05. With Psi frozen each cell's phi is an independent near-convex K-dim problem, which SHOULD
converge fast -- so a too-low, too-hard-decayed lr is the prime suspect.

This holds ONE chunk of held-out cells fixed and, for each (lr, gamma), runs the E-step at increasing
epoch budgets from the SAME phi init, measuring how much phi still moves between budgets
(mean_L1 vs the largest budget). Converged fast = small movement at a small budget.

VERDICT: if lr=0.05 reaches mean_L1<0.05 at a much smaller budget than lr=0.01, the lr was the cause
and the fix is a one-line change to 04e/07c. If not, it is the guide scale/init and needs a different
fix.

Usage: estep_lr_sweep.py <eval_model_dir> <test_h5mu> <out_csv> [n_cells=4096]
"""
import sys, os, importlib.util
import numpy as np, pandas as pd, torch, h5py
from scipy.sparse import csr_matrix

P = "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa"
spec = importlib.util.spec_from_file_location("h07c", f"{P}/07c_heldout_structure.py")
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)

run_dir, test_h5mu, out_csv = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
BUDGETS = [100, 300, 800, 1500]
CONFIGS = [(0.01, 0.05), (0.05, 0.1), (0.1, 0.1), (0.1, 0.5)]   # (lr, gamma); first = current

m = h.load_model(run_dir)
psi = np.asarray(m["psi"], np.float32); K = psi.shape[0]
psi_t = torch.tensor(psi, dtype=torch.float32, device=h.DEVICE)
dir_conc = m.get("dir_conc"); jp = bool(m.get("junc_specific_prior", False))
pi = np.asarray(m["pi"], float) if m.get("pi") is not None else None
alpha_pi = np.asarray(m["alpha_pi"], float) if m.get("alpha_pi") is not None else None
bb = m.get("bb_conc")
print(f"model K={K} bb_conc={bb}", flush=True)

with h5py.File(test_h5mu, "r") as f:
    s = f["mod/splicing"]; Nt = min(N, int(s["obs/_index"].shape[0]))
    jr = np.array(s["layers/junc_ratio_original"][:Nt], np.float32)
    atse = h.read_sparse_layer(s["layers/cell_by_cluster_matrix_original"]).tocsr()[:Nt].astype(np.float32)
atse_d = atse.toarray(); junc = np.rint(jr * atse_d).astype(np.float32); del jr
chunk = h.build_chunk_adata(csr_matrix(junc), csr_matrix(atse_d)); del junc, atse_d
phi0 = h.sample_pi_phi_init(pi, K, Nt, np.random.default_rng(0))
print(f"chunk {Nt} cells x {psi.shape[1]} junctions", flush=True)

rows = []
for lr, gamma in CONFIGS:
    phis = {}
    for b in BUDGETS:
        h._E_STEP_CONV.clear()
        phis[b] = h.infer_phi_chunk(chunk, psi_t, K, dir_conc, jp, pi, alpha_pi, bb,
                                    b, 10**9, phi0, lr=lr, gamma=gamma)   # patience huge -> no early stop
    ref = phis[BUDGETS[-1]]
    for i, b in enumerate(BUDGETS):
        d = np.abs(phis[b] - ref).sum(1)
        prev = phis[BUDGETS[i-1]] if i > 0 else None
        dprev = float(np.abs(phis[b]-prev).sum(1).mean()) if prev is not None else np.nan
        pn = phis[b] / phis[b].sum(1, keepdims=True)
        perp = float(np.median(np.exp(-(pn*np.log(pn+1e-12)).sum(1))))
        rows.append({"lr": lr, "gamma": gamma, "epochs": b,
                     "mean_L1_vs_max": float(d.mean()), "mean_L1_vs_prev": dprev,
                     "corr_vs_max": float(np.corrcoef(phis[b].ravel(), ref.ravel())[0,1]),
                     "perplexity": perp})
    print(f"lr={lr} gamma={gamma}:", flush=True)
    for r in rows[-len(BUDGETS):]:
        print(f"   ep={r['epochs']:>4}  L1_vs_max={r['mean_L1_vs_max']:.4f}  "
              f"L1_vs_prev={r['mean_L1_vs_prev'] if np.isnan(r['mean_L1_vs_prev']) else round(r['mean_L1_vs_prev'],4)}  "
              f"corr={r['corr_vs_max']:.3f}  perp={r['perplexity']:.2f}", flush=True)

df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
print("\n=== VERDICT: epochs to reach mean_L1_vs_max < 0.05 (converged) per config ===", flush=True)
for (lr, gamma), g in df.groupby(["lr", "gamma"]):
    conv = g[g.mean_L1_vs_max < 0.05]["epochs"]
    print(f"  lr={lr} gamma={gamma}: {'converged by '+str(int(conv.min()))+' epochs' if len(conv) else 'NOT converged by 1500'}", flush=True)
print(f"\nwrote {out_csv}", flush=True)
