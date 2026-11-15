"""Check 15b (amendment 11): rerun of check 15 without the atlas-wide shift.
    python c15b_simulate.py acc    -> $V/c15b/acc/{sim_full,sim_f30,sim_f10}_train.npz, truth.npz, sim_summary.json
    python c15b_simulate.py diff   -> $V/c15b/diff/...
Truth psi, gamma_e, halves, planted ATSEs and delta are the amendment 10 arrays ($V/c15/truth.npz; same construction and
seeds 10-12). acc: no halves, no shifts (thinning/DM seeds 31-33). diff: per planted ATSE, half B shifted in a random 20 %
of scored types (seed 13), donor = the type's largest-psi junction, receiver random among the others (seed 13); seeds 41-43."""
import os, sys, json
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

ARM = sys.argv[1]; assert ARM in ("acc", "diff")
C15 = os.path.join(vlib.V, "c15"); OUT = os.path.join(vlib.V, "c15b", ARM); os.makedirs(OUT, exist_ok=True)
tr = dict(np.load(os.path.join(C15, "truth.npz"), allow_pickle=True))
P = vlib.paths("TMS"); _, ev = vlib.read_var(P["train"]); _, inv = vlib.event_index(ev)
assert np.array_equal(inv, tr["inv"]) and np.array_equal(vlib.read_cells(P["train"]).astype(str), tr["cell_id"].astype(str))
Y = vlib.read_counts(P["train"]).tocsr().astype(np.float64)
T = len(tr["types"]); g = tr["g"]; truth, pool, gam = tr["truth"], tr["pool"], tr["gamma"]; E_ = len(tr["scored_e"])
jidx = [np.flatnonzero(inv == e) for e in range(E_)]
summ = dict(arm=ARM, cells=int(len(g)), types=T, scored_atses=int(tr["scored_e"].sum()))
if ARM == "acc":
    phi = g.astype(np.int64); psi_g = np.vstack([truth, pool[None, :]])                 # groups 0..T-1 types, T unscored
    shift = np.zeros((T, E_)); truthB = truth; chosen = np.zeros((T, E_), bool); seeds = (31, 32, 33)
else:
    half_b = tr["half_b"]; planted, delta = tr["planted"], tr["delta"]
    rng = np.random.default_rng(13); truthB = truth.copy(); shift = np.zeros((T, E_)); chosen = np.zeros((T, E_), bool)
    n_ch = int(round(0.2 * T))
    for e in np.flatnonzero(planted):
        js = jidx[e]
        for k in rng.permutation(T)[:n_ch]:
            p = truth[k, js]; jf = int(np.argmax(p)); others = [i for i in range(len(js)) if i != jf]
            jt = others[int(rng.integers(len(others)))]
            s = min(delta[e], p[jf]); truthB[k, js[jf]] -= s; truthB[k, js[jt]] += s; shift[k, e] = s; chosen[k, e] = True
    grp = g * 2 + half_b.astype(int); phi = grp.astype(np.int64)
    psi_g = np.vstack([np.vstack([truth[k], truthB[k]]) if k < T else np.vstack([pool, pool]) for k in range(T + 1)])
    seeds = (41, 42, 43)
    sc = tr["scored_e"]
    summ.update(planted_atses=int(planted.sum()), types_per_planted_atse=n_ch,
                planted_pairs=int((shift > 0).sum()), chosen_pairs=int(chosen.sum()),
                full_delta_share=float(np.isclose(shift[chosen], np.broadcast_to(delta[None, :], chosen.shape)[chosen]).mean()),
                per_type_planted_share_median=float(np.median((shift[:, sc] > 0).mean(1))))
np.savez_compressed(os.path.join(OUT, "truth.npz"), **{k: v for k, v in tr.items() if k not in ("truthB", "shift")},
                    truthB=truthB, shift=shift, chosen=chosen, arm=ARM)
for name, f, seed in (("full", None, seeds[0]), ("f30", 0.3, seeds[1]), ("f10", 0.1, seeds[2])):
    S = vlib.simulate_fast(Y, inv, phi, psi_g, gam, thin=f, seed=seed)
    sp.save_npz(os.path.join(OUT, f"sim_{name}_train.npz"), S); summ[f"reads_{name}"] = float(S.sum())
    print(name, f"{S.sum():,.0f} reads", flush=True)
summ["reads_real"] = float(Y.sum())
json.dump(summ, open(os.path.join(OUT, "sim_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))
