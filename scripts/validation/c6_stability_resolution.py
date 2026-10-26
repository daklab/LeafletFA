"""Check 6 (CPU): program stability up to resolution (pre-registration amendment 3).

    python c6_stability_resolution.py

For TMS and Allen (baseline fits, K 20, seeds 0-2) and every ordered seed pair i -> j:
  1. many-to-one: each live seed-i program's centered psi regressed on the live seed-j programs' centered
     psi by non-negative least squares; smallest subset (<= 3) with R^2 >= 0.8
  2. meaning: cell-type activity profile r (best single / summed set) and top-50 marker-junction Jaccard
  3. classification: one-to-one (Hungarian, cosine >= 0.8, live) > up to splitting (R^2 >= 0.8 with <= 3)
     > not reproduced; merges = seed-j programs that need >= 2 seed-i programs
Writes tsv/c6_programs.tsv, tsv/c6_pairs.tsv, tsv/c6_splits_merges.tsv.
"""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment, nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

R2_MIN, MAX_SET, COS_MIN, LIVE, TOP, MIN_TYPE = 0.8, 3, 0.8, 0.01, 50, 20


def tissue_cell_type(path):
    import h5py
    with h5py.File(path, "r") as f:
        o = f["obs"]
        return vlib._strings(o[o.attrs["_index"]]).astype(str), vlib._strings(o["tissue_cell_type"]).astype(str)


def load(ds, seed, cells_ref, tct):
    z = np.load(os.path.join(vlib.B, ds, f"full_K20_s{seed}_f64", "fit", "fit.npz"), allow_pickle=True)
    assert np.array_equal(z["cell_id"].astype(str), cells_ref)
    psi = z["psi_mean"].astype(np.float64); pi = np.asarray(z["pi"], np.float64); pi = pi / pi.sum()
    phi = z["phi"].astype(np.float64)
    live = np.flatnonzero(np.bincount(phi.argmax(1), minlength=psi.shape[0]) / len(phi) >= LIVE)
    C = psi - pi @ psi                                                # centered psi (A70)
    # markers: departure of each program from the pi-weighted mean of the OTHER programs
    dep = np.empty_like(psi)
    for k in range(psi.shape[0]):
        w = np.delete(pi, k); w = w / w.sum()
        dep[k] = np.abs(psi[k] - w @ np.delete(psi, k, axis=0))
    prof = pd.DataFrame(phi).groupby(tct).mean()
    n = pd.Series(tct).value_counts()
    prof = prof.loc[n[n >= MIN_TYPE].index]
    return dict(C=C, live=live, dep=dep, prof=prof, phi=phi)


def best_subset(c, X, cols):
    """smallest s <= MAX_SET with NNLS R^2 >= R2_MIN; returns (s or None, members, weights, R2, R2_all)"""
    ss = (c ** 2).sum()
    _, r_all = nnls(X, c); r2_all = 1 - r_all ** 2 / ss
    for s in range(1, MAX_SET + 1):
        best = (-np.inf, None, None)
        for sub in itertools.combinations(range(X.shape[1]), s):
            w, r = nnls(X[:, sub], c)
            r2 = 1 - r ** 2 / ss
            if r2 > best[0]:
                best = (r2, sub, w)
        if best[0] >= R2_MIN:
            return s, [cols[i] for i in best[1]], best[2], best[0], r2_all
    return None, [cols[i] for i in best[1]], best[2], best[0], r2_all


def top_types(prof, k, n=3):
    return "; ".join(f"{t} ({v:.2f})" for t, v in prof[k].sort_values(ascending=False).head(n).items())


def main():
    rows, pairs, sm = [], [], []
    for ds in ("TMS", "Allen"):
        cells, tct = tissue_cell_type(vlib.paths(ds)["train"])
        F = {s: load(ds, s, cells, tct) for s in (0, 1, 2)}
        for i, j in itertools.permutations((0, 1, 2), 2):
            A, Bf = F[i], F[j]
            Cn = lambda C: C / np.linalg.norm(C, axis=1, keepdims=True).clip(1e-12)
            R = Cn(A["C"]) @ Cn(Bf["C"]).T
            hi, hj = linear_sum_assignment(-R); hung = dict(zip(hi, hj))
            X = Bf["C"][Bf["live"]].T                                    # J x n_live_j
            counts = {"one-to-one": 0, "up to splitting": 0, "not reproduced": 0}
            need_i = {}
            for a in A["live"]:
                b1 = hung[a]; cos1 = R[a, b1]
                s, members, w, r2, r2_all = best_subset(A["C"][a], X, list(Bf["live"]))
                need_i[a] = (s, members)
                if cos1 >= COS_MIN and b1 in Bf["live"]:
                    cls = "one-to-one"
                elif s is not None:
                    cls = "up to splitting"
                else:
                    cls = "not reproduced"
                counts[cls] += 1
                # meaning-level: profiles and markers
                pa = A["prof"][a]
                common = pa.index.intersection(Bf["prof"].index)
                r_single = max((np.corrcoef(pa.loc[common], Bf["prof"].loc[common, b])[0, 1], b) for b in Bf["live"])
                summed = Bf["prof"].loc[common, members].sum(1)
                r_set = np.corrcoef(pa.loc[common], summed)[0, 1]
                top_a = set(np.argsort(-A["dep"][a])[:TOP])
                b_cos = max(Bf["live"], key=lambda b: R[a, b])
                jac_single = len(top_a & set(np.argsort(-Bf["dep"][b_cos])[:TOP])) / len(top_a | set(np.argsort(-Bf["dep"][b_cos])[:TOP]))
                comb = np.abs(Bf["C"][members].T @ w)
                top_set = set(np.argsort(-comb)[:TOP])
                jac_set = len(top_a & top_set) / len(top_a | top_set)
                rows.append(dict(ds=ds, seed_i=i, seed_j=j, program=a + 1, top_types=top_types(A["prof"], a),
                                 hungarian_match=b1 + 1, hungarian_cos=cos1, match_live=b1 in Bf["live"],
                                 n_needed=s, set_members=",".join(str(m + 1) for m in members),
                                 set_weights=",".join(f"{x:.2f}" for x in w), r2_best_set=r2, r2_all_live=r2_all,
                                 cls=cls, profile_r_best_single=r_single[0], profile_best_single=r_single[1] + 1,
                                 profile_r_set=r_set, marker_jaccard_best_single=jac_single, marker_jaccard_set=jac_set))
                if cls == "up to splitting":
                    sm.append(dict(ds=ds, kind="split", seed_i=i, seed_j=j, program_i=a + 1, types_i=top_types(A["prof"], a),
                                   members_j=";".join(f"{m + 1} [{top_types(Bf['prof'], m)}]" for m in members),
                                   r2=r2, weights=",".join(f"{x:.2f}" for x in w)))
            n_live = len(A["live"])
            pairs.append(dict(ds=ds, seed_i=i, seed_j=j, live_i=n_live, live_j=len(Bf["live"]), **counts,
                              reproduced_frac=(counts["one-to-one"] + counts["up to splitting"]) / n_live))
        # merges: seed-j programs needing >= 2 seed-i programs = splits seen from j; already listed with roles
        # swapped (pair j -> i). Re-label them for readability.
    P = pd.DataFrame(pairs); Rw = pd.DataFrame(rows); S = pd.DataFrame(sm)
    if len(S):
        S = pd.concat([S, S.assign(kind="merge", seed_i=S.seed_j, seed_j=S.seed_i)
                       .rename(columns={"program_i": "merged_program_in_seed_i_of_split"})], ignore_index=True)
    out = os.path.join(vlib.V, "tsv")
    Rw.to_csv(os.path.join(out, "c6_programs.tsv"), sep="\t", index=False)
    P.to_csv(os.path.join(out, "c6_pairs.tsv"), sep="\t", index=False)
    S.to_csv(os.path.join(out, "c6_splits_merges.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
    print(P.round(3).to_string(index=False))
    for ds in ("TMS", "Allen"):
        p = P[P.ds == ds]
        print(f"{ds}: min reproduced fraction over ordered pairs {p.reproduced_frac.min():.2f} "
              f"-> {'>= 0.5 in every pair' if (p.reproduced_frac >= 0.5).all() else 'below 0.5 in some pair'}")
    ok = (P.reproduced_frac >= 0.5).all()
    print("READING:", "'programs are reproducible up to resolution' may be used" if ok else "keep the current wording")
    print(Rw[["ds", "seed_i", "seed_j", "program", "cls", "hungarian_cos", "n_needed", "set_members", "r2_best_set",
              "r2_all_live", "profile_r_best_single", "profile_r_set", "marker_jaccard_best_single", "marker_jaccard_set"]]
          .round(2).to_string(index=False))
    if len(S):
        print(S[S.kind == "split"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
