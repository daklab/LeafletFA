"""LeafCutter on check 10 cell types (CPU). Record: VALIDATION_PREREG.md "LeafCutter descriptive control for 10.2".
    python c10_leafcutter.py prepare <TMS|Allen> <halfA|full>     # write inputs; halfA = the 10.2 control, full = the 10.6 catalog benchmark (all TRAIN mice)
    python c10_leafcutter.py collect <TMS|Allen>                  # control only: calls, confirmation (ii), both nulls, LeafletFA on the same types
Inputs and array list: c10_lc/<ds>_<design>/. Submission: c10_lc_submit.sh."""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import Data, Groups, type_setup, mouse_halves, eligibility, estimates, raw_pb, reads_cov, tv, lead_sign, log, N_PERM  # noqa: E402
from c9_ds_null import lc_name  # noqa: E402
ROOT = os.path.join(vlib.V, "c10_lc"); MIN_MICE = 4; CALL_READS, CALL_COV = 20, 3


def slug(t):
    return "".join(c if c.isalnum() else "_" for c in t)


def selected(D, g, G, inA):
    elig = eligibility(D, g, G - 1, inA); keep = []
    for k in np.flatnonzero(elig):
        r = g == k
        if len(np.unique(D.mouse[r & inA])) >= MIN_MICE and len(np.unique(D.mouse[r & ~inA])) >= MIN_MICE: keep.append(k)
    return keep


def prepare(ds, design):
    D = Data(ds, 0); types, g = type_setup(D); G = len(types) + 1; inA = mouse_halves(D)
    ev = D.post["event_id"]; names = np.array([lc_name(j, ev[D.inv[i]]) for i, j in enumerate(D.post["junction_id"])])
    if design == "halfA":
        keep = selected(D, g, G, inA); use = inA
    else:
        n = pd.DataFrame(dict(t=g, m=D.mouse)).groupby("t").m.nunique(); keep = [k for k in range(G - 1) if n.get(k, 0) >= MIN_MICE]; use = np.ones(len(g), bool)
    out = os.path.join(ROOT, f"{ds}_{design}"); os.makedirs(out, exist_ok=True)
    mice = np.unique(D.mouse[use]); mi = {m: i for i, m in enumerate(mice)}; rows = np.flatnonzero(use); mc = np.array([mi[m] for m in D.mouse[rows]])
    Y = D.Y[rows].tocsr(); Mo = sp.csr_matrix((np.ones(len(rows)), (mc, np.arange(len(rows)))), shape=(len(mice), len(rows)))
    tot = np.asarray((Mo @ Y).todense()); jobs = []; rec = []
    for k in keep:
        s = g[rows] == k; T = np.asarray((Mo[:, np.flatnonzero(s)] @ Y[np.flatnonzero(s)]).todense()); n = np.bincount(mc[s], minlength=len(mice)); nr = np.bincount(mc[~s], minlength=len(mice))
        cols, grp, M = [], [], []
        for i, m in enumerate(mice):
            if n[i] > 0: cols.append(f"{m}_armT"); grp.append("armT"); M.append(T[i])
            if nr[i] > 0: cols.append(f"{m}_armR"); grp.append("armR"); M.append(tot[i] - T[i])
        M = np.vstack(M).T; ok = M.sum(1) > 0; dd = os.path.join(out, slug(types[k])); os.makedirs(dd, exist_ok=True)
        pd.DataFrame(M[ok].astype(int), index=names[ok], columns=cols).to_csv(os.path.join(dd, "counts.tsv.gz"), sep="\t", compression="gzip", index_label="chrom")
        open(os.path.join(dd, "groups.txt"), "w").write("".join(f"{c}\t{x}\n" for c, x in zip(cols, grp)))
        jobs.append(dd); rec.append(dict(type=types[k], code=int(k), dir=dd, samples_type=int((n > 0).sum()), samples_rest=int((nr > 0).sum()))); log(f"{types[k]}: {rec[-1]['samples_type']} vs {rec[-1]['samples_rest']} samples")
    open(os.path.join(out, "array_jobs.txt"), "w").write("\n".join(jobs) + "\n"); pd.DataFrame(rec).to_csv(os.path.join(out, "types.tsv"), sep="\t", index=False)
    log(f"{ds} {design}: {len(jobs)} types, {(G - 1) - len(jobs)} without {MIN_MICE} mice")


def boot(tbl, cols, B=2000):
    g = tbl.groupby("cell_type")[cols].sum(); k = tbl.groupby("cell_type").size().values; rng = np.random.default_rng(0); out = {}
    for c in cols:
        v = g[c].values; idx = rng.integers(0, len(v), (B, len(v))); r = v[idx].sum(1) / k[idx].sum(1)
        out[c] = [float(tbl[c].mean()), float(np.quantile(r, .025)), float(np.quantile(r, .975))]
    return out


def confirm(D, g, G, rB, K, E, lj, sgn, delta, inA):
    def rs(gg):
        y, n = raw_pb(D.Y, D.Ncell, D.inv, rB, gg[rB], G)
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.nan_to_num(y[:-1] / n[:-1]); b = np.nan_to_num((y.sum(0) - y[:-1]) / (n.sum(0) - n[:-1]))
        t = tv(a, b, D.inv, D.n_atse); s = np.sign(np.take_along_axis(a - b, np.maximum(lj, 0), axis=1))
        return (t[K, E] > delta / 2) & (s[K, E] == sgn)
    real = rs(g).astype(float); within = np.zeros(len(K)); free = np.zeros(len(K))
    for p in range(N_PERM):
        rng = np.random.default_rng(1000 + p); gp = g.copy()
        for m in np.unique(D.mouse[~inA]):
            i = np.flatnonzero((~inA) & (D.mouse == m)); gp[i] = rng.permutation(g[i])
        within += rs(gp)
    for p in range(5):
        rng = np.random.default_rng(2000 + p); gp = g.copy(); gp[rB] = rng.permutation(g[rB]); free += rs(gp)
    return real, within / N_PERM, free / 5


def collect(ds):
    D = Data(ds, 0); types, g = type_setup(D); G = len(types) + 1; inA = mouse_halves(D); rA, rB = np.flatnonzero(inA), np.flatnonzero(~inA)
    delta = json.load(open(os.path.join(vlib.V, "tsv", f"c10_{ds}_s0_delta.json")))["delta"]; out = os.path.join(ROOT, f"{ds}_halfA"); R = pd.read_csv(os.path.join(out, "types.tsv"), sep="\t")
    ev = D.post["event_id"]; jname = {lc_name(j, ev[D.inv[i]]): i for i, j in enumerate(D.post["junction_id"])}
    rd, cv = reads_cov(D, rA, g[rA], G); lj = np.full((G - 1, D.n_atse), -1); sg = np.zeros((G - 1, D.n_atse)); rows = []; st = []
    for x in R.itertuples():
        fs, fe = os.path.join(x.dir, "ds_cluster_significance.txt"), os.path.join(x.dir, "ds_effect_sizes.txt")
        if not (os.path.exists(fs) and os.path.exists(fe)): st.append(dict(type=x.type, tested=np.nan, sig=np.nan, calls=0)); continue
        c = pd.read_csv(fs, sep="\t"); c = c[c.status == "Success"]; e = pd.read_csv(fe, sep="\t"); dcol = [k for k in e.columns if k.startswith("deltapsi")][0]; assert dcol == "deltapsi_armT", dcol
        e["j"] = e.intron.map(jname); assert e.j.notna().all(); e["a"] = D.inv[e.j.astype(int)]; e["cl"] = e.intron.str.split(":").str[0] + ":" + e.intron.str.split(":").str[3]
        sigc = set(c[c["p.adjust"] < 0.05].cluster); e = e[e.cl.isin(sigc)]; n = 0
        for a, y in e.groupby("a"):
            if 0.5 * y[dcol].abs().sum() > delta and rd[x.code, a] >= CALL_READS and cv[x.code, a] >= CALL_COV:
                l = y.loc[y[dcol].abs().idxmax()]; lj[x.code, a] = int(l.j); sg[x.code, a] = np.sign(l[dcol]); rows.append((x.code, a)); n += 1
        st.append(dict(type=x.type, tested=len(c), sig=len(sigc), calls=n))
    K, E = np.array([r[0] for r in rows]), np.array([r[1] for r in rows]); real, w, f = confirm(D, g, G, rB, K, E, lj, sg[K, E], delta, inA)
    T = pd.DataFrame(dict(cell_type=np.array(types)[K], atse=E, real=real, null_within=w, null_free=f)); T["diff"] = T.real - T.null_within
    T.to_csv(os.path.join(out, "calls.tsv.gz"), sep="\t", index=False, float_format="%.4g"); pd.DataFrame(st).to_csv(os.path.join(out, "per_type.tsv"), sep="\t", index=False)
    # LeafletFA on the same types: the saved half-A calls, sign from half-A point estimates (lead asserted equal to the saved one)
    A = pd.read_csv(os.path.join(vlib.V, "c10_null_diag", f"{ds}_calls.tsv.gz"), sep="\t"); A = A[A.type_code.isin(R.code)]
    p, r, *_ = estimates(D, Groups(D, rA, g[rA], G), draws=False, seed=101, delta=delta); ljF, sgF = lead_sign(p[:-1], r[:-1], D.inv, D.n_atse)
    KF, EF = A.type_code.values, A.atse.values; assert (ljF[KF, EF] == A["lead"].values).all(), "lead differs from saved"
    rF, wF, fF = confirm(D, g, G, rB, KF, EF, ljF, sgF[KF, EF], delta, inA); assert np.allclose(rF, A.conf_ii.values), "real (ii) differs from saved"
    L = pd.DataFrame(dict(cell_type=A.cell_type.values, real=rF, null_within=wF, null_free=fF)); L["diff"] = L.real - L.null_within
    S = dict(dataset=ds, delta=delta, types=len(R), types_with_output=int(pd.DataFrame(st).tested.notna().sum()), tested=float(np.nansum([s["tested"] for s in st])),
             LeafCutter=dict(calls=len(T), types_with_calls=int(T.cell_type.nunique()), **boot(T, ["real", "null_within", "null_free", "diff"])),
             LeafletFA_same_types=dict(calls=len(L), types_with_calls=int(L.cell_type.nunique()), **boot(L, ["real", "null_within", "null_free", "diff"])))
    json.dump(S, open(os.path.join(out, "summary.json"), "w"), indent=1); print(json.dumps(S, indent=1))


if __name__ == "__main__":
    (prepare if sys.argv[1] == "prepare" else collect)(*sys.argv[2:])
