"""Check 15b differential arm (amendment 11), LeafCutter: as c15_leafcutter.py on $V/c15b/diff, plus per-type AUROC and
LeafletFA - LeafCutter AUROC differences (pooled and per type) with type-bootstrap CIs. Original docstring follows.
Check 15, Measure 2, LeafCutter arm (CPU; amendment 10). Per scored type with >= 4 mice having cells in both halves:
half A (baseline) vs half B on per-mouse pseudobulks of the simulated counts; leafcutter_ds settings as check 10
(--min_samples_per_intron 4 --min_samples_per_group 4). Score = -log10 p of the ATSE's cluster (status Success).
    python c15b_leafcutter.py prepare <full|f30|f10>    -> $V/c15/lc_<D>/<type>/{counts.tsv.gz,groups.txt}, array_jobs.txt
    python c15b_leafcutter.py collect <full|f30|f10>    -> $V/c15/score/<D>_m2_lc.tsv, <D>_m2_lc_summary.json
Submission: c15_lc_submit.sh <D> <array spec>."""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
from c10_catalog import labels  # noqa: E402
from c9_ds_null import lc_name  # noqa: E402
from c10_leafcutter import slug  # noqa: E402

C15 = os.path.join(vlib.V, "c15b", "diff"); C15A = os.path.join(vlib.V, "c15"); RB = [-0.5, 0.5, 19.5, 99.5, np.inf]; RL = ["0", "1-19", "20-99", ">=100"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def setup():
    tr = np.load(os.path.join(C15, "truth.npz"), allow_pickle=True)
    types = tr["types"].astype(str); T = len(types); g = tr["g"]; half_b = tr["half_b"]
    _, _, mouse = labels("TMS", vlib.paths("TMS")["train"])
    mm = pd.DataFrame(dict(g=g, m=mouse, b=half_b))[g < T].groupby(["g", "m"]).b.agg(["min", "max"])
    both = mm[(mm["min"] == 0) & (mm["max"] == 1)].reset_index().groupby("g").size().reindex(range(T), fill_value=0).to_numpy()
    return tr, types, g, half_b, mouse, np.flatnonzero(both >= 4)


def prepare(D):
    tr, types, g, half_b, mouse, keep = setup()
    var = pd.read_csv(os.path.join(C15A, "var.tsv"), sep="\t")
    names = np.array([lc_name(j, e) for j, e in zip(var.junction_id, var.event_id)])
    Y = sp.load_npz(os.path.join(C15, f"sim_{D}_train.npz")).tocsr()
    out = os.path.join(C15, f"lc_{D}"); os.makedirs(out, exist_ok=True); jobs, rec = [], []
    for k in keep:
        r = np.flatnonzero(g == k); cols, grp, M = [], [], []
        for m in np.unique(mouse[r]):
            for arm, hb in (("armA", False), ("armB", True)):
                i = r[(mouse[r] == m) & (half_b[r] == hb)]
                if len(i):
                    cols.append(f"{m}_{arm}"); grp.append(arm); M.append(np.asarray(Y[i].sum(0)).ravel())
        M = np.vstack(M).T; ok = M.sum(1) > 0; dd = os.path.join(out, slug(types[k])); os.makedirs(dd, exist_ok=True)
        pd.DataFrame(M[ok].astype(int), index=names[ok], columns=cols).to_csv(os.path.join(dd, "counts.tsv.gz"), sep="\t", compression="gzip", index_label="chrom")
        open(os.path.join(dd, "groups.txt"), "w").write("".join(f"{c}\t{x}\n" for c, x in zip(cols, grp)))
        jobs.append(dd); rec.append(dict(type=types[k], code=int(k), dir=dd, samples_A=grp.count("armA"), samples_B=grp.count("armB")))
    open(os.path.join(out, "array_jobs.txt"), "w").write("\n".join(jobs) + "\n"); pd.DataFrame(rec).to_csv(os.path.join(out, "types.tsv"), sep="\t", index=False)
    log(f"{D}: {len(jobs)} types written")


def auc_prec(iu, nu, y, w):
    wp = np.bincount(iu, weights=w * y, minlength=nu); wn = np.bincount(iu, weights=w * ~y, minlength=nu)
    Wp, Wn = wp.sum(), wn.sum()
    if Wp == 0 or Wn == 0:
        return np.nan, np.nan
    cn = np.cumsum(wn) - wn; auc = (wp * (cn + 0.5 * wn)).sum() / (Wp * Wn)
    cp_d, ca_d = np.cumsum(wp[::-1]), np.cumsum((wp + wn)[::-1]); i = np.searchsorted(cp_d, 0.5 * Wp)
    return auc, cp_d[i] / ca_d[i]


def collect(D):
    tr, types, *_ = setup(); ev = pd.read_csv(os.path.join(C15A, "var.tsv"), sep="\t").event_id.to_numpy()
    u, inv = vlib.event_index(ev); eid = {e: i for i, e in enumerate(u)}
    assert np.array_equal(inv, tr["inv"]), "ATSE order differs from truth.npz"
    U = dict(np.load(os.path.join(C15, "score", f"{D}_m2_units.npz")))
    lc = np.full(len(U["type"]), np.nan); R = pd.read_csv(os.path.join(C15, f"lc_{D}", "types.tsv"), sep="\t"); st = []
    key = U["type"].astype(np.int64) * len(eid) + U["atse"]; order = np.argsort(key); ks = key[order]
    for x in R.itertuples():
        f = os.path.join(x.dir, "ds_cluster_significance.txt")
        if not os.path.exists(f):
            st.append(dict(type=x.type, output=False)); continue
        c = pd.read_csv(f, sep="\t"); c = c[c.status == "Success"]
        a = c.cluster.str.split(":", n=1).str[1].str.rsplit("_", n=1).str[0].map(eid)
        assert a.notna().all(); q = x.code * len(eid) + a.astype(np.int64).to_numpy()
        pos = np.searchsorted(ks, q); hit = (pos < len(ks)) & (ks[np.minimum(pos, len(ks) - 1)] == q)
        lc[order[pos[hit]]] = -np.log10(np.maximum(c.p.to_numpy()[hit], 1e-300))
        st.append(dict(type=x.type, output=True, tested=len(c), matched=int(hit.sum())))
    both = np.isfinite(lc); rb = pd.cut(U["reads"], RB, labels=RL).astype(str); rows = []
    sel = [("all", "all", np.ones(len(lc), bool))] + [("delta", str(d), ~U["label"] | np.isclose(U["delta"], d)) for d in (0.1, 0.3, 0.5)] \
        + [("reads", l, rb == l) for l in RL] + [("own_truth", "own", U["own"])]
    for by, lev, m in sel:
        cov = float(both[m].mean()); m2 = m & both; tp = U["type"][m2]; yv = U["label"][m2]; ut, it = np.unique(tp, return_inverse=True)
        r = dict(depth=D, by=by, bin=lev, units=int(m.sum()), lc_coverage=cov, units_both=int(m2.sum()), planted_both=int(yv.sum()), types=len(ut))
        if len(ut):
            aucs = {}; rng = np.random.default_rng(0); W = np.stack([np.bincount(rng.integers(0, len(ut), len(ut)), minlength=len(ut)) for _ in range(1000)])
            for nm, sc in (("lc", lc[m2]), ("lfa", U["s_lfa"][m2]), ("raw", U["s_raw"][m2])):
                u, iu = np.unique(sc, return_inverse=True); a, p = auc_prec(iu, len(u), yv, np.ones(len(iu)))
                bs = np.array([auc_prec(iu, len(u), yv, W[b][it].astype(np.float64)) for b in range(1000)])
                aucs[nm] = (iu, len(u))
                r.update({f"auroc_{nm}": a, f"auroc_{nm}_lo": np.nanquantile(bs[:, 0], .025), f"auroc_{nm}_hi": np.nanquantile(bs[:, 0], .975),
                          f"prec50_{nm}": p, f"prec50_{nm}_lo": np.nanquantile(bs[:, 1], .025), f"prec50_{nm}_hi": np.nanquantile(bs[:, 1], .975)})
            # amendment 11: per-type AUROC on shared units, and per-type delta (LeafletFA - LeafCutter) with a type bootstrap
            pt = {nm: [] for nm in ("lc", "lfa")}
            for j in range(len(ut)):
                mj = it == j
                for nm, sc in (("lc", lc[m2]), ("lfa", U["s_lfa"][m2])):
                    if 0 < yv[mj].sum() < mj.sum():
                        uj, iuj = np.unique(sc[mj], return_inverse=True); pt[nm].append(auc_prec(iuj, len(uj), yv[mj], np.ones(mj.sum()))[0])
                    else:
                        pt[nm].append(np.nan)
            a_lc, a_lf = np.array(pt["lc"]), np.array(pt["lfa"]); ok = np.isfinite(a_lc) & np.isfinite(a_lf); dd = a_lf[ok] - a_lc[ok]
            if ok.any():
                rb_ = np.random.default_rng(0); bm = [dd[rb_.integers(0, len(dd), len(dd))].mean() for _ in range(1000)]
                r.update(pertype_n=int(ok.sum()), pertype_lc_mean=float(a_lc[ok].mean()), pertype_lfa_mean=float(a_lf[ok].mean()),
                         pertype_lc_median=float(np.median(a_lc[ok])), pertype_lfa_median=float(np.median(a_lf[ok])),
                         pertype_d_lfa_lc=float(dd.mean()), pertype_d_lo=float(np.quantile(bm, .025)), pertype_d_hi=float(np.quantile(bm, .975)))
            dpool = r.get("auroc_lfa", np.nan) - r.get("auroc_lc", np.nan); bsd = []
            if "lc" in aucs:
                for b in range(1000):
                    w = W[b][it].astype(np.float64)
                    bsd.append(auc_prec(*aucs["lfa"], yv, w)[0] - auc_prec(*aucs["lc"], yv, w)[0])
                r.update(pooled_d_lfa_lc=dpool, pooled_d_lo=float(np.nanquantile(bsd, .025)), pooled_d_hi=float(np.nanquantile(bsd, .975)))
        rows.append(r); log(f"{by} {lev}: coverage {cov:.3f}")
    M = pd.DataFrame(rows); M.to_csv(os.path.join(C15, "score", f"{D}_m2_lc.tsv"), sep="\t", index=False)
    json.dump(dict(depth=D, types=len(R), per_type=st), open(os.path.join(C15, "score", f"{D}_m2_lc_summary.json"), "w"), indent=1, default=int)
    print(M.round(3).to_string(index=False))


if __name__ == "__main__":
    (prepare if sys.argv[1] == "prepare" else collect)(sys.argv[2])
