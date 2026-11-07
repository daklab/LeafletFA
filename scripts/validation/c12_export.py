"""Ship the check 12 (a)+(b) interval in the catalog export (Karin 2026-11-07: check 12 reading accepted; validated rows only).

    python c12_export.py <TMS|Allen>     # CPU, sbatch; -> $V/c12/export/catalog_<ds>_s0.tsv.gz and <ds>_intervals.npz

No new estimate and no new fit. Reads only saved outputs plus the TEST counts (for test_reads / test_cells on every row):
- c10/<ds>_s0_estimates.npz (points), c10/export/catalog_<ds>_s0.tsv.gz (the check 10 export, row order kept),
- portal_browser/precomputed/<ds>_train_tables.npz (TRAIN junction reads, ATSE reads, covered cells, mice),
- c12/<ds>_units.npz and tsv/c12_<ds>_tau2.tsv (check 12).

Interval (amendment 7, reading 2): m +/- 1.96 sqrt(v + tau^2_b), clipped to [0, 1], with m, v the mean and variance of the
(a) DM values over ALL TEST cells of the type (combined draws) and tau^2_b the check 12 value for the unit's TEST read bin
(identical whether calibrated on half A or B; asserted). It is a 95 % predictive interval for the pseudobulk psi of the
type's held-out TEST cells at their read depth. Rows outside the check 12 units (< 100 TEST reads or < 10 TEST cells
with reads at the ATSE) get NA and an interval_status saying why. The 'rest' interval has no validated version and is
not exported.
"""
import json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
import c10_catalog as C  # noqa: E402
from c12_intervals import binlab  # noqa: E402

V = vlib.V; T = os.path.join(V, "tsv"); OUT = os.path.join(V, "c12", "export")
PRE = os.path.join(V, "portal_browser", "precomputed")
MIN_READS, MIN_CELLS = 100, 10


def log(m):
    print(m, flush=True)


def ab(m, v, t2):
    s = 1.96 * np.sqrt(v + t2)
    return np.clip(m - s, 0, 1), np.clip(m + s, 0, 1)


def main(ds):
    os.makedirs(OUT, exist_ok=True)
    E = np.load(os.path.join(V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)
    Tt = np.load(os.path.join(PRE, f"{ds}_train_tables.npz"))
    U = np.load(os.path.join(V, "c12", f"{ds}_units.npz"))
    z = np.load(os.path.join(C.fit_dir(ds, 0), "fit.npz"), allow_pickle=True)
    types = E["types"].astype(str); assert np.array_equal(types, Tt["types"].astype(str))
    inv = np.asarray(z["inv_ev"], np.int64); jid = z["junction_id"].astype(str); NT, NJ, NE = len(types), len(inv), len(z["event_id"])
    assert E["point"].shape == (NT, NJ) and Tt["ereads"].shape == (NT, NE)

    # ---- tau^2: one value per read bin, the same in both calibration directions ----
    tz = pd.read_csv(os.path.join(T, f"c12_{ds}_tau2.tsv"), sep="\t"); tz = tz[tz.draws == "combined"]
    tau2 = {}
    for b, g in tz.groupby("test_read_bin"):
        assert g.tau2.nunique() == 1 and g.reached_95.all(), (b, g)
        tau2[b] = float(g.tau2.iloc[0])
    log(f"tau2 {tau2}")

    # ---- reproduce the check 12 cross-half (a)+(b) coverage from the saved arrays (guards the formula) ----
    cv = pd.read_csv(os.path.join(T, f"c12_{ds}_coverage.tsv"), sep="\t")
    for cal, ev in (("A", "B"), ("B", "A")):
        q = U[f"qual_{ev}"]; rb = binlab(U[f"N_{ev}"])
        for b in ("100-499", "500-inf"):
            msk = q & (rb == b)
            lo, hi = ab(U[f"dmc_{ev}_m"][msk].astype(np.float64), U[f"dmc_{ev}_v"][msk].astype(np.float64), tau2[b])
            o = U[f"obs_{ev}"][msk]; c = float(((o >= lo) & (o <= hi)).mean())
            ref = cv[(cv.draws == "combined") & (cv.method == f"(a)+(b), calibrated on {cal}") & (cv.evaluated_on == ev) & (cv.test_read_bin == b)]
            assert len(ref) == 1 and int(ref.junctions.iloc[0]) == int(msk.sum()) and abs(float(ref.coverage.iloc[0]) - c) < 1e-12, (cal, b, c, ref)
    log("check 12 cross-half (a)+(b) coverage reproduced exactly")

    # ---- the shipped interval: all TEST cells ----
    k, j, e = U["k"], U["j"], U["e"]
    assert np.array_equal(inv[j], e)
    assert np.allclose(U["lo10"], E["lo"][k, j], atol=1e-6) and np.allclose(U["hi10"], E["hi"][k, j], atol=1e-6), "units not in the estimates' index"
    rb = binlab(U["N_all"]); assert set(np.unique(rb)) <= {"100-499", "500-inf"}
    m, v = U["dmc_all_m"].astype(np.float64), U["dmc_all_v"].astype(np.float64)
    assert np.isfinite(m).all() and np.isfinite(v).all()
    t2 = np.where(rb == "100-499", tau2["100-499"], tau2["500-inf"])
    lo_u, hi_u = ab(m, v, t2)
    LO = np.full((NT, NJ), np.nan, np.float32); HI = np.full((NT, NJ), np.nan, np.float32)
    LO[k, j], HI[k, j] = lo_u, hi_u
    o = U["obs_all"]; desc = {}
    for b in ("100-499", "500-inf"):
        msk = rb == b
        desc[b] = dict(junctions=int(msk.sum()), coverage_all_test=float(((o[msk] >= lo_u[msk]) & (o[msk] <= hi_u[msk])).mean()),
                       median_width=float(np.median(hi_u[msk] - lo_u[msk])))
    log(f"descriptive, shipped interval vs all TEST cells (tau2 from halves, so not out-of-sample): {desc}")

    # ---- TEST reads and cells with reads per (type, ATSE), every row ----
    P = vlib.paths(ds)
    _, tlab, _ = C.labels(ds, P["test"])
    Y = vlib.read_counts(P["test"]).tocsr(); assert Y.shape[1] == NJ
    Em = sp.csr_matrix((np.ones(NJ), (np.arange(NJ), inv)), shape=(NJ, NE))
    N = (Y @ Em).tocsr(); code = {x: i for i, x in enumerate(types)}
    g = np.array([code.get(x, NT) for x in tlab]); O = C.onehot(g, NT + 1)
    TR = np.asarray((O @ N).todense())[:NT]; TC = np.asarray((O @ (N > 0).astype(np.float64)).todense())[:NT]
    assert np.array_equal(TR[k, e], U["N_all"]), "TEST reads differ from check 12"
    unit_te = np.zeros((NT, NE), bool); unit_te[k, e] = True
    assert np.array_equal(unit_te, (TR >= MIN_READS) & (TC >= MIN_CELLS)), "check 12 units are not exactly the >=100 reads, >=10 cells rule"
    log(f"TEST reads match check 12; units = rule ({int(unit_te.sum()):,} type x ATSE)")

    # ---- new export: the check 10 export with the interval replaced and columns added ----
    X = pd.read_csv(os.path.join(V, "c10", "export", f"catalog_{ds}_s0.tsv.gz"), sep="\t", low_memory=False)
    ti = X.cell_type.map(code).values; jx = {x: i for i, x in enumerate(jid)}; ji = X.junction_id.map(jx).values
    assert np.isfinite(ti.astype(float)).all() and np.isfinite(ji.astype(float)).all()
    ti, ji = ti.astype(int), ji.astype(int); ei = inv[ji]
    assert (np.abs(X.psi.values - E["point"][ti, ji]) <= 5e-4 * np.abs(E["point"][ti, ji]) + 1e-9).all()  # %.4g export
    assert np.array_equal(X.reads.values, Tt["ereads"][ti, ei].astype(np.int64))
    assert np.array_equal(X.cells_covering.values, Tt["cov"][ti, ei].astype(np.int64))
    jr = Tt["jreads"][ti, ji].astype(np.int64); er = Tt["ereads"][ti, ei].astype(np.float64)
    tr, tc = TR[ti, ei].astype(np.int64), TC[ti, ei].astype(np.int64)
    status = np.where(tr < MIN_READS, f"no interval: fewer than {MIN_READS} TEST reads",
                      np.where(tc < MIN_CELLS, f"no interval: fewer than {MIN_CELLS} TEST cells with reads", "check 12 (a)+(b)"))
    lo_r, hi_r = LO[ti, ji], HI[ti, ji]
    assert np.array_equal(np.isfinite(lo_r), status == "check 12 (a)+(b)")
    cols = list(X.columns)
    X["psi_lo"], X["psi_hi"] = lo_r, hi_r
    X["interval_status"] = status
    X["test_reads"], X["test_cells"] = tr, tc
    X["raw_psi"] = np.where(er > 0, jr / np.where(er > 0, er, 1), np.nan)
    X["junction_reads"] = jr
    X["mice"] = Tt["mice"][ti, ei].astype(np.int64)
    X["source"] = "LeafletFA point (check 10); interval check 12 (a)+(b), predictive for held-out TEST cells"
    order = cols[:cols.index("psi_hi") + 1] + ["interval_status", "test_reads", "test_cells", "raw_psi", "junction_reads"] + \
        [c for c in cols[cols.index("psi_hi") + 1:] if c != "source"] + ["mice", "source"]
    # Karin 2026-11-07: the check 10.2 call columns (failed check) are not shipped; they stay in the internal check 10 export.
    order = [c for c in order if c not in ("tv", "specific", "call_eligible")]
    X = X[order].rename(columns={"reads": "atse_reads"})
    path = os.path.join(OUT, f"catalog_{ds}_s0.tsv.gz")
    X.to_csv(path, sep="\t", index=False, float_format="%.4g")
    n_unit_rows = int(np.isfinite(lo_r).sum())
    units_no_train = int(unit_te.sum() - len(np.unique(ti[np.isfinite(lo_r)] * NE + ei[np.isfinite(lo_r)])))
    np.savez_compressed(os.path.join(OUT, f"{ds}_intervals.npz"), types=types, lo=LO, hi=HI,
                        test_reads=TR.astype(np.uint32), test_cells=TC.astype(np.uint16), tau2=json.dumps(tau2))
    rec = dict(dataset=ds, rows=len(X), rows_with_interval=n_unit_rows, rows_with_interval_frac=n_unit_rows / len(X),
               status_counts=pd.Series(status).value_counts().to_dict(), unit_atses_without_train_reads=units_no_train,
               tau2=tau2, descriptive_all_test=desc, columns=list(X.columns), md5=dict(export=C_md5(path)))
    json.dump(rec, open(os.path.join(OUT, f"{ds}_record.json"), "w"), indent=1)
    log(json.dumps(rec, indent=1))


def C_md5(p):
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    main(sys.argv[1])
