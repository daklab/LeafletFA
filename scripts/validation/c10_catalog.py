"""Check 10 (CPU): the cell-type-resolved splicing catalog (pre-registration amendment 6).

    python c10_catalog.py full  --dataset TMS --seed 0     # 10.1 accuracy, 10.2 delta / calls / mouse halves / null, 10.3, 10.4, 10.5 export
    python c10_catalog.py calls --dataset TMS --seed 3     # call set only (all TRAIN cells), for 10.2c across seeds
    python c10_catalog.py seeds --dataset TMS              # 10.2c Jaccard over the seeds' call sets

Estimator. Per cell, the posterior-mean usage rho~_cj = (g_e rho_cj + y_cj) / (g_e + N_ce), rho = phi . psi; cell-type psi =
mean of rho~ over the type's TRAIN cells. Because rho~ = w_ce rho_cj + y_cj / (g_e + N_ce) with w_ce = 1 - v_ce,
v_ce = N_ce / (g_e + N_ce) (0 where the cell has no reads), the type sum is
    sum_k psi_kj [ sum_c phi_ck - sum_c v_ce phi_ck ] + sum_c y_cj / (g_e + N_ce),
so each draw needs one sparse product over observed (cell, ATSE) pairs. Point estimate: the fit's phi and psi_mean.
Interval: 2.5-97.5 % over 200 joint draws of q(psi) and q(phi) (checkpoint), g_e at its fitted value.
Comparator: raw TRAIN pseudobulk, falling back to the TRAIN population psi where the type has 0 reads at the event.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, c9_lib as L  # noqa: E402
from torch.distributions import biject_to, constraints  # noqa: E402

S_DRAWS, MIN_TRAIN, MIN_TEST_CELLS, MIN_TEST_READS, MIN_TEST_COV = int(os.environ.get("C10_DRAWS", 200)), 10, 5, 20, 3
CALL_READS, CALL_COV, N_PERM = 20, 3, int(os.environ.get("C10_PERM", 20))
OUT = os.environ.get("C10_OUT", os.path.join(vlib.V, "c10"))   # smoke tests redirect OUT and T
T = os.environ.get("C10_T", os.path.join(vlib.V, "tsv"))
EPI = ("EPITHELIAL", "KERATINOCYTE", "BASAL", "HEPATOCYTE", "DUCTAL", "INTESTINAL", "ISLET", "KIDNEY CELL", "LUMINAL")
MES = ("FIBROBLAST", "MESENCHYMAL", "STROMAL", "SMOOTH MUSCLE", "PERICYTE", "SKELETAL MUSCLE", "CARDIAC MUSCLE", "VLMC")
FGFR2_IIIB = "chr7_130199905_130200982_-"
LYMPHOID = ("B CELL", "T CELL", "NK CELL")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def labels(ds, path):
    with h5py.File(path, "r") as f:
        o = f["obs"]
        if ds == "TMS":
            t = vlib._strings(o["tissue_cell_type"]).astype(str)
        else:   # Allen subclass as in SpliceVI STAGE5 s111 (cluster number removed); pending Karin's confirmation
            c = vlib._strings(o["cell_ontology_class"]).astype(str)
            t = np.array([x.split("_", 1)[1] if "_" in x else x for x in c])
        mouse = vlib._strings(o["mouse.id"]).astype(str)
        cells = vlib._strings(o[o.attrs["_index"]]).astype(str)
    return cells, t, mouse


class Data:
    def __init__(self, ds, seed):
        P = vlib.paths(ds)
        self.post = L.load_posterior(seed) if ds == "TMS" else load_posterior_ds(ds, seed)
        self.cells, self.t, self.mouse = labels(ds, P["train"])
        assert np.array_equal(self.post["cell_id"], self.cells)
        self.inv = self.post["inv"].numpy(); self.n_atse = self.post["n_atse"]; self.J = len(self.inv)
        z = np.load(os.path.join(fit_dir(ds, seed), "fit.npz"), allow_pickle=True)
        self.gam = np.asarray(z["bb_conc_vec"], np.float64)
        self.Y = vlib.read_counts(P["train"]).tocsr()
        E = sp.csr_matrix((np.ones(self.J), (np.arange(self.J), self.inv)), shape=(self.J, self.n_atse))
        self.Ncell = (self.Y @ E).tocsr()                                         # cells x ATSE
        Nc = self.Ncell.tocoo()
        self.V = sp.csr_matrix((Nc.data / (self.gam[Nc.col] + Nc.data), (Nc.row, Nc.col)), shape=self.Ncell.shape)
        Yc = self.Y.tocoo()
        Ncj = np.asarray(self.Ncell[Yc.row, self.inv[Yc.col]]).ravel()
        self.R = sp.csr_matrix((Yc.data / (self.gam[self.inv[Yc.col]] + Ncj), (Yc.row, Yc.col)), shape=self.Y.shape)
        self.test = None

    def load_test(self, ds):
        P = vlib.paths(ds)
        cells, t, mouse = labels(ds, P["test"])
        Y = vlib.read_counts(P["test"]).tocsr()
        E = sp.csr_matrix((np.ones(self.J), (np.arange(self.J), self.inv)), shape=(self.J, self.n_atse))
        self.test = dict(t=t, Y=Y, N=(Y @ E).tocsr())


def fit_dir(ds, seed):
    return L.fit_dir(seed) if ds == "TMS" else os.path.join(vlib.B, ds, f"full_K20_s{seed}_f64", "fit")


def load_posterior_ds(ds, seed):
    d = fit_dir(ds, seed)
    old = L.fit_dir
    L.fit_dir = lambda s: d
    try:
        return L.load_posterior(seed)
    finally:
        L.fit_dir = old


def onehot(g, G):
    return sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))


class Groups:
    """precomputed per-group blocks for fast repeated group sums over draws; mouse codes for the bootstrap (note 3)"""
    def __init__(self, D, rows, g, G):
        self.rows, self.g, self.G = rows, g, G
        self.n = np.bincount(g, minlength=G).astype(np.float64)
        O = onehot(g, G)
        self.O = O
        self.R = D.R[rows].tocsr()
        self.Rs = np.asarray((O @ self.R).todense())                             # G x J, read term (draw-free)
        Vr = D.V[rows]
        self.idx = [np.flatnonzero(g == k) for k in range(G)]
        self.VT = [Vr[i].T.tocsr() for i in self.idx]                          # ATSE x cells_k
        mouse = D.mouse[rows]
        self.mcode = []
        for i in self.idx:
            _, c = np.unique(mouse[i], return_inverse=True); self.mcode.append(c)
        self.n_mice = np.array([c.max() + 1 if len(c) else 0 for c in self.mcode])

    def boot_weights(self, rng):
        """per-row weights: each group's mice resampled with replacement; a cell's weight = draws of its mouse"""
        w = np.zeros(len(self.rows))
        for i, c in zip(self.idx, self.mcode):
            if len(c):
                w[i] = np.bincount(rng.integers(0, c.max() + 1, c.max() + 1), minlength=c.max() + 1)[c]
        return w

    def sums(self, D, phi, psi, w=None):
        """(G x J) (weighted) sums of rho~ over each group's cells, and the weighted cell counts (G,)"""
        if w is None:
            Phis = self.O @ phi; Rs = self.Rs; pw = phi; nw = self.n
        else:
            pw = phi * w[:, None]; Phis = self.O @ pw
            Rs = np.asarray((self.O @ sp.diags(w) @ self.R).todense()); nw = self.O @ w
        out = np.empty((self.G, D.J), np.float32)
        for k in range(self.G):
            A = Phis[k][None, :] - self.VT[k] @ pw[self.idx[k]]                # ATSE x K
            out[k] = np.einsum("jk,kj->j", A[D.inv], psi) + Rs[k]
        return out, nw


def phi_draw(D, gen):
    sb = biject_to(constraints.simplex)
    loc, sc = D.post["phi_loc"], D.post["phi_scale"]
    return sb(loc + sc * torch.randn(loc.shape, generator=gen, dtype=loc.dtype)).numpy()


def psi_draw(D, gen):
    loc, sc = D.post["psi_loc"], D.post["psi_scale"]
    return L.segment_softmax(loc + sc * torch.randn(loc.shape, generator=gen), D.post["inv"], D.n_atse).numpy().astype(np.float64)


def estimates(D, Gr, draws=True, seed=0, delta=None, pairs=()):
    """Point means (G x J) and leave-one-group-out means from the fit's phi and psi (all cells, unweighted).
    With draws (amendment 6 notes 2-3): each of S_DRAWS draws is a joint q(phi), q(psi) draw combined with a mouse bootstrap
    within each group. Returns intervals of the group and rest means and P(TV > delta) (group vs rest; listed pairs) from
    these combined draws, plus parameter-only intervals (same q draws, no bootstrap) for the coverage comparison.
    Returns point, rest, iv, ivr, P, Pp, iv_param."""
    n = Gr.n; tot_n = n.sum()
    S, _ = Gr.sums(D, D.post["phi"][Gr.rows], D.post["psi_mean"])
    point = S / n[:, None]
    rest = (S.sum(0)[None, :] - S) / (tot_n - n)[:, None]
    if not draws:
        return point, rest, None, None, None, None, None
    gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed + 7)
    Sb = np.empty((S_DRAWS, Gr.G, D.J), np.float32); Sp = np.empty_like(Sb); Nb = np.empty((S_DRAWS, Gr.G))
    for s in range(S_DRAWS):
        ph, ps = phi_draw(D, gen)[Gr.rows], psi_draw(D, gen)
        Sp[s], _ = Gr.sums(D, ph, ps)
        Sb[s], Nb[s] = Gr.sums(D, ph, ps, Gr.boot_weights(rng))
        if s % 50 == 0:
            log(f"  draw {s}/{S_DRAWS}")
    tot_b, ntot_b = Sb.sum(1), Nb.sum(1)                                      # S x J, S
    lo = np.empty((Gr.G, D.J), np.float32); hi = np.empty_like(lo); lo_r = np.empty_like(lo); hi_r = np.empty_like(lo)
    lo_p = np.empty_like(lo); hi_p = np.empty_like(lo)
    P = np.zeros((Gr.G, D.n_atse), np.float32)
    for k in range(Gr.G):
        nk = np.maximum(Nb[:, k], 1e-12)[:, None]
        x = Sb[:, k, :] / nk; r = (tot_b - Sb[:, k, :]) / (ntot_b - Nb[:, k])[:, None]
        lo[k], hi[k] = np.quantile(x, [0.025, 0.975], axis=0)
        lo_r[k], hi_r[k] = np.quantile(r, [0.025, 0.975], axis=0)
        lo_p[k], hi_p[k] = np.quantile(Sp[:, k, :] / n[k], [0.025, 0.975], axis=0)
        if delta is not None:
            P[k] = (tv(x, r, D.inv, D.n_atse) > delta).mean(0)
    Pp = {}
    for (i, j) in pairs:
        Pp[(i, j)] = (tv(Sb[:, i, :] / np.maximum(Nb[:, i], 1e-12)[:, None], Sb[:, j, :] / np.maximum(Nb[:, j], 1e-12)[:, None],
                         D.inv, D.n_atse) > delta).mean(0)
    del Sb, Sp
    return point, rest, (lo, hi), (lo_r, hi_r), P, Pp, (lo_p, hi_p)


def tv(a, b, inv, n_atse):
    """[G x J] -> [G x ATSE] total variation"""
    d = np.abs(a - b)
    out = np.zeros((a.shape[0], n_atse))
    np.add.at(out.T, inv, d.T)
    return 0.5 * out


def raw_pb(Y, N, inv, rows, g, G):
    O = onehot(g, G)
    y = np.asarray((O @ Y[rows]).todense()); n = np.asarray((O @ N[rows]).todense())[:, inv]
    return y, n


def call(D, point, rest, P, reads, cov, delta):
    """amendment 6 note 2: point TV > delta, P(TV > delta) >= 0.95 over the draws, and the read / cell condition"""
    T_ = tv(point, rest, D.inv, D.n_atse)
    return (T_ > delta) & (P >= 0.95) & (reads >= CALL_READS) & (cov >= CALL_COV), T_


def within_mouse_halves(rows_by_type, mouse, seed):
    rng = np.random.default_rng(seed); a, b = [], []
    for rows in rows_by_type:
        for m in np.unique(mouse[rows]):
            r = rng.permutation(rows[mouse[rows] == m]); h = (len(r) + 1) // 2
            a.append(r[:h]); b.append(r[h:])
    return np.concatenate(a), np.concatenate(b)


def lead_sign(a, b, inv, n_atse):
    """per (group, ATSE): index of the junction with the largest |a - b| and the sign of a - b there"""
    d = a - b
    G = a.shape[0]; lj = np.full((G, n_atse), -1); best = np.full((G, n_atse), -1.0)
    ad = np.abs(d)
    for j in range(a.shape[1]):                                                   # ATSE sizes are small; vectorize over groups
        e = inv[j]; m = ad[:, j] > best[:, e]
        best[m, e] = ad[m, j]; lj[m, e] = j
    sign = np.sign(np.take_along_axis(d, np.maximum(lj, 0), axis=1))
    return lj, sign


def type_setup(D):
    n = pd.Series(D.t).value_counts()
    types = sorted(n[n >= MIN_TRAIN].index)
    code = {t: i for i, t in enumerate(types)}
    g = np.array([code.get(t, len(types)) for t in D.t])                          # last group = all other cells
    return types, g


def reads_cov(D, rows, g, G):
    O = onehot(g, G)
    return np.asarray((O @ D.Ncell[rows]).todense()), np.asarray((O @ (D.Ncell[rows] > 0).astype(np.float64)).todense())


def run_full(ds, seed):
    os.makedirs(OUT, exist_ok=True); os.makedirs(T, exist_ok=True)
    tag = f"{ds}_s{seed}"
    D = Data(ds, seed); D.load_test(ds); log(f"{ds}: {len(D.t):,} TRAIN cells, {D.J:,} junctions")
    types, g = type_setup(D); G = len(types) + 1; allrows = np.arange(len(D.t))
    Gr = Groups(D, allrows, g, G)
    reads, cov = reads_cov(D, allrows, g, G)
    # ---- 10.2 delta first (amendment 6 note 2): within-type split halves, point estimates ----
    a_rows, b_rows = within_mouse_halves([np.flatnonzero(g == k) for k in range(G - 1)], D.mouse, 0)
    hr = np.concatenate([a_rows, b_rows]); hg = np.concatenate([g[a_rows] * 2, g[b_rows] * 2 + 1])
    ph = estimates(D, Groups(D, hr, hg, 2 * (G - 1)), draws=False)[0]
    half_tv = tv(ph[0::2], ph[1::2], D.inv, D.n_atse)
    ok = (reads[:-1] >= CALL_READS) & (cov[:-1] >= CALL_COV)
    delta = float(np.ceil(np.quantile(half_tv[ok], 0.99) / 0.05) * 0.05)
    json.dump(dict(dataset=ds, delta=delta, q99=float(np.quantile(half_tv[ok], 0.99)), units=int(ok.sum())),
              open(os.path.join(T, f"c10_{tag}_delta.json"), "w"))
    log(f"delta = {delta}")
    point, rest, iv, ivr, P, _, ivp = estimates(D, Gr, draws=True, seed=seed, delta=delta)
    # raw TRAIN pseudobulk with population fallback
    yr, nr = raw_pb(D.Y, D.Ncell, D.inv, allrows, g, G)
    pop = yr.sum(0) / np.maximum(nr.sum(0), 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(nr > 0, yr / np.maximum(nr, 1e-12), pop[None, :])
    # ---- 10.1 accuracy vs TEST pseudobulk; 10.3 interval coverage ----
    te = D.test; code = {t: i for i, t in enumerate(types)}
    gt = np.array([code.get(t, len(types)) for t in te["t"]]); trows = np.arange(len(gt))
    yt, nt = raw_pb(te["Y"], te["N"], D.inv, trows, gt, G)
    Ot = onehot(gt, G); nte_cells = np.bincount(gt, minlength=G)
    te_reads = np.asarray((Ot @ te["N"]).todense()); te_cov = np.asarray((Ot @ (te["N"] > 0).astype(np.float64)).todense())
    with np.errstate(invalid="ignore", divide="ignore"):
        tpsi = yt / nt
    err_l = tv(point, np.nan_to_num(tpsi), D.inv, D.n_atse); err_r = tv(raw, np.nan_to_num(tpsi), D.inv, D.n_atse)
    rows = []
    for k, t in enumerate(types):
        if nte_cells[k] < MIN_TEST_CELLS:
            continue
        u = np.flatnonzero((te_reads[k] >= MIN_TEST_READS) & (te_cov[k] >= MIN_TEST_COV))
        rows.append(pd.DataFrame(dict(cell_type=t, atse=u, train_cells=int(Gr.n[k]), train_reads=reads[k, u],
                                      test_reads=te_reads[k, u], err_leafletfa=err_l[k, u], err_raw=err_r[k, u])))
    U = pd.concat(rows, ignore_index=True)
    U["size_bin"] = pd.cut(U.train_cells, [9, 49, 199, 999, np.inf], labels=["10-49", "50-199", "200-999", ">=1000"])
    U["read_bin"] = pd.cut(U.train_reads, [-1, 0, 19, 99, np.inf], labels=["0", "1-19", "20-99", ">=100"])
    U["diff"] = U.err_leafletfa - U.err_raw
    U.to_csv(os.path.join(OUT, f"{tag}_accuracy_units.tsv.gz"), sep="\t", index=False)
    rng = np.random.default_rng(0)

    def boot(sub):
        per = sub.groupby("cell_type")["diff"].agg(["sum", "size"])
        b = [per.iloc[rng.integers(0, len(per), len(per))] for _ in range(1000)]
        v = [x["sum"].sum() / x["size"].sum() for x in b]
        return np.quantile(v, [0.025, 0.975])
    acc = []
    for kind, col in (("all", None), ("size", "size_bin"), ("reads", "read_bin")):
        for lev, sub in ([("all", U)] if col is None else U.groupby(col, observed=True)):
            if len(sub) == 0:
                continue
            lo_, hi_ = boot(sub)
            acc.append(dict(dataset=ds, by=kind, bin=str(lev), units=len(sub), types=sub.cell_type.nunique(),
                            err_leafletfa=sub.err_leafletfa.mean(), err_raw=sub.err_raw.mean(), diff=sub["diff"].mean(),
                            diff_lo=lo_, diff_hi=hi_))
    A = pd.DataFrame(acc); A.to_csv(os.path.join(T, f"c10_{tag}_accuracy.tsv"), sep="\t", index=False)
    a_all = A[A.by == "all"].iloc[0]
    small = A[(A.by == "size") & (A.bin == "10-49")]; low = A[(A.by == "reads") & (A.bin == "1-19")]
    better = any(len(x) and x.diff_hi.iloc[0] < 0 for x in (small, low))
    pass_101 = bool(a_all.err_leafletfa <= a_all.err_raw and better)
    log(A.round(4).to_string(index=False)); log(f"10.1 {'PASS' if pass_101 else 'FAIL'}")
    cal = []
    tp0 = np.nan_to_num(tpsi)
    inside = (tp0 >= iv[0]) & (tp0 <= iv[1])                                    # parameter + mouse bootstrap (note 3)
    inside_p = (tp0 >= ivp[0]) & (tp0 <= ivp[1])                                # parameter-only (note 2)
    for k, t in enumerate(types):
        for lo_b, hi_b in ((100, 499), (500, np.inf)):
            u = np.flatnonzero((te_reads[k] >= lo_b) & (te_reads[k] <= hi_b) & (te_cov[k] >= 10))
            j = np.flatnonzero(np.isin(D.inv, u) & (nt[k] > 0))
            if len(j):
                cal.append(dict(cell_type=t, test_read_bin=f"{lo_b}-{hi_b}", junctions=len(j), one_mouse=bool(Gr.n_mice[k] < 2),
                                inside_bootstrap=int(inside[k, j].sum()), inside_param_only=int(inside_p[k, j].sum())))
    Cl = pd.DataFrame(cal); Cl.to_csv(os.path.join(OUT, f"{tag}_calibration_by_type.tsv"), sep="\t", index=False)
    Cs = Cl.groupby("test_read_bin")[["junctions", "inside_bootstrap", "inside_param_only"]].sum()
    Cs["coverage_bootstrap"] = Cs.inside_bootstrap / Cs.junctions; Cs["coverage_param_only"] = Cs.inside_param_only / Cs.junctions
    Cs.to_csv(os.path.join(T, f"c10_{tag}_calibration.tsv"), sep="\t"); log(Cs.to_string())
    spec, TVr = call(D, point[:-1], rest[:-1], P[:-1], reads[:-1], cov[:-1], delta)
    np.savez_compressed(os.path.join(OUT, f"{tag}_estimates.npz"), types=np.array(types), point=point[:-1], rest=rest[:-1],
                        lo=iv[0][:-1], hi=iv[1][:-1], lo_rest=ivr[0][:-1], hi_rest=ivr[1][:-1], reads=reads[:-1], cov=cov[:-1],
                        cells=Gr.n[:-1], n_mice=Gr.n_mice[:-1], specific=spec, tv=TVr, p_tv=P[:-1], delta=delta,
                        lo_param=ivp[0][:-1], hi_param=ivp[1][:-1])
    # ---- 10.2a/b mouse halves, 10.7 pairwise ----
    mice = np.unique(D.mouse); mA = set(np.random.default_rng(0).permutation(mice)[: len(mice) // 2])
    inA = np.array([m in mA for m in D.mouse])

    prs = gb_pairs(types) if ds == "TMS" else []

    def half(mask, labels_g, s, pairs=()):
        rows_ = np.flatnonzero(mask); Gh = Groups(D, rows_, labels_g[rows_], G)
        p, r, i1, i2, Ph, Pp, _ = estimates(D, Gh, draws=True, seed=s, delta=delta, pairs=pairs)
        rd, cv = reads_cov(D, rows_, labels_g[rows_], G)
        y_, n_ = raw_pb(D.Y, D.Ncell, D.inv, rows_, labels_g[rows_], G)
        return dict(p=p[:-1], r=r[:-1], P=Ph[:-1], Pp=Pp, reads=rd[:-1], cov=cv[:-1], y=y_, n=n_)
    HA = half(inA, g, seed + 101, [ij for _, ij in prs]); HB = half(~inA, g, seed + 202, [ij for _, ij in prs])
    cA, tvA = call(D, HA["p"], HA["r"], HA["P"], HA["reads"], HA["cov"], delta)
    lj, sgn = lead_sign(HA["p"], HA["r"], D.inv, D.n_atse)

    def confirm(H):
        ci, _ = call(D, H["p"], H["r"], H["P"], H["reads"], H["cov"], delta)
        with np.errstate(invalid="ignore", divide="ignore"):
            rp = H["y"][:-1] / H["n"][:-1]; rr = (H["y"].sum(0) - H["y"][:-1]) / (H["n"].sum(0) - H["n"][:-1])
        tvB = tv(np.nan_to_num(rp), np.nan_to_num(rr), D.inv, D.n_atse)
        sB = np.sign(np.take_along_axis(np.nan_to_num(rp) - np.nan_to_num(rr), np.maximum(lj, 0), axis=1))
        cii = (tvB > delta / 2) & (sB == sgn)
        return ci[cA].mean() if cA.any() else np.nan, cii[cA].mean() if cA.any() else np.nan
    conf_i, conf_ii = confirm(HB)
    null = []
    for p_ in range(N_PERM):
        rng_p = np.random.default_rng(1000 + p_); gp = g.copy()
        for m in np.unique(D.mouse[~inA]):
            idx = np.flatnonzero((~inA) & (D.mouse == m)); gp[idx] = rng_p.permutation(g[idx])
        null.append(confirm(half(~inA, gp, seed + 300 + p_)))
        log(f"  null {p_}: {null[-1]}")
    null = np.array(null)
    rep = dict(dataset=ds, seed=seed, delta=delta, calls_full=int(spec.sum()), calls_halfA=int(cA.sum()),
               confirmed_i=conf_i, confirmed_ii=conf_ii, null_i=float(np.nanmean(null[:, 0])), null_ii=float(np.nanmean(null[:, 1])))
    rep["pass_102"] = bool(conf_ii >= 0.5 and conf_ii >= 5 * rep["null_ii"])
    pd.DataFrame([rep]).to_csv(os.path.join(T, f"c10_{tag}_replication.tsv"), sep="\t", index=False); log(rep)
    if ds == "TMS":
        pairwise(D, prs, HA, HB, delta, tag, g, ~inA)
        known_biology(D, types, point, spec, te, tpsi, tag)
    export(D, types, point, rest, iv, spec, TVr, reads, cov, Gr.n, tag)


def gb_pairs(types):
    """the 70 GB contrasts as (name, (i, j)) over the kept types"""
    import re
    a50 = pd.read_csv("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/LEAFCUTTER_CMP_DELTAONLY/analysis_group/a50_contrast_comparison.tsv", sep="\t")
    norm = lambda s: re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")
    code = {norm(t): i for i, t in enumerate(types)}
    return [(c, (code[c.split("_VS_")[0]], code[c.split("_VS_")[1]])) for c in a50.contrast
            if c.split("_VS_")[0] in code and c.split("_VS_")[1] in code]


def pairwise(D, prs, HA, HB, delta, tag, g, inB):
    """10.7: the 70 GB contrasts, rule as 10.2 between two types; half A calls, half B confirmation; null = the two
    types' labels permuted among B's cells within each mouse (20 permutations), model-free confirmation (ii)"""
    out = []
    for c, (i, j) in prs:
        def rule(H):
            T_ = tv(H["p"][[i]], H["p"][[j]], D.inv, D.n_atse)[0]
            ok = (H["reads"][i] >= CALL_READS) & (H["cov"][i] >= CALL_COV) & (H["reads"][j] >= CALL_READS) & (H["cov"][j] >= CALL_COV)
            return (T_ > delta) & (H["Pp"][(i, j)] >= 0.95) & ok
        cA_, cB_ = rule(HA), rule(HB)
        with np.errstate(invalid="ignore", divide="ignore"):
            pi_, pj_ = HB["y"][i] / HB["n"][i], HB["y"][j] / HB["n"][j]
        tvB = tv(np.nan_to_num(pi_)[None], np.nan_to_num(pj_)[None], D.inv, D.n_atse)[0]
        lj, sg = lead_sign(HA["p"][[i]], HA["p"][[j]], D.inv, D.n_atse)
        sB = np.sign((np.nan_to_num(pi_) - np.nan_to_num(pj_))[np.maximum(lj[0], 0)])
        cii = (tvB > delta / 2) & (sB == sg[0])
        nulls = []
        if cA_.any():
            rows_ij = np.flatnonzero(inB & np.isin(g, [i, j]))
            for p_ in range(N_PERM):
                rng_p = np.random.default_rng(5000 + p_); gp = g[rows_ij].copy()
                for m in np.unique(D.mouse[rows_ij]):
                    w = D.mouse[rows_ij] == m; gp[w] = rng_p.permutation(gp[w])
                yy, nn_ = raw_pb(D.Y, D.Ncell, D.inv, rows_ij, (gp == j).astype(int), 2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    a_, b_ = np.nan_to_num(yy[0] / nn_[0]), np.nan_to_num(yy[1] / nn_[1])
                tvN = tv(a_[None], b_[None], D.inv, D.n_atse)[0]
                sN = np.sign((a_ - b_)[np.maximum(lj[0], 0)])
                nulls.append(((tvN > delta / 2) & (sN == sg[0]))[cA_].mean())
        out.append(dict(contrast=c, calls_A=int(cA_.sum()), confirmed_i=cB_[cA_].mean() if cA_.any() else np.nan,
                        confirmed_ii=cii[cA_].mean() if cA_.any() else np.nan, null_ii=np.mean(nulls) if nulls else np.nan))
    P = pd.DataFrame(out); P.to_csv(os.path.join(T, f"c10_{tag}_pairwise.tsv"), sep="\t", index=False)
    w = P.calls_A
    conf = np.nansum(P.confirmed_ii * w) / max(w.sum(), 1); nul = np.nansum(P.null_ii * w) / max(w.sum(), 1)
    log(f"10.7: {len(P)} contrasts, {w.sum()} calls in half A, pooled confirmed (ii) {conf:.3f}, null {nul:.3f} -> "
        f"{'PASS' if conf >= 0.5 and conf >= 5 * nul else 'FAIL'}")


def known_biology(D, types, point, spec, te, tpsi, tag):
    P = vlib.paths("TMS")
    with h5py.File(P["train"], "r") as f:
        gene = vlib._strings(vlib._splicing_root(f)["var"]["gene_name"]).astype(str)
    jid = D.post["junction_id"]; ev_names = D.post["event_id"]
    ty = np.array(types); up = np.char.upper(ty)
    epi = np.array([any(k in u for k in EPI) for u in up]); mes = np.array([any(k in u for k in MES) for u in up]) & ~epi
    jb = int(np.flatnonzero(jid == FGFR2_IIIB)[0]); eb = D.inv[jb]
    res = dict(fgfr2_epi_min=float(point[:-1][epi, jb].min()), fgfr2_mes_max=float(point[:-1][mes, jb].max()),
               fgfr2_all_epi_above_all_mes=bool(point[:-1][epi, jb].min() > point[:-1][mes, jb].max()),
               fgfr2_specific_in_epithelial=int(spec[epi, eb].sum()))
    pe = np.unique(D.inv[gene == "Ptprc"])
    lym = np.array([any(k in u for k in LYMPHOID) for u in up])
    hits = [(types[k], ev_names[e]) for k in np.flatnonzero(lym) for e in pe if spec[k, e]]
    res["ptprc_specific_lymphoid"] = [f"{t} {e}" for t, e in hits]
    json.dump(res, open(os.path.join(T, f"c10_{tag}_known_biology.json"), "w"), indent=1); log(res)


def export(D, types, point, rest, iv, spec, TVr, reads, cov, ncell, tag):
    P = vlib.paths("TMS" if tag.startswith("TMS") else "Allen")
    with h5py.File(P["train"], "r") as f:
        v = vlib._splicing_root(f)["var"]
        gene = vlib._strings(v["gene_name"]).astype(str)
        ann = vlib._strings(v["annotation_status"]).astype(str) if "annotation_status" in v else np.array(["NA"] * D.J)
    chunks = []
    for k, t in enumerate(types):
        j = np.flatnonzero(reads[k][D.inv] > 0)
        e = D.inv[j]
        chunks.append(pd.DataFrame(dict(dataset=tag.split("_")[0], event_id=D.post["event_id"][e], junction_id=D.post["junction_id"][j],
                                        gene=gene[j], annotation_status=ann[j], cell_type=t, psi=point[k, j], psi_lo=iv[0][k, j],
                                        psi_hi=iv[1][k, j], psi_rest=rest[k, j], reads=reads[k, e].astype(int), cells=int(ncell[k]),
                                        cells_covering=cov[k, e].astype(int), tv=TVr[k, e], specific=spec[k, e],
                                        source="LeafletFA (check 10)")))
    X = pd.concat(chunks, ignore_index=True)
    os.makedirs(os.path.join(OUT, "export"), exist_ok=True)
    X.to_csv(os.path.join(OUT, "export", f"catalog_{tag}.tsv.gz"), sep="\t", index=False, float_format="%.4g")
    log(f"export: {len(X):,} rows")


def run_calls(ds, seed):
    D = Data(ds, seed); types, g = type_setup(D); G = len(types) + 1
    delta = json.load(open(os.path.join(T, f"c10_{ds}_s0_delta.json")))["delta"]
    Gr = Groups(D, np.arange(len(D.t)), g, G)
    point, rest, iv, ivr, P, _, ivp = estimates(D, Gr, draws=True, seed=seed, delta=delta)
    reads, cov = reads_cov(D, np.arange(len(D.t)), g, G)
    spec, _ = call(D, point[:-1], rest[:-1], P[:-1], reads[:-1], cov[:-1], delta)
    np.savez_compressed(os.path.join(OUT, f"{ds}_s{seed}_calls.npz"), types=np.array(types), specific=spec)
    log(f"{ds} seed {seed}: {int(spec.sum())} calls")


def run_seeds(ds):
    import itertools, glob
    F = sorted(glob.glob(os.path.join(OUT, f"{ds}_s*_calls.npz")))
    S = {f: np.load(f, allow_pickle=True) for f in F}
    rows = []
    for a, b in itertools.combinations(F, 2):
        x, y = S[a]["specific"], S[b]["specific"]; u = (x | y).sum()
        rows.append(dict(a=os.path.basename(a), b=os.path.basename(b), jaccard=(x & y).sum() / u if u else np.nan))
    J = pd.DataFrame(rows); J.to_csv(os.path.join(T, f"c10_{ds}_seed_jaccard.tsv"), sep="\t", index=False)
    log(f"{ds}: median Jaccard over {len(J)} pairs {J.jaccard.median():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["full", "calls", "seeds"])
    ap.add_argument("--dataset", default="TMS"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"full": lambda: run_full(a.dataset, a.seed), "calls": lambda: run_calls(a.dataset, a.seed), "seeds": lambda: run_seeds(a.dataset)}[a.mode]()
