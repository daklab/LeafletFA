"""Check 12 = amendment 7 (VALIDATION_PREREG.md; approved by Karin 2026-11-06, 30 CPU-h cap): predictive and recalibrated
cell-type intervals. CPU, seed-0 fits, TEST counts; no refit.

    python c12_intervals.py <TMS|Allen>                      # full run -> tsv/c12_<ds>_*.tsv, c12/<ds>_units.npz
    C10_DRAWS=10 C12_SMOKE_TYPES=3 python c12_intervals.py TMS   # smoke: cost only (first 10 of the 200 draws, 3 types)

Implementation (literal reading of the amendment; notes are repeated in the report):
- Draws: the check 10 loop of c10_catalog.estimates, same seeds (torch generator seed 0, numpy seed 7), so the 200 combined
  draws (q(phi), q(psi) + mouse bootstrap, type mean = Sb/Nb) and the parameter-only draws (Sp/n) are regenerated exactly.
  Check: the check 10 interval rebuilt from these draws must reproduce the saved 10.3 coverage.
- Units: the 10.3 units, (type, junction) of ATSEs with >= 100 TEST reads and >= 10 covering TEST cells. In a half, the same
  rule on that half's TEST cells, with the read bin from that half's reads.
- Halves: the 10.2a mouse split (mouse_halves: seed 0 over TRAIN mice). TEST cells take the half of their mouse.id; TEST
  mice with no TRAIN cells are in neither half and are used only in the all-TEST evaluation.
- (a) DM: per draw s and replicate r (1,000 values), every TEST cell of the type with N_ce > 0 gets counts at the ATSE from
  DM(N_ce, gamma_e * psi_t^(s)) (Dirichlet via normalized gammas, then multinomial); counts are summed over the comparison
  set's cells (all TEST, half A, half B) and divided by the summed N. If all gammas of a cell underflow to 0, the cell's
  Dirichlet draw is set to psi_t^(s) itself (counted and reported).
- (a) binomial: Bin(sum of N_ce, psi_tj^(s)) / sum of N_ce per junction, 5 replicates per draw.
- (b): tau^2 per dataset x read bin, smallest on the grid {0, 1e-5 * 2^i, i = 0..20} giving >= 95 % coverage of the
  calibration half's units with m +/- 1.96 sqrt(v + tau^2), m and v the mean and variance of that half's (a) DM values;
  if none reaches 95 %, the largest grid value (reported). Applied, clipped to [0, 1], to the other half's units.
"""
import json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
import c10_catalog as C  # noqa: E402
from c10_catalog import Data, Groups, type_setup, mouse_halves, reads_cov, raw_pb, onehot, phi_draw, psi_draw, log  # noqa: E402

S_DRAWS = C.S_DRAWS; R_REP = 5
SMOKE = int(os.environ.get("C12_SMOKE_TYPES", 0))
OUT = os.path.join(vlib.V, "c12"); T = os.path.join(vlib.V, "tsv")
BINS = ((100, 499, "100-499"), (500, np.inf, "500-inf"))
GRID = np.array([0.0] + [1e-5 * 2 ** i for i in range(21)])


def binlab(r):
    out = np.full(len(r), "", object)
    for lo, hi, lab in BINS:
        out[(r >= lo) & (r <= hi)] = lab
    return out


def main(ds):
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    D = Data(ds, 0); D.load_test(ds)
    types, g = type_setup(D); G = len(types) + 1; allrows = np.arange(len(D.t)); Gr = Groups(D, allrows, g, G)
    reads, _ = reads_cov(D, allrows, g, G)
    te = D.test; _, _, tmouse = C.labels(ds, vlib.paths(ds)["test"])
    code = {t: i for i, t in enumerate(types)}; gt = np.array([code.get(t, len(types)) for t in te["t"]])
    mice = np.unique(D.mouse); mA = set(np.random.default_rng(0).permutation(mice)[: len(mice) // 2])
    assert np.array_equal(np.isin(D.mouse, list(mA)), mouse_halves(D))
    half = np.where(np.isin(tmouse, list(mA)), 0, np.where(np.isin(tmouse, mice), 1, -1))   # 0 = A, 1 = B, -1 = neither
    log(f"{ds}: TEST cells {len(gt):,}; half A {int((half == 0).sum()):,}, B {int((half == 1).sum()):,}, neither {int((half == -1).sum()):,}")
    sets = {"all": np.ones(len(gt), bool), "A": half == 0, "B": half == 1}
    Nt, Yt = te["N"].tocsr(), te["Y"].tocsr()
    TR, TC, TY = {}, {}, {}
    for k_, m_ in sets.items():
        O = onehot(gt[m_], G)
        TR[k_] = np.asarray((O @ Nt[m_]).todense()); TC[k_] = np.asarray((O @ (Nt[m_] > 0).astype(np.float64)).todense())
        TY[k_] = np.asarray((O @ Yt[m_]).todense())
    # units: (type, ATSE) meeting the 10.3 rule on all TEST cells (half units are a subset)
    ktypes = list(range(len(types)))
    if SMOKE:
        ktypes = list(np.random.default_rng(1).choice(len(types), SMOKE, replace=False))
    U = []
    for k in range(len(types)):
        e = np.flatnonzero((TR["all"][k] >= 100) & (TC["all"][k] >= 10))
        U += [(k, x) for x in e]
    U = np.array(U); log(f"(type, ATSE) units: {len(U):,} in {len(np.unique(U[:, 0]))} types")
    order = np.argsort(D.inv, kind="stable"); starts = np.searchsorted(D.inv[order], np.arange(D.n_atse + 1))
    jof = lambda e: order[starts[e]:starts[e + 1]]  # noqa: E731
    uj_k, uj_e, uj_j = [], [], []
    for k, e in U:
        j = jof(e); uj_k.append(np.full(len(j), k)); uj_e.append(np.full(len(j), e)); uj_j.append(j)
    uj_k, uj_e, uj_j = map(np.concatenate, (uj_k, uj_e, uj_j)); NJ = len(uj_j)
    log(f"unit junctions: {NJ:,}")
    # ---- draws (check 10 seeds and order) ----
    n = Gr.n; gen = torch.Generator().manual_seed(0); rng = np.random.default_rng(7)
    Xc = np.empty((S_DRAWS, NJ), np.float64); Xp = np.empty((S_DRAWS, NJ), np.float32); td = time.time()
    for s in range(S_DRAWS):
        ph, ps = phi_draw(D, gen)[Gr.rows], psi_draw(D, gen)
        Sp, _ = Gr.sums(D, ph, ps); Sb, Nb = Gr.sums(D, ph, ps, Gr.boot_weights(rng))
        Xp[s] = Sp[uj_k, uj_j] / n[uj_k]; Xc[s] = Sb[uj_k, uj_j] / np.maximum(Nb[uj_k], 1e-12)
        if s % 50 == 0:
            log(f"  draw {s}/{S_DRAWS}")
    t_draws = time.time() - td; log(f"draws: {t_draws:.0f} s")
    # ---- check 10 interval (as 10.3) from the regenerated draws ----
    lo10, hi10 = np.quantile(Xc, [0.025, 0.975], axis=0); lo10p, hi10p = np.quantile(Xp, [0.025, 0.975], axis=0)
    # observed TEST pseudobulk psi per set
    obs, Nsum, qual, rbin = {}, {}, {}, {}
    for k_ in sets:
        Nsum[k_] = TR[k_][uj_k, uj_e]
        with np.errstate(invalid="ignore", divide="ignore"):
            obs[k_] = TY[k_][uj_k, uj_j] / Nsum[k_]
        qual[k_] = (Nsum[k_] >= 100) & (TC[k_][uj_k, uj_e] >= 10); rbin[k_] = binlab(Nsum[k_])
    ins = (obs["all"] >= lo10) & (obs["all"] <= hi10)
    if S_DRAWS == 200 and not SMOKE:
        saved = pd.read_csv(os.path.join(T, f"c10_{ds}_s0_calibration.tsv"), sep="\t").set_index("test_read_bin")
        for lab in ("100-499", "500-inf"):
            m = rbin["all"] == lab; c = ins[m].mean()
            log(f"check 10 coverage rebuilt {lab}: {c:.6f} (saved {saved.loc[lab, 'coverage_bootstrap']:.6f}), junctions {m.sum()} ({saved.loc[lab, 'junctions']})")
            assert m.sum() == saved.loc[lab, "junctions"] and abs(c - saved.loc[lab, "coverage_bootstrap"]) < 1e-6, "check 10 draws not reproduced"
    # ---- (a) simulations, per type ----
    res = {f"{kind}_{st}_{q}": np.full(NJ, np.nan, np.float32) for kind in ("dmc", "dmp", "bnc", "bnp") for st in sets for q in ("lo", "hi", "m", "v")}
    gam = D.gam; rng2 = np.random.default_rng(20261106); work_done = 0; work_total = 0; ts = time.time(); underflow = 0
    tcells = {k: np.flatnonzero(gt == k) for k in range(len(types))}
    for k in range(len(types)):
        sel = np.flatnonzero(uj_k == k)
        if len(sel) == 0:
            continue
        es = np.unique(uj_e[sel]); Nk = Nt[tcells[k]][:, es].tocoo()
        work_total += len(Nk.data) * np.mean([len(jof(e)) for e in es])
    log(f"simulation work (pairs x junctions): total {work_total:,.0f}")
    for k in ktypes:
        sel = np.flatnonzero(uj_k == k)
        if len(sel) == 0:
            continue
        tk = tcells[k]; hk = half[tk]; es = np.unique(uj_e[sel]); epos = {e: i for i, e in enumerate(es)}
        Nk = Nt[tk][:, es].tocoo()                                           # cells x unit ATSEs
        pc, pa, pN = Nk.row, Nk.col, Nk.data.astype(np.int64)
        sims = {(kind, st): np.empty((S_DRAWS * R_REP, len(sel)), np.float32) for kind in ("dmc", "dmp", "bnc", "bnp") for st in sets}
        col = {e: sel[uj_e[sel] == e] for e in es}                          # unit-junction rows per ATSE (in jof order)
        loc = {e: np.searchsorted(sel, col[e]) for e in es}
        sizes = {}
        for e in es:
            sizes.setdefault(len(col[e]), []).append(e)
        for m, el in sizes.items():
            el = np.array(el); ai = np.array([epos[e] for e in el]); pm = np.isin(pa, ai)
            P_c, P_a, P_N = pc[pm], pa[pm], pN[pm]; P_h = hk[P_c]
            uidx = np.searchsorted(ai, P_a)                                   # pair -> index in el (ai sorted as es is)
            rows_e = np.stack([col[e] for e in el])                           # n_el x m global unit-junction rows
            locs = np.stack([loc[e] for e in el])
            gm = gam[el][:, None]
            Wst = {st: (np.ones(len(P_N), bool) if st == "all" else P_h == (0 if st == "A" else 1)) for st in sets}
            Mst = {st: sp.csr_matrix((Wst[st].astype(np.float64), (uidx, np.arange(len(uidx)))), shape=(len(el), len(uidx))) for st in sets}
            Nset = {st: Mst[st] @ P_N.astype(np.float64) for st in sets}
            for kind, X in (("c", Xc), ("p", Xp)):
                for s in range(S_DRAWS):
                    psi = np.clip(X[s][rows_e].astype(np.float64), 0, None); psi /= psi.sum(1, keepdims=True)   # n_el x m
                    alpha = (gm * psi)[uidx]                                    # pairs x m
                    for r in range(R_REP):
                        gs = rng2.standard_gamma(alpha); tot = gs.sum(1, keepdims=True); z = tot[:, 0] <= 0
                        if z.any():
                            underflow += int(z.sum()); gs[z] = psi[uidx[z]]; tot[z] = 1.0
                        y = rng2.multinomial(P_N, gs / tot)                    # pairs x m
                        i = s * R_REP + r
                        for st in sets:
                            ys = Mst[st] @ y
                            with np.errstate(invalid="ignore", divide="ignore"):
                                sims[("dm" + kind, st)][i, locs.ravel()] = (ys / Nset[st][:, None]).ravel()
                            Nn = Nset[st][:, None] * np.ones((1, m))
                            with np.errstate(invalid="ignore", divide="ignore"):
                                sims[("bn" + kind, st)][i, locs.ravel()] = (rng2.binomial(Nn.astype(np.int64), psi) / Nn).ravel()
            work_done += len(P_N) * m
        for (kind, st), A in sims.items():
            ok = np.isfinite(A).all(0)
            q = np.full((2, len(sel)), np.nan); q[:, ok] = np.quantile(A[:, ok], [0.025, 0.975], axis=0)
            res[f"{kind}_{st}_lo"][sel], res[f"{kind}_{st}_hi"][sel] = q
            res[f"{kind}_{st}_m"][sel] = np.where(ok, A.mean(0), np.nan); res[f"{kind}_{st}_v"][sel] = np.where(ok, A.var(0), np.nan)
        log(f"  type {k} {types[k]}: {len(sel):,} unit junctions, {len(pN):,} cell-ATSE pairs, {time.time() - ts:.0f} s cumulative")
    t_sims = time.time() - ts
    cost = dict(dataset=ds, draws=S_DRAWS, smoke_types=SMOKE, t_load_s=round(td - t0), t_draws_s=round(t_draws), t_sims_s=round(t_sims),
                work_done=float(work_done), work_total=float(work_total), unit_junctions=int(NJ), dirichlet_underflow_cells=underflow)
    log(json.dumps(cost))
    json.dump(cost, open(os.path.join(OUT, f"{ds}{'_smoke' if SMOKE else ''}_cost.json"), "w"), indent=1)
    if SMOKE:
        return
    # ---- tables ----
    nm = Gr.n_mice[uj_k]; trd = reads[uj_k, uj_e]
    np.savez_compressed(os.path.join(OUT, f"{ds}_units.npz"), k=uj_k, e=uj_e, j=uj_j, n_mice=nm, train_reads=trd, lo10=lo10, hi10=hi10, lo10p=lo10p, hi10p=hi10p,
                        **{f"obs_{st}": obs[st] for st in sets}, **{f"N_{st}": Nsum[st] for st in sets}, **{f"qual_{st}": qual[st] for st in sets}, **res)
    rows = []

    def cov(mask, lo, hi, o, method, draws, evalset):
        for lab in ("100-499", "500-inf"):
            m = mask & (rbin[evalset] == lab) & np.isfinite(lo)
            if m.sum():
                rows.append(dict(dataset=ds, draws=draws, method=method, evaluated_on=evalset, test_read_bin=lab, junctions=int(m.sum()),
                                 coverage=float(((o[m] >= lo[m]) & (o[m] <= hi[m])).mean()), median_width=float(np.median(hi[m] - lo[m]))))
    taus = []
    for dk, dname in (("c", "combined"), ("p", "parameter-only")):
        l10, h10 = (lo10, hi10) if dk == "c" else (lo10p, hi10p)
        for st in sets:
            q = qual[st] if st != "all" else np.ones(NJ, bool)
            cov(q, l10, h10, obs[st], "check 10 interval", dname, st)
            cov(q, res[f"dm{dk}_{st}_lo"], res[f"dm{dk}_{st}_hi"], obs[st], "(a) DM", dname, st)
            cov(q, res[f"bn{dk}_{st}_lo"], res[f"bn{dk}_{st}_hi"], obs[st], "(a) binomial", dname, st)
        for cal, ev in (("A", "B"), ("B", "A")):
            lo_ab, hi_ab = np.full(NJ, np.nan), np.full(NJ, np.nan)
            for lab in ("100-499", "500-inf"):
                cm = qual[cal] & (rbin[cal] == lab) & np.isfinite(res[f"dm{dk}_{cal}_m"])
                m_, v_, o_ = res[f"dm{dk}_{cal}_m"][cm], res[f"dm{dk}_{cal}_v"][cm], obs[cal][cm]
                cv = [((o_ >= m_ - 1.96 * np.sqrt(v_ + t2)) & (o_ <= m_ + 1.96 * np.sqrt(v_ + t2))).mean() for t2 in GRID]
                ok = np.flatnonzero(np.array(cv) >= 0.95); t2 = GRID[ok[0]] if len(ok) else GRID[-1]
                taus.append(dict(dataset=ds, draws=dname, calibrated_on=cal, test_read_bin=lab, tau2=t2, tau=float(np.sqrt(t2)), reached_95=bool(len(ok)),
                                 calibration_junctions=int(cm.sum()), calibration_coverage=float(cv[ok[0]] if len(ok) else cv[-1])))
                em = qual[ev] & (rbin[ev] == lab)
                mm, vv = res[f"dm{dk}_{ev}_m"][em], res[f"dm{dk}_{ev}_v"][em]
                lo_ab[em] = np.clip(mm - 1.96 * np.sqrt(vv + t2), 0, 1); hi_ab[em] = np.clip(mm + 1.96 * np.sqrt(vv + t2), 0, 1)
            cov(qual[ev], lo_ab, hi_ab, obs[ev], f"(a)+(b), calibrated on {cal}", dname, ev)
            if dk == "c":   # descriptive breakdowns, primary draws
                for nlab, nmask in (("1-2", nm <= 2), ("3-5", (nm >= 3) & (nm <= 5)), (">=6", nm >= 6)):
                    cov(qual[ev] & nmask, lo_ab, hi_ab, obs[ev], f"(a)+(b) cal {cal}; mice of type {nlab}", dname, ev)
                for rlab, rmask in (("<1000", trd < 1000), (">=1000", trd >= 1000)):
                    cov(qual[ev] & rmask, lo_ab, hi_ab, obs[ev], f"(a)+(b) cal {cal}; TRAIN reads {rlab}", dname, ev)
    Cv = pd.DataFrame(rows); Tz = pd.DataFrame(taus)
    Cv.to_csv(os.path.join(T, f"c12_{ds}_coverage.tsv"), sep="\t", index=False); Tz.to_csv(os.path.join(T, f"c12_{ds}_tau2.tsv"), sep="\t", index=False)
    prim = Cv[(Cv.draws == "combined") & Cv.method.str.match(r"^\(a\)\+\(b\), calibrated on")]
    a_only = Cv[(Cv.draws == "combined") & (Cv.method == "(a) DM")]
    S = dict(dataset=ds, ab_all_in_band=bool(((prim.coverage >= 0.90) & (prim.coverage <= 1.0)).all() and len(prim) == 4),
             a_all_in_band={st: bool(((a_only[a_only.evaluated_on == st].coverage >= 0.9) & (a_only[a_only.evaluated_on == st].coverage <= 1)).all()) for st in sets},
             cost=cost, t_total_s=round(time.time() - t0))
    json.dump(S, open(os.path.join(T, f"c12_{ds}_summary.json"), "w"), indent=1)
    log(Cv.round(4).to_string(index=False)); log(Tz.to_string(index=False)); log(json.dumps(S))


if __name__ == "__main__":
    main(sys.argv[1])
