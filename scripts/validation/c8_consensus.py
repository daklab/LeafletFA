"""Check 8 (CPU): consensus programs, cNMF-style (pre-registration amendment 4).

    python c8_consensus.py cluster     # pool live centered psi of all TMS K20 fits, cluster, write the consensus fit
    (E-step, GPU: run_leafletfa_baseline.py --reference-fit c8_consensus/consensus_fit.npz --K <Kc>
                  --project train=<TMS train> --out c8_consensus/estep)
    python c8_consensus.py report      # profiles, markers, within-cluster Jaccard, reading, pseudobulk check

Fits: seeds 0-2 from the SpliceVI baseline, seeds 3-14 in c8_consensus/fits (TMS, K 20, training cells).
  cluster  average linkage, cosine distance on centered psi, cut 0.2; keep recurrence >= 0.6 of fits;
           consensus psi = mean of member psi (per fit first), pi = mean over fits of summed member pi,
           per-ATSE concentration = median over fits
  report   per consensus program: recurrence, members, mean pairwise centered cosine, argmax fraction,
           top tissue-cell types by mean consensus phi, top 10 marker junctions, within-cluster top-50
           marker Jaccard; reading (>= 8 programs with recurrence >= 0.6 and Jaccard >= 0.5);
           pseudobulk Mann-Whitney check of the top 10 ATSEs of the strongest 3 programs vs a random-ATSE null
Writes c8_consensus/consensus_fit.npz and tsv/c8_*.tsv.
"""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

LIVE, CUT, REC_MIN, JAC_MIN, N_PASS, TOP, MIN_TYPE = 0.01, 0.2, 0.6, 0.5, 8, 50, 20
USER, NONUSER, MIN_READS, N_ATSE, N_NULL = 0.2, 0.02, 20, 10, 200
C8 = os.path.join(vlib.V, "c8_consensus")
SEEDS = range(15)


def fit_path(s):
    if s <= 2:
        return os.path.join(vlib.B, "TMS", f"full_K20_s{s}_f64", "fit", "fit.npz")
    return os.path.join(C8, "fits", f"full_K20_s{s}_f64", "fit", "fit.npz")


def markers(psi, pi):
    """check 6: |psi_k - pi-weighted mean of the other programs|, per junction"""
    dep = np.empty_like(psi)
    for k in range(psi.shape[0]):
        w = np.delete(pi, k); w = w / w.sum()
        dep[k] = np.abs(psi[k] - w @ np.delete(psi, k, axis=0))
    return dep


def load_fits():
    fits, ref = [], None
    for s in SEEDS:
        p = fit_path(s)
        if not os.path.exists(p):
            print(f"seed {s}: no fit"); continue
        z = np.load(p, allow_pickle=True)
        phi = z["phi"].astype(np.float64)
        if not np.isfinite(phi).all():
            print(f"seed {s}: non-finite phi, excluded"); continue
        if ref is None:
            ref = z
        for k in ("cell_id", "junction_id", "event_id", "inv_ev"):
            assert np.array_equal(z[k], ref[k]), (s, k)
        psi = z["psi_mean"].astype(np.float64); pi = np.asarray(z["pi"], np.float64); pi = pi / pi.sum()
        live = np.flatnonzero(np.bincount(phi.argmax(1), minlength=psi.shape[0]) / len(phi) >= LIVE)
        fits.append(dict(seed=s, psi=psi, pi=pi, C=psi - pi @ psi, live=live, conc=z["bb_conc_vec"].astype(np.float64),
                         nan_restarts=len(z["nan_steps"])))
    return fits, ref


def pool(fits):
    rows = [(f["seed"], fi, k) for fi, f in enumerate(fits) for k in f["live"]]
    X = np.vstack([fits[fi]["C"][k] for _, fi, k in rows])
    return pd.DataFrame(rows, columns=["seed", "fit_idx", "program"]), X


def clusters(D, method, cut):
    return fcluster(linkage(D, method), t=cut, criterion="distance")


def summarize(M, lab, n_fits, S, top_sets):
    """per cluster: recurrence, members, mean pairwise cosine, Jaccard across fits"""
    out = []
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        seeds = M.seed.values[idx]
        cos = [S[i, j] for i, j in itertools.combinations(idx, 2)]
        jac = [len(top_sets[i] & top_sets[j]) / len(top_sets[i] | top_sets[j])
               for i, j in itertools.combinations(idx, 2) if M.seed.values[i] != M.seed.values[j]]
        out.append(dict(cluster=c, n_members=len(idx), n_fits=len(set(seeds)), recurrence=len(set(seeds)) / n_fits,
                        mean_pairwise_cosine=np.mean(cos) if cos else np.nan,
                        jaccard_top50=np.mean(jac) if jac else np.nan))
    return pd.DataFrame(out)


def cluster_stage():
    fits, ref = load_fits()
    n = len(fits)
    print(f"{n} fits: seeds {[f['seed'] for f in fits]}")
    if n < 12:
        raise SystemExit("fewer than 12 fits completed: stop and report (amendment 4)")
    M, X = pool(fits)
    D = pdist(X, "cosine"); S = 1 - squareform(D)
    dep = {f["seed"]: markers(f["psi"], f["pi"]) for f in fits}
    top_sets = [set(np.argsort(-dep[s][k])[:TOP]) for s, k in zip(M.seed, M.program)]
    sens = []
    for method, cut in (("average", 0.1), ("average", 0.2), ("average", 0.3), ("complete", 0.2)):
        lab = clusters(D, method, cut)
        T = summarize(M, lab, n, S, top_sets)
        kept = T[T.recurrence >= REC_MIN]
        sens.append(dict(linkage=method, cut=cut, n_live_programs=len(M), n_clusters=len(T), n_kept=len(kept),
                         n_kept_and_jaccard=int((kept.jaccard_top50 >= JAC_MIN).sum())))
        if (method, cut) == ("average", CUT):
            M["cluster"] = lab; TP = T
    pd.DataFrame(sens).to_csv(os.path.join(vlib.V, "tsv", "c8_sensitivity.tsv"), sep="\t", index=False)
    TP = TP.sort_values(["recurrence", "jaccard_top50"], ascending=False).reset_index(drop=True)
    keep = TP[TP.recurrence >= REC_MIN].cluster.tolist()
    TP["consensus"] = TP.cluster.map({c: i for i, c in enumerate(keep)})
    M["consensus"] = M.cluster.map(dict(zip(TP.cluster, TP.consensus)))
    M.to_csv(os.path.join(vlib.V, "tsv", "c8_members.tsv"), sep="\t", index=False)
    TP.to_csv(os.path.join(vlib.V, "tsv", "c8_clusters.tsv"), sep="\t", index=False)
    Kc = len(keep)
    psi_c, pi_c = np.zeros((Kc, X.shape[1])), np.zeros(Kc)
    by_seed = {f["seed"]: f for f in fits}
    for i, c in enumerate(keep):
        m = M[M.cluster == c]
        per_fit = [by_seed[s]["psi"][g.program.values].mean(0) for s, g in m.groupby("seed")]
        psi_c[i] = np.mean(per_fit, 0)
        pi_c[i] = np.mean([by_seed[s]["pi"][m.program.values[m.seed.values == s]].sum() for s in by_seed])
    pi_c = pi_c / pi_c.sum()
    conc = np.median(np.vstack([f["conc"] for f in fits]), 0)
    os.makedirs(C8, exist_ok=True)
    out = os.path.join(C8, "consensus_fit.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; not overwritten")
    np.savez_compressed(out, psi_mean=psi_c.astype(np.float32), pi=pi_c.astype(np.float32),
                        bb_conc_vec=conc.astype(np.float32), dir_conc=float(Kc), K=Kc, junction_id=ref["junction_id"],
                        event_id=ref["event_id"], inv_ev=ref["inv_ev"], cell_id=ref["cell_id"],
                        seeds=np.array([f["seed"] for f in fits]), clusters=np.array(keep))
    print(pd.DataFrame(sens).to_string(index=False))
    print(TP.round(3).to_string(index=False))
    print(f"{Kc} consensus programs -> {out}")


def report_stage():
    P = vlib.paths("TMS")
    z = np.load(os.path.join(C8, "consensus_fit.npz"), allow_pickle=True)
    psi, pi = z["psi_mean"].astype(np.float64), z["pi"].astype(np.float64)
    Kc = psi.shape[0]
    e = np.load(os.path.join(C8, "estep", "factors_train.npz"), allow_pickle=True)
    phi = e["phi"].astype(np.float64)
    cells, _, mouse = vlib.obs_labels(P["train"])
    import h5py
    with h5py.File(P["train"], "r") as f:
        o = f["obs"]; tct = vlib._strings(o["tissue_cell_type"]).astype(str)
        gene = vlib._strings(vlib._splicing_root(f)["var"]["gene_name"]).astype(str)
    assert np.array_equal(e["cell_id"].astype(str), cells) and np.array_equal(z["cell_id"].astype(str), cells)
    jid = z["junction_id"].astype(str); ev_u = z["event_id"].astype(str); inv = np.asarray(z["inv_ev"], np.int64)
    TP = pd.read_csv(os.path.join(vlib.V, "tsv", "c8_clusters.tsv"), sep="\t")
    TP = TP[TP.consensus.notna()].copy(); TP["consensus"] = TP.consensus.astype(int)
    TP = TP.set_index("consensus").sort_index()
    n_t = pd.Series(tct).value_counts(); types = n_t[n_t >= MIN_TYPE].index
    prof = pd.DataFrame(phi).groupby(tct).mean().loc[types]
    dep = markers(psi, pi)
    argfrac = np.bincount(phi.argmax(1), minlength=Kc) / len(phi)
    rows = []
    for k in range(Kc):
        tt = prof[k].sort_values(ascending=False).head(5)
        tj = np.argsort(-dep[k])[:10]
        rows.append(dict(program=f"CP{k + 1}", recurrence=TP.loc[k, "recurrence"], n_fits=TP.loc[k, "n_fits"],
                         n_members=TP.loc[k, "n_members"], mean_pairwise_cosine=TP.loc[k, "mean_pairwise_cosine"],
                         jaccard_top50=TP.loc[k, "jaccard_top50"], pi=pi[k], argmax_fraction=argfrac[k],
                         passes=bool(TP.loc[k, "recurrence"] >= REC_MIN and TP.loc[k, "jaccard_top50"] >= JAC_MIN),
                         top_types="; ".join(f"{t} {v:.2f}" for t, v in tt.items()),
                         top_junctions="; ".join(f"{gene[j]} {jid[j]} ({psi[k, j]:.2f})" for j in tj)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(vlib.V, "tsv", "c8_consensus_programs.tsv"), sep="\t", index=False)
    prof.columns = [f"CP{k + 1}" for k in range(Kc)]
    prof.to_csv(os.path.join(vlib.V, "tsv", "c8_celltype_usage.tsv"), sep="\t")
    n_pass = int(R.passes.sum())
    reading = "consensus restores interpretable programs" if n_pass >= N_PASS else "current wording stays"
    print(R.drop(columns=["top_junctions"]).round(3).to_string(index=False))
    print(f"\n{n_pass} of {Kc} consensus programs pass (recurrence >= {REC_MIN}, Jaccard >= {JAC_MIN}): {reading}")

    # ── pseudobulk check, strongest 3 ──
    strong = R[R.recurrence >= REC_MIN].sort_values(["jaccard_top50", "recurrence"], ascending=False).head(3).index
    Y = vlib.read_counts(P["train"])
    grp = pd.Series(np.char.add(np.char.add(tct, "||"), mouse.astype(str)))
    g_u, g_inv = np.unique(grp.values, return_inverse=True)
    import scipy.sparse as sp
    G = sp.csr_matrix((np.ones(len(g_inv)), (g_inv, np.arange(len(g_inv)))), shape=(len(g_u), len(g_inv)))
    PB = np.asarray((G @ Y).todense(), np.float64)                                    # samples x J
    E = sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)))
    PBe = np.asarray(PB @ E)                                                          # samples x ATSE
    g_type = np.array([g.split("||")[0] for g in g_u])
    ev_gene = pd.Series(gene).groupby(inv).agg(lambda g: ",".join(sorted(set(g))))

    def test(e_, j, users, non):
        ok = PBe[:, e_] >= MIN_READS
        su, sn = ok & np.isin(g_type, users), ok & np.isin(g_type, non)
        if su.sum() < 3 or sn.sum() < 3:
            return None
        u = PB[su, j] / PBe[su, e_]; v = PB[sn, j] / PBe[sn, e_]
        return np.median(u) - np.median(v), mannwhitneyu(u, v, alternative="two-sided").pvalue, int(su.sum()), int(sn.sum())

    trows, nrows = [], []
    for k in strong:
        mk = prof.iloc[:, k]
        users = list(mk[mk >= USER].index)
        if len(users) < 3:
            users = list(mk.sort_values(ascending=False).head(3).index)
        non = list(mk[mk <= NONUSER].index)
        w = np.delete(pi, k); w = w / w.sum(); bar = w @ np.delete(psi, k, axis=0)
        d = psi[k] - bar
        tv = pd.Series(np.abs(d)).groupby(inv).sum() * 0.5
        n_ok = 0
        for e_ in tv.sort_values(ascending=False).index:
            cols = np.flatnonzero(inv == e_); j = cols[np.argmax(np.abs(d[cols]))]
            r = test(e_, j, users, non)
            if r is None:
                continue
            trows.append(dict(program=f"CP{k + 1}", atse=ev_u[e_], genes=ev_gene[e_], tv=tv[e_], junction=jid[j],
                              psi_program=psi[k, j], psi_others=bar[j], delta_pseudobulk=r[0], p=r[1],
                              n_user_samples=r[2], n_nonuser_samples=r[3], n_user_types=len(users),
                              n_nonuser_types=len(non), concordant=bool(np.sign(r[0]) == np.sign(d[j]))))
            n_ok += 1
            if n_ok == N_ATSE:
                break
        rng = np.random.default_rng(1000 + int(k)); got = 0
        for e_ in rng.permutation(len(ev_u)):
            cols = np.flatnonzero(inv == e_); j = rng.choice(cols)
            r = test(e_, j, users, non)
            if r is None:
                continue
            nrows.append(dict(program=f"CP{k + 1}", atse=ev_u[e_], p=r[1])); got += 1
            if got == N_NULL:
                break
    T = pd.DataFrame(trows)
    from statsmodels.stats.multitest import multipletests
    T["q"] = multipletests(T.p, method="fdr_bh")[1]
    T["confirmed"] = T.concordant & (T.q < 0.05)
    T.to_csv(os.path.join(vlib.V, "tsv", "c8_pseudobulk_top_atses.tsv"), sep="\t", index=False)
    N = pd.DataFrame(nrows); N.to_csv(os.path.join(vlib.V, "tsv", "c8_pseudobulk_null.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["atse"]).round(4).to_string(index=False))
    print(T.groupby("program").agg(confirmed=("confirmed", "sum"), concordant=("concordant", "sum"), n=("p", "size")))
    print("null fraction P < 0.05:", N.groupby("program").p.apply(lambda p: (p < 0.05).mean()).round(3).to_dict())


if __name__ == "__main__":
    {"cluster": cluster_stage, "report": report_stage}[sys.argv[1]]()
