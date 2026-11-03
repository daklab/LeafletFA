"""Per-pair junction tables for the SpliceVI paper's DS comparison (STAGE5 §189; Karin 2026-11-02). Exports only; no scoring.

    python ds_pair_tables.py --data X.h5mu --fit-dir RUN/fit --pairs PAIRS --out DIR [--easysci-broad]

For each pair "A__vs__B" (group 1 = A, group 2 = B; cells grouped by obs `broad_cell_type`), one table DIR/<pair>.tsv.gz:
  junction_id, event_id, psi_group1, psi_group2, delta_psi (= psi_group1 - psi_group2),
  p_absdpsi_ge_0.1             fraction of 200 joint q(phi), q(psi) draws with |delta_psi| >= 0.1 (parameter draws only),
  p_absdpsi_ge_0.1_mouseboot   the same draws combined with the check 10 mouse bootstrap within each group (amendment 6 note 3),
  junction_reads_group1/2, event_reads_group1/2, units_group1/2, units_with_event_reads_group1/2.
Group psi = the check 10 estimator: mean over the group's units of rho~ = (g_e rho + y) / (g_e + N), rho = phi psi (point: the fit's
phi and psi_mean; draws: c10_catalog.Groups.sums). Draw seed 0, as check 10's seed-0 run.
--combined PATH also writes the single table of S189_LEAFLETFA_TABLE_SPEC.md (pair, junction_id, score, dpsi; every junction, no read
filter; dpsi = group A - group B of A__vs__B); score = p_absdpsi_ge_0.1_mouseboot, the check 10 draw scheme (q(phi), q(psi) + mouse bootstrap).
--easysci-broad recomputes broad_cell_type with Smriti's add_broad_cell_type_obs (as her celltype_dsde_splicevi.py does on load);
the stored column and the recomputed one are compared and the agreement is logged.
"""
import argparse, hashlib, json, os, sys, types as _t
import numpy as np, pandas as pd, scipy.sparse as sp, torch, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, c9_lib as L  # noqa: E402
import c10_catalog as C  # noqa: E402

SMRITI_UTILS = "/gpfs/commons/home/svaidyanathan/repos/SpliceVI_Repositories/SpliceVI-utils"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def obs_frame(path):
    with h5py.File(path, "r") as f:
        o = f["obs"]
        idx = vlib._strings(o[o.attrs["_index"]]).astype(str)
        cols = {}
        for k in o.keys():
            if k == o.attrs["_index"]:
                continue
            try:
                cols[k] = vlib._strings(o[k]).astype(str)
            except Exception:  # noqa: BLE001  (numeric columns are not needed)
                pass
    return pd.DataFrame(cols, index=idx)


def broad_types(path, easysci):
    obs = obs_frame(path)
    stored = obs["broad_cell_type"].values
    if not easysci:
        return stored, None
    import logging, importlib, glob
    sys.path.insert(0, SMRITI_UTILS)
    mod = None
    for f in glob.glob(os.path.join(SMRITI_UTILS, "**", "*.py"), recursive=True):
        if "def add_broad_cell_type_obs" in open(f, errors="ignore").read():
            spec = importlib.util.spec_from_file_location("smriti_broad", f); mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod); src = f; break
    fake = _t.SimpleNamespace(obs=obs.copy(), mod={})
    mod.add_broad_cell_type_obs(fake, logging.getLogger("broad"))
    rec = fake.obs["broad_cell_type"].values
    return rec, dict(source=src, agree_with_stored=float(np.mean(rec == stored)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--fit-dir", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--easysci-broad", action="store_true")
    ap.add_argument("--delta", type=float, default=0.1); ap.add_argument("--combined")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pairs = [l.split("\t")[0].strip() for l in open(a.pairs) if "__vs__" in l]
    ct, broad_rec = broad_types(a.data, a.easysci_broad)
    names = sorted(set(sum([p.split("__vs__") for p in pairs], [])))
    miss = [n for n in names if n not in set(ct)]
    assert not miss, f"pair cell types missing from broad_cell_type: {miss}"

    old = L.fit_dir; L.fit_dir = lambda s: a.fit_dir
    try:
        post = L.load_posterior(0)
    finally:
        L.fit_dir = old
    cells = vlib.read_cells(a.data).astype(str)
    assert np.array_equal(post["cell_id"], cells), "fit cell order differs from the data file"
    assert np.array_equal(post["junction_id"], vlib.read_var(a.data)[0].astype(str)), "fit junction order differs from the data file"
    obs = obs_frame(a.data)
    D = _t.SimpleNamespace(post=post, inv=post["inv"].numpy(), n_atse=post["n_atse"], mouse=obs["animal"].values.astype(str))
    D.J = len(D.inv)
    z = np.load(os.path.join(a.fit_dir, "fit.npz"), allow_pickle=True)
    D.gam = np.asarray(z["bb_conc_vec"], np.float64)
    D.Y = vlib.read_counts(a.data).tocsr()
    E = sp.csr_matrix((np.ones(D.J), (np.arange(D.J), D.inv)), shape=(D.J, D.n_atse))
    D.Ncell = (D.Y @ E).tocsr()
    Nc = D.Ncell.tocoo()
    D.V = sp.csr_matrix((Nc.data / (D.gam[Nc.col] + Nc.data), (Nc.row, Nc.col)), shape=D.Ncell.shape)
    Yc = D.Y.tocoo(); Ncj = np.asarray(D.Ncell[Yc.row, D.inv[Yc.col]]).ravel()
    D.R = sp.csr_matrix((Yc.data / (D.gam[D.inv[Yc.col]] + Ncj), (Yc.row, Yc.col)), shape=D.Y.shape)

    code = {n: i for i, n in enumerate(names)}; G = len(names) + 1                      # last group = all other units
    g = np.array([code.get(t, G - 1) for t in ct])
    Gr = C.Groups(D, np.arange(len(cells)), g, G)
    S, _ = Gr.sums(D, post["phi"], post["psi_mean"]); point = S / Gr.n[:, None]
    gen = torch.Generator().manual_seed(0); rng = np.random.default_rng(7)
    Sp = np.empty((C.S_DRAWS, G, D.J), np.float32); Sb = np.empty_like(Sp); Nb = np.empty((C.S_DRAWS, G))
    for s in range(C.S_DRAWS):
        ph, ps = C.phi_draw(D, gen), C.psi_draw(D, gen)
        Sp[s], _ = Gr.sums(D, ph, ps)
        Sb[s], Nb[s] = Gr.sums(D, ph, ps, Gr.boot_weights(rng))
        if s % 50 == 0:
            C.log(f"draw {s}/{C.S_DRAWS}")
    y_g, n_g = C.raw_pb(D.Y, D.Ncell, D.inv, np.arange(len(cells)), g, G)                # G x J (junction reads, event reads)
    cov = np.asarray((C.onehot(g, G) @ (D.Ncell > 0).astype(np.float64)).todense())[:, D.inv]
    man = []; score_col = f"p_absdpsi_ge_{a.delta:g}_mouseboot"
    if a.combined:
        open(a.combined, "w").write("pair\tjunction_id\tscore\tdpsi\n")
    for p in pairs:
        A, B = (code[x] for x in p.split("__vs__"))
        xa, xb = Sp[:, A] / Gr.n[A], Sp[:, B] / Gr.n[B]
        ba, bb = Sb[:, A] / np.maximum(Nb[:, A], 1e-12)[:, None], Sb[:, B] / np.maximum(Nb[:, B], 1e-12)[:, None]
        T = pd.DataFrame(dict(junction_id=post["junction_id"], event_id=post["event_id"][D.inv],
                              psi_group1=point[A], psi_group2=point[B], delta_psi=point[A] - point[B],
                              **{f"p_absdpsi_ge_{a.delta:g}": (np.abs(xa - xb) >= a.delta).mean(0),
                                 f"p_absdpsi_ge_{a.delta:g}_mouseboot": (np.abs(ba - bb) >= a.delta).mean(0)},
                              junction_reads_group1=y_g[A].astype(int), junction_reads_group2=y_g[B].astype(int),
                              event_reads_group1=n_g[A].astype(int), event_reads_group2=n_g[B].astype(int),
                              units_group1=int(Gr.n[A]), units_group2=int(Gr.n[B]),
                              units_with_event_reads_group1=cov[A].astype(int), units_with_event_reads_group2=cov[B].astype(int)))
        f = os.path.join(a.out, f"{p}.tsv.gz")
        T.to_csv(f, sep="\t", index=False, float_format="%.6g")
        if a.combined:
            X = pd.DataFrame(dict(pair=p, junction_id=T.junction_id, score=T[score_col], dpsi=T.delta_psi))
            assert X.notna().all().all() and X.score.between(0, 1).all(), f"{p}: table does not meet the spec"
            X.to_csv(a.combined, sep="\t", index=False, header=False, mode="a", float_format="%.6g")
        man.append(dict(pair=p, file=os.path.basename(f), rows=len(T), units_group1=int(Gr.n[A]), units_group2=int(Gr.n[B]), md5=md5(f)))
    pd.DataFrame(man).to_csv(os.path.join(a.out, "MANIFEST.tsv"), sep="\t", index=False)
    if a.combined:
        C.log(f"combined table {a.combined}: {len(pairs) * D.J:,} rows, md5 {md5(a.combined)}")
    json.dump(dict(data=a.data, data_md5=md5(a.data), combined=a.combined, score=score_col if a.combined else None,
                   combined_md5=md5(a.combined) if a.combined else None, fit_dir=a.fit_dir, fit_md5=md5(os.path.join(a.fit_dir, "fit.npz")),
                   pairs_file=a.pairs, pairs=len(pairs), draws=C.S_DRAWS, draw_seed=0, delta=a.delta,
                   broad_cell_type_recomputed=broad_rec, columns=list(T.columns)),
              open(os.path.join(a.out, "export_record.json"), "w"), indent=1)
    C.log(f"{len(pairs)} pair tables -> {a.out}")


if __name__ == "__main__":
    main()
