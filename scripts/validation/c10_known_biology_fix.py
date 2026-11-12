"""Check 10.4 bug fix (amendment 6, note 4): the parts of 10.4 that c10_catalog.py did not implement.

    python c10_known_biology_fix.py          # writes tsv/c10_known_biology_fix.json and tsv/c10_Allen_s0_nrxn_pairs.tsv
    python c10_known_biology_fix.py nrxn     # Nrxn part only, keeping the Ptprc result (note 5)
    python c10_known_biology_fix.py nrxn_ss4 # note 6: true SS4 (SpliceVI §111 exons / ATSEs), adds key nrxn_ss4; other keys kept

Reads the saved seed-0 check 10 outputs (c10/<ds>_s0_estimates.npz), the fits' junction / event order (fit.npz) and the TEST
counts; no refit, no new draws.
- Ptprc (TMS): flagged (B or T cell type, Ptprc ATSE) calls; direction confirmed if TEST pseudobulk psi(type) - psi(all other
  TEST cells) has the sign of psi_t - psi_-t at the call's lead junction.
- Nrxn1-3 SS4 (Allen): the ATSE holding the gene's single annotated 30-nt exon; pairs of neuronal cell groups with point TV > delta
  and non-overlapping 95 % intervals (primary intervals) at >= 1 junction.
"""
import itertools, json, os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, c9_lib as L  # noqa: E402
from c10_catalog import labels, fit_dir, lead_sign, CALL_READS, CALL_COV  # noqa: E402

T = os.path.join(vlib.V, "tsv")
META = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"
SS4 = {"Nrxn1": ("chr17", 90992115, 90992144), "Nrxn2": ("chr19", 6448255, 6448284), "Nrxn3": ("chr12", 88850279, 88850308)}
# note 6 (2026-11-11): the SS4 dict above holds the note 4 30-nt exons, which are NOT SS4 (early Nrxn-alpha site; kept for the
# record). True SS4 = SpliceVI STAGE5 §111 (fixed 2026-10-06): exon, ATSE, inclusion and skipping junction.
SS4_TRUE = {"Nrxn1": ("chr17", 90163792, 90163881, "ENSMUSG00000024109.18_atse_7", "chr17_90162433_90163791_-", "chr17_90162433_90208322_-"),
            "Nrxn2": ("chr19", 6513716, 6513805, "ENSMUSG00000033768.16_atse_6", "chr19_6509949_6513715_+", "chr19_6509949_6516922_+"),
            "Nrxn3": ("chr12", 90199147, 90199236, "ENSMUSG00000066392.11_atse_7", "chr12_90169091_90199146_+", "chr12_90169091_90204510_+")}
LYMPH = ("B CELL", "T CELL")                                                     # stated rule: B or T cells (note 4)


def load(ds):
    E = np.load(os.path.join(vlib.V, "c10", f"{ds}_s0_estimates.npz"), allow_pickle=True)
    z = np.load(os.path.join(fit_dir(ds, 0), "fit.npz"), allow_pickle=True)
    jid, ev, inv = z["junction_id"].astype(str), z["event_id"].astype(str), np.asarray(z["inv_ev"], np.int64)
    with h5py.File(vlib.paths(ds)["train"], "r") as f:
        v = vlib._splicing_root(f)["var"]
        assert np.array_equal(vlib._strings(v["junction_id"]).astype(str), jid)
        gene = vlib._strings(v["gene_name"]).astype(str)
    return E, jid, ev, inv, gene


def ptprc():
    E, jid, ev, inv, gene = load("TMS")
    types = E["types"].astype(str); n_atse = len(ev)
    _, tt, _ = labels("TMS", vlib.paths("TMS")["test"])
    Y = vlib.read_counts(vlib.paths("TMS")["test"]).tocsr()
    N = (Y @ sp.csr_matrix((np.ones(len(inv)), (np.arange(len(inv)), inv)), shape=(len(inv), n_atse))).tocsr()
    pe = np.unique(inv[gene == "Ptprc"]); pj = np.flatnonzero(np.isin(inv, pe))
    lj, sg = lead_sign(E["point"][:, pj], E["rest"][:, pj], inv[pj], n_atse)   # only Ptprc junctions (same result, fast)
    lj = np.where(lj >= 0, pj[np.maximum(lj, 0)], -1)
    Yp, Np = Y[:, pj].toarray(), N[:, pe].toarray()                             # TEST cells x Ptprc junctions / ATSEs
    ycol, ncol = {j: i for i, j in enumerate(pj)}, {e: i for i, e in enumerate(pe)}
    up = np.char.upper(types)
    rows = []
    for k in np.flatnonzero([any(x in u for x in LYMPH) for u in up]):
        for e in pe:
            if not E["specific"][k, e]:
                continue
            j = lj[k, e]; it = tt == types[k]
            yt, nt = Yp[it, ycol[j]].sum(), Np[it, ncol[e]].sum(); yr, nr = Yp[~it, ycol[j]].sum(), Np[~it, ncol[e]].sum()
            d = (yt / nt - yr / nr) if nt > 0 and nr > 0 else np.nan
            rows.append(dict(cell_type=types[k], event_id=ev[e], lead_junction=jid[j], train_sign=int(sg[k, e]),
                             psi_t=float(E["point"][k, j]), psi_rest=float(E["rest"][k, j]), test_cells=int(it.sum()),
                             test_reads_event=int(nt), test_diff=float(d), confirmed=bool(np.sign(d) == sg[k, e])))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(T, "c10_TMS_s0_ptprc_direction.tsv"), sep="\t", index=False)
    return dict(ptprc_calls_B_or_T=len(R), ptprc_confirmed_in_test=int(R.confirmed.sum()) if len(R) else 0,
                ptprc_pass=bool(len(R) and R.confirmed.any()))


def neuronal_groups(types):
    m = pd.read_csv(META, usecols=["cluster_label", "class_label"], low_memory=False).drop_duplicates()
    c2c = dict(zip(m.cluster_label, m.class_label))
    with h5py.File(vlib.paths("Allen")["train"], "r") as f:
        cl = vlib._strings(f["obs"]["cell_ontology_class"]).astype(str)
    grp = np.array([x.split("_", 1)[1] if "_" in x else x for x in cl]); cls = pd.Series(cl).map(c2c).values
    out = {}
    for t in types:
        v = pd.Series(cls[grp == t]).value_counts()
        out[t] = v.index[0] if len(v) else None
    return np.array([out[t] in ("Glutamatergic", "GABAergic") for t in types])


def nrxn():
    E, jid, ev, inv, gene = load("Allen")
    types = E["types"].astype(str); delta = float(E["delta"]); neu = neuronal_groups(types)
    parts = np.array([j.rsplit("_", 3) for j in jid]); ch, s, e_ = parts[:, 0], parts[:, 1].astype(int), parts[:, 2].astype(int)
    res, pair_rows = dict(delta=delta, neuronal_groups=int(neu.sum()), groups=len(types)), []
    for g, (c, a, b) in SS4.items():
        hit = (gene == g) & (ch == c) & ((e_ == a - 1) | (s == b) | ((s < a) & (e_ > b)))
        cand = np.unique(inv[hit])
        res[g] = dict(ss4_exon=f"{c}:{a}-{b}", candidate_atses=[ev[x] for x in cand])
        if len(cand) != 1:
            res[g]["result"] = f"{len(cand)} ATSEs qualify; not tested"; continue
        e = cand[0]; J = np.flatnonzero(inv == e); idx = np.flatnonzero(neu)
        ok = (E["reads"][:, e] >= CALL_READS) & (E["cov"][:, e] >= CALL_COV)
        n, n_rc = 0, 0
        for x, y in itertools.combinations(idx, 2):
            tvp = 0.5 * np.abs(E["point"][x, J] - E["point"][y, J]).sum()
            sep = np.any((E["lo"][x, J] > E["hi"][y, J]) | (E["lo"][y, J] > E["hi"][x, J]))
            d = bool(tvp > delta and sep); n += d; n_rc += d and ok[x] and ok[y]
            if d:
                pair_rows.append(dict(gene=g, event_id=ev[e], a=types[x], b=types[y], tv=tvp, read_cell_condition=bool(ok[x] and ok[y])))
        res[g].update(event_id=ev[e], junctions=[jid[j] for j in J], neuronal_pairs=len(idx) * (len(idx) - 1) // 2,
                      pairs_differ=int(n), pairs_differ_with_read_cell_condition=int(n_rc), pass_=bool(n >= 1))
    pd.DataFrame(pair_rows).to_csv(os.path.join(T, "c10_Allen_s0_nrxn_pairs.tsv"), sep="\t", index=False)
    return res


def nrxn_ss4():
    """note 6: the 10.4 Nrxn rule on the true SS4 ATSEs (SpliceVI §111); same pair rule and outputs as nrxn()."""
    E, jid, ev, inv, gene = load("Allen")
    types = E["types"].astype(str); delta = float(E["delta"])
    cls = _classes(types); neu = np.isin(cls, ["Glutamatergic", "GABAergic"])   # = neuronal_groups(types), one read of the file
    parts = np.array([j.rsplit("_", 3) for j in jid]); ch, s, e_ = parts[:, 0], parts[:, 1].astype(int), parts[:, 2].astype(int)
    jx = {x: i for i, x in enumerate(jid)}; ex = {x: i for i, x in enumerate(ev)}
    res, pair_rows = dict(delta=delta, neuronal_groups=int(neu.sum()), groups=len(types), definition="SpliceVI STAGE5 §111 (fixed 2026-10-06)"), []
    for g, (c, a, b, atse, inc, skp) in SS4_TRUE.items():
        hit = (gene == g) & (ch == c) & ((e_ == a - 1) | (s == b) | ((s < a) & (e_ > b)))    # note 5 rule, cross-check only
        cand = [ev[x] for x in np.unique(inv[hit])]
        e = ex[atse]
        assert inv[jx[inc]] == e and inv[jx[skp]] == e, (g, atse)
        J = np.flatnonzero(inv == e); idx = np.flatnonzero(neu)
        ok = (E["reads"][:, e] >= CALL_READS) & (E["cov"][:, e] >= CALL_COV)
        n, n_rc = 0, 0
        for x, y in itertools.combinations(idx, 2):
            tvp = 0.5 * np.abs(E["point"][x, J] - E["point"][y, J]).sum()
            sep = np.any((E["lo"][x, J] > E["hi"][y, J]) | (E["lo"][y, J] > E["hi"][x, J]))
            d = bool(tvp > delta and sep); n += d; n_rc += d and ok[x] and ok[y]
            if d:
                pair_rows.append(dict(gene=g, event_id=atse, a=types[x], b=types[y], tv=tvp, read_cell_condition=bool(ok[x] and ok[y])))
        res[g] = dict(ss4_exon=f"{c}:{a}-{b}", event_id=atse, inclusion_junction=inc, skipping_junction=skp,
                      note5_rule_candidate_atses=cand, junctions=[jid[j] for j in J], neuronal_pairs=len(idx) * (len(idx) - 1) // 2,
                      pairs_differ=int(n), pairs_differ_with_read_cell_condition=int(n_rc), pass_=bool(n >= 1),
                      inclusion_psi_median_by_neuronal_group=None)
        # descriptive (not part of the rule): median point psi of the inclusion junction, excitatory vs inhibitory groups
        pi = E["point"][:, jx[inc]]
        res[g]["inclusion_psi_median_excitatory"] = float(np.median(pi[(cls == "Glutamatergic") & ok])) if ((cls == "Glutamatergic") & ok).any() else None
        res[g]["inclusion_psi_median_inhibitory"] = float(np.median(pi[(cls == "GABAergic") & ok])) if ((cls == "GABAergic") & ok).any() else None
        res[g].pop("inclusion_psi_median_by_neuronal_group")
    pd.DataFrame(pair_rows).to_csv(os.path.join(T, "c10_Allen_s0_nrxn_pairs_ss4.tsv"), sep="\t", index=False)
    return res


def _classes(types):
    m = pd.read_csv(META, usecols=["cluster_label", "class_label"], low_memory=False).drop_duplicates()
    c2c = dict(zip(m.cluster_label, m.class_label))
    with h5py.File(vlib.paths("Allen")["train"], "r") as f:
        cl = vlib._strings(f["obs"]["cell_ontology_class"]).astype(str)
    grp = np.array([x.split("_", 1)[1] if "_" in x else x for x in cl]); cls = pd.Series(cl).map(c2c).values
    out = {}
    for t in types:
        v = pd.Series(cls[grp == t]).value_counts()
        out[t] = v.index[0] if len(v) else None
    return np.array([out[t] for t in types], dtype=object)


if __name__ == "__main__":
    F = os.path.join(T, "c10_known_biology_fix.json")
    if sys.argv[1:] == ["nrxn_ss4"]:
        out = json.load(open(F)); out["nrxn_ss4"] = nrxn_ss4()
        json.dump(out, open(F, "w"), indent=1, default=str)
        print(json.dumps(out["nrxn_ss4"], indent=1, default=str))
        sys.exit(0)
    out = dict(ptprc=json.load(open(F))["ptprc"], nrxn=nrxn()) if sys.argv[1:] == ["nrxn"] else dict(ptprc=ptprc(), nrxn=nrxn())
    json.dump(out, open(F, "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
