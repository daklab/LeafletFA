"""Allen sense checks (descriptive; VALIDATION_PREREG "Record — Allen sense checks", 2026-11-16).
    python allen_sense.py     -> $V/allen_sense/{controls,lists,hierarchy,frame}*.tsv, summary.json, plots
1 positive controls (Nrxn1-3 SS4, Gria2 flip/flop, Grin1 exon 5) in excitatory x inhibitory pairs; 2 neuron x glia enrichment
of Irimia 2014 neural microexons and Rbfox 2014 target exons; 3 hierarchy (pair sets, clustering vs taxonomy); 4 frame."""
import gzip, json, os, sys
import numpy as np, pandas as pd, h5py
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact, spearmanr, pearsonr
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, save  # noqa: E402

V = vlib.V; OUT = os.path.join(V, "allen_sense"); SRC = os.path.join(OUT, "sources")
PRE = os.path.join(V, "portal_browser", "precomputed")
META = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"
MIN_READS, MIN_CELLS, TOP, TOP_SENS, BG_EXCL = 20, 3, 200, 1000, 1000
GLIA = ("Astro", "Oligo", "Micro-PVM")
CONTROLS = {  # exon (1-based, mm10, GENCODE vM19), gene, expected sign of inclusion (excitatory - inhibitory) or None
    "Nrxn1 SS4": ("chr17", 90163792, 90163881, -1), "Nrxn2 SS4": ("chr19", 6513716, 6513805, -1), "Nrxn3 SS4": ("chr12", 90199147, 90199236, -1),
    "Gria2 flip": ("chr3", 80690404, 80690518, None), "Gria2 flop": ("chr3", 80691320, 80691434, None), "Grin1 exon 5 (N1)": ("chr2", 25311343, 25311405, None)}


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- inputs
def load():
    zf = np.load(os.path.join(V, "c10", "Allen_s0_estimates.npz"), allow_pickle=True)
    z = {k: zf[k] for k in ("types", "point", "reads", "cov")}                       # load once (NpzFile re-reads on every access)
    f = np.load(os.path.join(vlib.B, "Allen", "full_K20_s0_f64", "fit", "fit.npz"), allow_pickle=True)
    J = pd.read_parquet(os.path.join(PRE, "catalog_junctions.parquet"))
    assert np.array_equal(J.junction_id.values.astype(str), f["junction_id"].astype(str)), "catalog junction order != Allen fit"
    ev = json.load(open(os.path.join(PRE, "catalog_events.json")))
    assert list(ev["event_id"]) == list(np.unique(J.event_id)) and z["point"].shape[1] == len(J)
    return z, J, ev


def taxonomy(types):
    m = pd.read_csv(META, usecols=["cluster_label", "class_label", "subclass_label"], low_memory=False).drop_duplicates()
    c2 = m.set_index("cluster_label")
    with h5py.File(vlib.paths("Allen")["train"], "r") as h:
        cl = vlib._strings(h["obs"]["cell_ontology_class"]).astype(str)
    grp = np.array([x.split("_", 1)[1] if "_" in x else x for x in cl])
    rows = []
    for t in types:
        s = pd.Series(cl[grp == t])
        rows.append(dict(group=t, cls=c2.loc[s, "class_label"].value_counts().index[0], subclass=c2.loc[s, "subclass_label"].value_counts().index[0],
                         cls_frac=c2.loc[s, "class_label"].value_counts(normalize=True).iloc[0]))
    return pd.DataFrame(rows).set_index("group")


# ---------------------------------------------------------------- per-pair ranking
class Ranker:
    def __init__(self, z, J):
        self.P = z["point"]; self.R = z["reads"]; self.C = z["cov"]; self.inv = J.event_idx.values.astype(np.int64)
        self.order = np.argsort(self.inv, kind="stable"); inv_s = self.inv[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(inv_s)) + 1]; self.n_atse = len(self.starts)
        assert np.array_equal(inv_s[self.starts], np.arange(self.n_atse))
        self.inv_s = inv_s

    def pair(self, i, j):
        U = np.flatnonzero((self.R[i] >= MIN_READS) & (self.R[j] >= MIN_READS) & (self.C[i] >= MIN_CELLS) & (self.C[j] >= MIN_CELLS))
        d = (self.P[i] - self.P[j])[self.order]; ad = np.abs(d)
        o = np.lexsort((-ad, self.inv_s)); first = o[self.starts]                       # per ATSE: position of the largest |d|
        lead = self.order[first]; stat = ad[first]
        return U, stat[U], lead[U], np.minimum(self.R[i, U], self.R[j, U])


def exon_to_atse(J, chrom, s, e):
    """note 5 convention: junction ends at exon start - 1 (b) or starts at exon end (a). Returns (atse, inclusion junction) or None"""
    c = J[J.chromosome == chrom]
    L = c[c.b == s - 1]; Rr = c[c.a == e]
    if len(L):
        return int(L.event_idx.iloc[0]), int(L.index[0])
    if len(Rr):
        return int(Rr.event_idx.iloc[0]), int(Rr.index[0])
    return None


# ---------------------------------------------------------------- liftover (UCSC chain, point mapping)
def load_chain(path):
    ch = {}
    with gzip.open(path, "rt") as fh:
        hdr = None
        for line in fh:
            p = line.split()
            if not p:
                continue
            if p[0] == "chain":
                hdr = dict(t=p[2], ts=int(p[5]), q=p[7], qstrand=p[9], qs=int(p[10])); tp, qp = hdr["ts"], hdr["qs"]; continue
            size = int(p[0])
            if hdr["qstrand"] == "+":
                ch.setdefault(hdr["t"], []).append((tp, tp + size, hdr["q"], qp - tp))
            if len(p) == 3:
                tp += size + int(p[1]); qp += size + int(p[2])
            else:
                tp += size; qp += size
    return {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v]), [x[2] for x in v], np.array([x[3] for x in v])) for k, v in
            ((k, sorted(v)) for k, v in ch.items())}


def lift(ch, chrom, pos1):
    """1-based position -> (chrom, 1-based) or None"""
    if chrom not in ch:
        return None
    st, en, qc, off = ch[chrom]; p0 = pos1 - 1; i = np.searchsorted(st, p0, side="right") - 1
    if i < 0 or p0 >= en[i]:
        return None
    return qc[i], p0 + off[i] + 1


def read_lists():
    ir = pd.read_excel(os.path.join(SRC, "Irimia2014_mmc2.xlsx"), sheet_name="TableS2-Mouse", header=1)
    ir.columns = ["gene", "coord", "length", "full", "type", "regulation"]
    xy = ir.coord.astype(str).str.extract(r"(chr[^:]+):(\d+)-(\d+)"); n0 = len(ir); ok = xy[1].notna()
    ir, xy = ir[ok].copy(), xy[ok]; log(f"Irimia rows without a parseable coordinate dropped: {n0 - len(ir)} of {n0}")
    ir["chrom"], ir["s"], ir["e"] = xy[0], xy[1].astype(int), xy[2].astype(int)
    ir["expected"] = ir.regulation.map({"NEURAL-UP": 1, "NEURAL-DOWN": -1})
    ir["list"] = np.where((ir.type == "AltEx") & ir.length.between(3, 27), "Irimia2014 microexons (3-27 nt)",
                          np.where(ir.type == "AltEx", "Irimia2014 neural AltEx > 27 nt", "Irimia2014 other type"))
    rb = pd.read_excel(os.path.join(SRC, "WeynVanhentenryck2014_mmc2.xlsx"), sheet_name="6 BN target exons", header=2)
    rb = rb[(rb["AS type"] == "cassette") & rb.exonStart.notna() & rb.exonEnd.notna()].copy()
    rb = rb.rename(columns={"gene symbol": "gene"}); rb["chrom"], rb["s0"], rb["e"] = rb["chrom"], rb["exonStart"].astype(int), rb["exonEnd"].astype(int)
    rb["expected"] = pd.to_numeric(rb["direction"], errors="coerce"); rb["list"] = "Rbfox2014 BN target cassettes"   # '?' = no direction
    log(f"Rbfox direction values: {rb['direction'].astype(str).value_counts().to_dict()}")
    return ir, rb


def map_list(df, J, ch, starts):
    """try (build as-is | mm9->mm10) x start offsets; keep the combination matching most exons (annotation only)"""
    best = None
    for build in ("as_is", "mm9_to_mm10"):
        for off in starts:
            hits = []
            for r in df.itertuples():
                s, e = r.s0 + off if hasattr(r, "s0") else r.s + off, r.e
                c = r.chrom
                if build == "mm9_to_mm10":
                    a, b = lift(ch, c, s), lift(ch, c, e)
                    if a is None or b is None or a[0] != b[0] or b[1] - a[1] != e - s:
                        hits.append(None); continue
                    c, s, e = a[0], a[1], b[1]
                hits.append(exon_to_atse(J, c, s, e))
            n = sum(h is not None for h in hits)
            log(f"  {df['list'].iloc[0][:30]}: build {build}, start offset {off}: {n} of {len(df)} matched")
            if best is None or n > best[0]:
                best = (n, build, off, hits)
    df = df.copy(); df["atse"] = [h[0] if h else -1 for h in best[3]]; df["incl_junction"] = [h[1] if h else -1 for h in best[3]]
    return df, dict(build=best[1], start_offset=best[2], matched=best[0], rows=len(df))


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True); apply_theme(); import matplotlib.pyplot as plt
    z, J, ev = load(); types = z["types"].astype(str); T = taxonomy(types); T.to_csv(os.path.join(OUT, "taxonomy.tsv"), sep="\t")
    rk = Ranker(z, J); G = len(types); cls = T.loc[types, "cls"].values; sub = T.loc[types, "subclass"].values
    neuron = np.isin(cls, ["Glutamatergic", "GABAergic"]); glia = np.isin(types, GLIA)
    log(T.cls.value_counts().to_string())
    # junction count per ATSE
    njun = np.bincount(J.event_idx.values, minlength=rk.n_atse)
    # ---- all pairs ----
    pairs = []; store = {}
    for i in range(G):
        for j in range(i + 1, G):
            if cls[i] == "Glutamatergic" and cls[j] == "GABAergic" or cls[j] == "Glutamatergic" and cls[i] == "GABAergic":
                kind = "excitatory vs inhibitory"
            elif (neuron[i] and glia[j]) or (neuron[j] and glia[i]):
                kind = "neuron vs glia"
            elif sub[i] == sub[j]:
                kind = "close (same subclass)"
            elif cls[i] == cls[j]:
                kind = "same class, other subclass"
            else:
                kind = "other"
            U, st, ld, mr = rk.pair(i, j); store[(i, j)] = (U, st, ld, mr)
            pairs.append(dict(i=i, j=j, A=types[i], B=types[j], kind=kind, units=len(U), n_ge_0_1=int((st >= 0.1).sum()), n_ge_0_2=int((st >= 0.2).sum()),
                              frac_ge_0_1=float((st >= 0.1).mean()) if len(U) else np.nan, frac_ge_0_2=float((st >= 0.2).mean()) if len(U) else np.nan,
                              top200_median=float(np.median(np.sort(st)[::-1][:TOP])) if len(U) else np.nan,
                              mean_abs_dpsi=float(st.mean()) if len(U) else np.nan))
    Pr = pd.DataFrame(pairs); Pr.to_csv(os.path.join(OUT, "hierarchy_pairs.tsv"), sep="\t", index=False, float_format="%.4f")
    edges = np.quantile(np.concatenate([v[3] for v in store.values()]), np.linspace(0, 1, 11)[1:-1])
    jb = lambda n: np.minimum(n, 4) - 2                                                    # 2, 3, >= 4 -> 0, 1, 2

    def top_bg(key):
        U, st, ld, mr = store[key]; o = np.argsort(-st, kind="stable"); rank = np.empty(len(U), int); rank[o] = np.arange(1, len(U) + 1)
        strat = np.searchsorted(edges, mr) * 3 + jb(njun[U])
        return U, st, ld, rank, strat
    S = {}
    # ---- 1. positive controls ----
    ctl = {k: exon_to_atse(J, *v[:3]) for k, v in CONTROLS.items()}; log(f"controls: {ctl}")
    rows = []
    for (i, j), _ in store.items():
        if Pr.loc[(Pr.i == i) & (Pr.j == j), "kind"].iloc[0] != "excitatory vs inhibitory":
            continue
        ex, inh = (i, j) if cls[i] == "Glutamatergic" else (j, i)
        U, st, ld, rank, _ = top_bg((i, j)); pos = {u: k for k, u in enumerate(U)}
        for name, m in ctl.items():
            if m is None or m[0] not in pos:
                continue
            k = pos[m[0]]
            rows.append(dict(control=name, excitatory=types[ex], inhibitory=types[inh], units=len(U), rank=int(rank[k]), percentile=rank[k] / len(U),
                             in_top200=bool(rank[k] <= TOP), abs_dpsi_lead=st[k], d_inclusion_exc_minus_inh=z["point"][ex, m[1]] - z["point"][inh, m[1]],
                             expected_sign=CONTROLS[name][3]))
    C1 = pd.DataFrame(rows); C1.to_csv(os.path.join(OUT, "controls_pairs.tsv"), sep="\t", index=False, float_format="%.4f")
    g1 = C1.groupby("control").agg(pairs=("rank", "size"), median_percentile=("percentile", "median"), share_top200=("in_top200", "mean"),
                                   median_rank=("rank", "median"), share_negative=("d_inclusion_exc_minus_inh", lambda x: float((x < 0).mean())),
                                   median_d_inclusion=("d_inclusion_exc_minus_inh", "median")).reset_index()
    g1["atse"] = g1.control.map(lambda c: ev["event_id"][ctl[c][0]] if ctl[c] else None)
    g1["inclusion_junction"] = g1.control.map(lambda c: J.junction_id.iloc[ctl[c][1]] if ctl[c] else None)
    g1["expected_sign"] = g1.control.map(lambda c: CONTROLS[c][3]); g1.to_csv(os.path.join(OUT, "controls_summary.tsv"), sep="\t", index=False, float_format="%.4f")
    S["controls_unmatched"] = [k for k, v in ctl.items() if v is None]; log(g1.to_string())
    # ---- 2. lists, neuron vs glia ----
    ch = load_chain(os.path.join(SRC, "mm9ToMm10.over.chain.gz")); ir, rb = read_lists()
    ir_m, ir_info = map_list(ir, J, ch, (0,)); rb_m, rb_info = map_list(rb, J, ch, (0, 1))
    S["list_mapping"] = dict(irimia=ir_info, rbfox=rb_info)
    L = pd.concat([ir_m[["gene", "list", "expected", "atse", "incl_junction", "length"]], rb_m[["gene", "list", "expected", "atse", "incl_junction"]]], ignore_index=True)
    L.to_csv(os.path.join(OUT, "lists_mapped.tsv"), sep="\t", index=False)
    rows2 = []
    for lst in sorted(L.list.unique()):
        Ls = L[(L.list == lst) & (L.atse >= 0)].drop_duplicates("atse"); ids = set(Ls.atse); inc = dict(zip(Ls.atse, Ls.incl_junction)); exp = dict(zip(Ls.atse, Ls.expected))
        for topn in (TOP, TOP_SENS):
            O = E = 0.0; a = b = c = d = 0; folds = []; agree = n_dir = 0
            for (i, j), _ in store.items():
                if Pr.loc[(Pr.i == i) & (Pr.j == j), "kind"].iloc[0] != "neuron vs glia":
                    continue
                nr, gl = (i, j) if neuron[i] else (j, i)
                U, st, ld, rank, strat = top_bg((i, j)); lis = np.isin(U, list(ids)); top = rank <= topn; bg = rank > BG_EXCL
                o = lis[top].sum(); e_ = 0.0
                for s_ in np.unique(strat[top]):
                    bs = bg & (strat == s_)
                    if bs.any():
                        e_ += (top & (strat == s_)).sum() * lis[bs].mean()
                O += o; E += e_; folds.append(o / e_ if e_ > 0 else np.nan)
                a += o; b += (top & ~lis).sum(); c += (bg & lis).sum(); d += (bg & ~lis).sum()
                for u in U[top & lis]:
                    if exp[u] in (1, -1):                                                   # direction known in the source
                        dd = z["point"][nr, inc[u]] - z["point"][gl, inc[u]]; n_dir += 1; agree += np.sign(dd) == np.sign(exp[u])
            orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows2.append(dict(list=lst, top=topn, listed_atses_in_catalog=len(ids), observed_in_top=int(O), expected_matched=E, fold_matched=O / E if E else np.nan,
                              median_pair_fold=float(np.nanmedian(folds)), fisher_or_unweighted=orr, fisher_p_greater=p, direction_calls=n_dir,
                              share_expected_direction=agree / n_dir if n_dir else np.nan))
    C2 = pd.DataFrame(rows2); C2.to_csv(os.path.join(OUT, "lists_enrichment.tsv"), sep="\t", index=False, float_format="%.4g"); log(C2.to_string())
    # ---- 3. hierarchy ----
    h3 = Pr.groupby("kind").agg(pairs=("units", "size"), median_units=("units", "median"), median_n_ge_0_1=("n_ge_0_1", "median"),
                                median_frac_ge_0_1=("frac_ge_0_1", "median"), median_frac_ge_0_2=("frac_ge_0_2", "median"),
                                median_top200_dpsi=("top200_median", "median"), median_mean_abs_dpsi=("mean_abs_dpsi", "median")).reset_index()
    h3.to_csv(os.path.join(OUT, "hierarchy_summary.tsv"), sep="\t", index=False, float_format="%.4f"); log(h3.to_string())
    Dm = np.zeros((G, G))
    for r in Pr.itertuples():
        Dm[r.i, r.j] = Dm[r.j, r.i] = r.mean_abs_dpsi
    Z = linkage(squareform(Dm, checks=False), "average")
    tax = np.where(sub[:, None] == sub[None, :], 0, np.where(cls[:, None] == cls[None, :], 1, 2)).astype(float)
    coph = cophenet(Z); tv = squareform(tax, checks=False)
    S["clustering"] = dict(ari_class_k3=adjusted_rand_score(cls, fcluster(Z, 3, "maxclust")),
                           ari_subclass=adjusted_rand_score(sub, fcluster(Z, len(set(sub)), "maxclust")), n_subclasses=len(set(sub)),
                           spearman_cophenetic_vs_taxonomy=spearmanr(coph, tv)[0], pearson_cophenetic_vs_taxonomy=pearsonr(coph, tv)[0],
                           spearman_distance_vs_taxonomy=spearmanr(squareform(Dm, checks=False), tv)[0])
    log(S["clustering"])
    # ---- 4. frame ----
    cl = ev["classes"]; vlen = np.full(rk.n_atse, -1)
    for k, eid in enumerate(ev["event_id"]):
        c = cl.get(eid)
        if c == "skipped_exon":
            s_ = ev["structure"].get(eid)
            if s_ and len(s_) == 3:
                vlen[k] = s_[1][1] - s_[1][0] + 1
        elif c in ("alt5", "alt3") and njun[k] == 2:
            jj = J[J.event_idx == k]
            if len(jj) == 2:
                a1, a2 = jj.a.values; b1, b2 = jj.b.values
                vlen[k] = abs(b1 - b2) if a1 == a2 else (abs(a1 - a2) if b1 == b2 else -1)
    acls = np.array([cl.get(e) for e in ev["event_id"]]); rows4 = []
    for scope in ("all pairs", "neuron vs glia"):
        for cname in ("skipped_exon", "alt5", "alt3", "all three"):
            tot = {"top": [0, 0], "bg": [0.0, 0.0]}
            for (i, j), _ in store.items():
                if scope != "all pairs" and Pr.loc[(Pr.i == i) & (Pr.j == j), "kind"].iloc[0] != scope:
                    continue
                U, st, ld, rank, strat = top_bg((i, j))
                ok = (vlen[U] > 0) & (np.isin(acls[U], ["skipped_exon", "alt5", "alt3"]) if cname == "all three" else (acls[U] == cname))
                m3 = (vlen[U] % 3 == 0); top = (rank <= TOP) & ok; bg = (rank > BG_EXCL) & ok
                tot["top"][0] += m3[top].sum(); tot["top"][1] += top.sum()
                for s_ in np.unique(strat[top]):
                    bs = bg & (strat == s_)
                    if bs.any():
                        w = (top & (strat == s_)).sum(); tot["bg"][0] += w * m3[bs].mean(); tot["bg"][1] += w
            rows4.append(dict(scope=scope, ats_class=cname, top_occurrences=tot["top"][1], share_mod3_top=tot["top"][0] / max(tot["top"][1], 1),
                              share_mod3_matched_bg=tot["bg"][0] / max(tot["bg"][1], 1e-12)))
    C4 = pd.DataFrame(rows4); C4.to_csv(os.path.join(OUT, "frame.tsv"), sep="\t", index=False, float_format="%.4f"); log(C4.to_string())
    S["n_pairs"] = len(Pr); S["pair_kinds"] = Pr.kind.value_counts().to_dict(); S["taxonomy_class_counts"] = T.cls.value_counts().to_dict()
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    # ---- plots (one per check) ----
    col = {"excitatory vs inhibitory": PALETTE["indigo"], "neuron vs glia": PALETTE["teal"], "close (same subclass)": PALETTE["ochre"],
           "same class, other subclass": PALETTE["olive"], "other": PALETTE["grey"]}
    fig, ax = plt.subplots(figsize=figsize("single", h=2.4), constrained_layout=True)
    names = list(g1.control)
    for k, n in enumerate(names):
        x = C1[C1.control == n].percentile.values; jit = (np.random.default_rng(k).random(len(x)) - 0.5) * 0.3
        ax.scatter(x, k + jit, s=4, lw=0, color=PALETTE["teal"], alpha=0.5, rasterized=len(x) > 5000)
        ax.text(1.02, k, f"n = {len(x):,}", va="center", fontsize=6, transform=ax.get_yaxis_transform())
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Rank percentile within pair (0 = largest difference)", xlim=(0, 1))
    save(fig, os.path.join(OUT, "plot1_controls")); plt.close(fig)
    fig, ax = plt.subplots(figsize=figsize("single", h=2.2), constrained_layout=True)
    c2 = C2[C2.top == TOP]; ax.barh(range(len(c2)), c2.fold_matched, color=PALETTE["teal"], height=0.6)
    ax.axvline(1, color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)))
    ax.set(yticks=range(len(c2)), yticklabels=[l.replace("Irimia2014 ", "Irimia 2014 ").replace("Rbfox2014 ", "Rbfox 2014 ") for l in c2.list],
           xlabel="Fold enrichment in top 200 (neuron vs glia, matched)")
    save(fig, os.path.join(OUT, "plot2_lists")); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=figsize("double", h=6.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.3]))
    kinds =["neuron vs glia", "excitatory vs inhibitory", "same class, other subclass", "close (same subclass)"]
    for k, kd in enumerate(kinds):
        x = Pr[Pr.kind == kd].frac_ge_0_1.dropna().values
        ax[0].boxplot([x], positions=[k], widths=0.5, showfliers=False, medianprops=dict(color=col[kd]))
        ax[0].scatter(k + (np.random.default_rng(k).random(len(x)) - 0.5) * 0.3, x, s=3, lw=0, color=col[kd], alpha=0.25, rasterized=True)
        ax[0].text(k, 1.0, f"n = {len(x):,}", ha="center", fontsize=6, transform=ax[0].get_xaxis_transform())
    ax[0].set(xticks=range(len(kinds)), xticklabels=["Neuron vs\nglia", "Exc. vs\ninh.", "Same class,\nother subclass", "Same\nsubclass"],
              ylabel="Fraction of ATSEs with |Δψ| ≥ 0.1"); panel_label(ax[0], "a")
    from scipy.cluster.hierarchy import dendrogram
    dn = dendrogram(Z, labels=list(types), orientation="right", ax=ax[1], color_threshold=0, above_threshold_color=PALETTE["grey"], leaf_font_size=5)
    cc = {"Glutamatergic": PALETTE["indigo"], "GABAergic": PALETTE["ochre"], "Non-Neuronal": PALETTE["teal"]}
    for t in ax[1].get_yticklabels():
        t.set_color(cc[T.loc[t.get_text(), "cls"]])
    ax[1].set(xlabel="Mean |Δψ| (average linkage)"); panel_label(ax[1], "b")
    save(fig, os.path.join(OUT, "plot3_hierarchy")); plt.close(fig)
    fig, ax = plt.subplots(figsize=figsize("single", h=2.2), constrained_layout=True)
    c4 = C4[C4.scope == "all pairs"]; x = np.arange(len(c4))
    ax.bar(x - 0.18, c4.share_mod3_top, 0.36, color=PALETTE["teal"], label="Top 200")
    ax.bar(x + 0.18, c4.share_mod3_matched_bg, 0.36, color=PALETTE["grey"], label="Matched background")
    ax.axhline(1 / 3, color=PALETTE["grey"], lw=0.6, ls=(0, (3, 2)))
    ax.set(xticks=x, xticklabels=["Skipped exon", "Alt. 5′", "Alt. 3′", "All three"], ylabel="Share with length ≡ 0 (mod 3)", ylim=(0, 1))
    ax.legend(frameon=False, handlelength=1.2, labelspacing=0.35, borderpad=0.2)
    save(fig, os.path.join(OUT, "plot4_frame")); plt.close(fig)
    log("wrote " + OUT)


if __name__ == "__main__":
    main()
