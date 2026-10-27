"""Diagnosis (CPU, read-only): why LeafletFA puts mesenchymal TMS TEST cells near 0.4 IIIb usage for Fgfr2
in SpliceVI Supp. Fig. S3, while raw reads and SpliceVI give ~0.

    python c7_fgfr2_s3_diagnosis.py

Event ENSMUSG00000030849.18_atse_3 (IIIb chr7_130199905_130200982_-, IIIc chr7_130198563_130200982_-).
Cells and reads: SpliceVI's S3 table (screen_evals/fig_s3/fgfr2_cells.tsv). Lineages as in the Genome
Biology build_fgfr2_example.py (MES / EPI keyword lists).
  1  prior and posterior IIIb usage with (a) the S3 program set (fit indices 0-8), (b) the stated rule
     (mean TRAIN phi > 0.01), (c) argmax >= 1 % of TRAIN cells, (d) all 20 programs
  2  per program: psi of IIIb, mean TRAIN phi, mean TEST phi in mesenchymal / epithelial cells, set membership
  3  the full-atlas K20 seed-0 fit (Genome Biology), which was trained on all cells including these
  4  mesenchymal cells: reads on the event; distance of projected phi from the prior mean pi
Writes tsv/c7_fgfr2_*.tsv.
"""
import os, sys
import numpy as np, pandas as pd, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa")
from _phi_rows import phi_row_of  # noqa: E402

S3 = "/gpfs/commons/home/kisaev/SpliceVI-utils/manuscript/age_analysis/audit_2026_09/screen_evals/fig_s3/fgfr2_cells.tsv"
JB = "chr7_130199905_130200982_-"
MES = ("FIBROBLAST", "MESENCHYMAL", "STROMAL", "SMOOTH MUSCLE", "PERICYTE", "SKELETAL MUSCLE", "CARDIAC MUSCLE", "VLMC")
EPI = ("EPITHELIAL", "KERATINOCYTE", "BASAL", "HEPATOCYTE", "DUCTAL", "INTESTINAL", "ISLET", "KIDNEY CELL", "LUMINAL")
ATLAS = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION"


def lineage(t):
    u = t.upper()
    return "mesenchymal" if any(k in u for k in MES) else ("epithelial" if any(k in u for k in EPI) else "other")


def scores(phi, psi, keep, jb, g, yb, n):
    p = phi[:, keep]; p = p / p.sum(1, keepdims=True)
    rho = p @ psi[keep, jb]
    return rho, (g * rho + yb) / (g + n)


def main():
    out = os.path.join(vlib.V, "tsv")
    D = pd.read_csv(S3, sep="\t")
    D["lineage"] = D.cell_type.map(lineage)
    yb, n = D.reads_IIIb.values.astype(float), D.reads_event.values.astype(float)
    B = os.path.join(vlib.B, "TMS", "full_K20_s0_f64")
    fit = np.load(os.path.join(B, "fit", "fit.npz"), allow_pickle=True)
    jb = int(np.flatnonzero(fit["junction_id"].astype(str) == JB)[0])
    psi, pi = fit["psi_mean"].astype(np.float64), np.asarray(fit["pi"], np.float64); pi = pi / pi.sum()
    g = float(fit["bb_conc_vec"][fit["inv_ev"][jb]])
    ptr = fit["phi"].astype(np.float64)
    fz = np.load(os.path.join(B, "factors_real.npz"), allow_pickle=True)
    pos = pd.Index(fz["cell_id"].astype(str)).get_indexer(D.cell.astype(str)); assert (pos >= 0).all()
    phi = fz["phi"].astype(np.float64)[pos]
    K = psi.shape[0]
    sets = {"S3 set (fit idx 0-8)": np.arange(9),
            "mean TRAIN phi > 0.01": np.flatnonzero(ptr.mean(0) > 0.01),
            "argmax >= 1% TRAIN": np.flatnonzero(np.bincount(ptr.argmax(1), minlength=K) / len(ptr) >= 0.01),
            "all 20": np.arange(K)}
    mes, epi, cov = (D.lineage == "mesenchymal").values, (D.lineage == "epithelial").values, n > 0
    rows = []
    for name, keep in sets.items():
        rho, post = scores(phi, psi, keep, jb, g, yb, n)
        for lin, m in (("mesenchymal", mes), ("epithelial", epi)):
            rows.append(dict(program_set=name, programs=",".join(str(k) for k in keep), lineage=lin, n_cells=int(m.sum()),
                             n_covered=int((m & cov).sum()), raw_IIIb_covered=np.nanmean(D.raw_IIIb.values[m & cov]),
                             prior_mean_all=rho[m].mean(), posterior_mean_all=post[m].mean(),
                             posterior_mean_covered=post[m & cov].mean(), posterior_median_all=np.median(post[m])))
    T1 = pd.DataFrame(rows)
    # check: S3 set reproduces S3's own column
    rho9, post9 = scores(phi, psi, np.arange(9), jb, g, yb, n)
    assert np.allclose(post9, D.leafletfa_live_posterior_IIIb.values, atol=1e-5)
    # 2 per-program
    P = pd.DataFrame(dict(program_fit_idx=np.arange(K), psi_IIIb=psi[:, jb], pi=pi, mean_train_phi=ptr.mean(0),
                          argmax_frac_train=np.bincount(ptr.argmax(1), minlength=K) / len(ptr),
                          mean_test_phi_mesenchymal=phi[mes].mean(0), mean_test_phi_epithelial=phi[epi].mean(0)))
    for name, keep in sets.items():
        P[f"in: {name}"] = P.program_fit_idx.isin(keep)
    # 3 full-atlas fit (trained on all cells, including these TEST cells)
    with h5py.File(f"{ATLAS}/MODEL_INPUT/20260526/train_splice_model_ready_2026-06-17.h5ad", "r") as f:
        ids = vlib._strings(f["obs"][f["obs"].attrs["_index"]]).astype(str)
        jid = vlib._strings(f["var"]["junction_id"]).astype(str)
    za = np.load(f"{ATLAS}/ATLAS_DM/runs_final_deltaonly/dm_simplex_K20_seed0_b8192/fit.npz", allow_pickle=True)
    prow = phi_row_of(za, len(ids)); arow = pd.Index(ids).get_indexer(D.cell.astype(str))
    have = (arow >= 0) & (prow[np.maximum(arow, 0)] >= 0)
    jba = int(np.flatnonzero(jid == JB)[0])
    pa = np.asarray(za["phi"]).astype(np.float64)[prow[arow[have]]]
    psa = np.asarray(za["psi_mean"]).astype(np.float64); ga = float(za["bb_conc_vec"][np.asarray(za["inv_ev"])[jba]])
    rows3 = []
    for name, keep in (("atlas all 20", np.arange(psa.shape[0])),
                       ("atlas mean phi > 0.01", np.flatnonzero(np.asarray(za["phi"]).mean(0) > 0.01))):
        rho, post = scores(pa, psa, keep, jba, ga, yb[have], n[have])
        for lin, m in (("mesenchymal", mes[have]), ("epithelial", epi[have])):
            rows3.append(dict(fit=name, lineage=lin, n_cells=int(m.sum()), prior_mean_all=rho[m].mean(),
                              posterior_mean_all=post[m].mean(), posterior_mean_covered=post[m & (n[have] > 0)].mean()))
    T3 = pd.DataFrame(rows3)
    # 4 mesenchymal cells: reads, phi vs prior mean
    tv = 0.5 * np.abs(phi - pi).sum(1)
    T4 = pd.DataFrame([dict(lineage=lin, n_cells=int(m.sum()), frac_with_reads=(n[m] > 0).mean(),
                            median_reads_if_covered=np.median(n[m & cov]) if (m & cov).any() else np.nan,
                            median_TV_phi_vs_pi=np.median(tv[m]), frac_TV_below_0_2=(tv[m] < 0.2).mean())
                       for lin, m in (("mesenchymal", mes), ("epithelial", epi), ("all", np.ones(len(D), bool)))])
    for t, nm in ((T1, "sets"), (P, "programs"), (T3, "atlas"), (T4, "coverage")):
        t.to_csv(os.path.join(out, f"c7_fgfr2_{nm}.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"gamma_e {g:.2f}; atlas fit covers {int(have.sum())} of {len(D)} cells")
    print(T1.drop(columns="programs").round(3).to_string(index=False))
    print(P.round(3).to_string(index=False))
    print(T3.round(3).to_string(index=False)); print(T4.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
