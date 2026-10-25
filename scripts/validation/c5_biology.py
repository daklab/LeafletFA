"""Check 5a (CPU): biological sanity of the TMS programs that replicate across seeds.

    python c5_biology.py

Programs: the seed-0 TMS fit; "replicating" = the check 3 centred-psi rule
(tsv/c3_TMS_seed_replication_centred.tsv). Per program:
  - cell types using it: mean TRAIN phi by label (tissue | broad type), top 5
  - ATSE contrast for program k = total variation of its composition from the pi-weighted mean of
    the OTHER programs, 0.5 * sum_{j in e} |psi_kj - psibar_(-k)j|; top 20 ATSEs and their genes
  - pre-registered known events: hit = gene in the program's top 1 % ATSEs AND the program's top-5
    cell types contain the expected lineage
Every live program is reported; replicating ones are flagged. Writes tsv/c5_programs.tsv,
tsv/c5_top_atses.tsv, tsv/c5_known_events.tsv.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

KNOWN = {  # gene set -> lineage keywords (matched case-insensitively in tissue | broad type labels)
    "Ptprc (lymphoid)": (["Ptprc"], ["t cell", "b cell", "natural killer", "lymph", "leukocyte"]),
    "Fgfr2 IIIb/IIIc (epithelial vs mesenchymal)": (["Fgfr2"], ["epithel", "keratinocyte", "basal", "fibroblast", "mesenchym", "stromal"]),
    "muscle exon (Tpm1/Tpm2/Mef2c/Ldb3/Ttn)": (["Tpm1", "Tpm2", "Mef2c", "Ldb3", "Ttn"], ["muscle", "myocyte", "cardiac", "cardiomyocyte", "satellite"]),
    "neuronal exon (Clta/Nrxn1/Ptbp2/Kif1bp)": (["Clta", "Nrxn1", "Ptbp2", "Kif1bp"], ["neuron"]),
}


def main():
    P = vlib.paths("TMS")
    import h5py
    with h5py.File(P["train"], "r") as f:
        var = vlib._splicing_root(f)["var"]
        gene = vlib._strings(var["gene_name"]).astype(str)
    _, ev = vlib.read_var(P["train"]); ev_u, inv = vlib.event_index(ev)
    cells, lab, _ = vlib.obs_labels(P["train"])
    z = np.load(os.path.join(vlib.B, "TMS", "full_K20_s0_f64", "fit", "fit.npz"), allow_pickle=True)
    assert np.array_equal(z["cell_id"].astype(str), cells)
    psi, phi, pi = z["psi_mean"].astype(np.float64), z["phi"].astype(np.float64), np.asarray(z["pi"], np.float64)
    K = psi.shape[0]
    rep = pd.read_csv(os.path.join(vlib.V, "tsv", "c3_TMS_seed_replication_centred.tsv"), sep="\t")
    replicating = set(rep.loc[rep["replicates"], "program_seed0"].astype(int))
    occ = np.bincount(phi.argmax(1), minlength=K) / len(phi)
    labs = pd.Series(lab)
    mean_phi = pd.DataFrame(phi).groupby(labs.values).mean()                  # labels x K
    n_lab = labs.value_counts()
    mean_phi = mean_phi.loc[n_lab[n_lab >= 20].index]

    ev_gene = pd.Series(gene).groupby(inv).agg(lambda g: ",".join(sorted(set(g))))
    prog_rows, top_rows, known_rows = [], [], []
    for k in range(K):
        if occ[k] < 0.01:
            continue
        w = np.delete(pi, k); w = w / w.sum()
        psibar = w @ np.delete(psi, k, axis=0)
        tv = np.bincount(inv, weights=np.abs(psi[k] - psibar), minlength=len(ev_u)) * 0.5
        order = np.argsort(-tv)
        top1 = set(order[: max(1, int(round(0.01 * len(order))))])
        ct = mean_phi[k].sort_values(ascending=False)
        top5 = list(ct.index[:5])
        prog_rows.append(dict(program=k + 1, replicates=(k + 1) in replicating, pi=pi[k], live_frac=occ[k],
                              top_cell_types="; ".join(f"{c} ({ct[c]:.2f})" for c in top5),
                              top_genes="; ".join(ev_gene[e] for e in order[:20])))
        for r, e in enumerate(order[:20]):
            top_rows.append(dict(program=k + 1, rank=r + 1, atse=ev_u[e], genes=ev_gene[e], tv=tv[e]))
        for name, (genes, kws) in KNOWN.items():
            in_top = [e for e in top1 if any(g in ev_gene[e].split(",") for g in genes)]
            lineage = any(any(kw in c.lower() for kw in kws) for c in top5)
            known_rows.append(dict(program=k + 1, replicates=(k + 1) in replicating, event=name,
                                   in_top1pct=len(in_top) > 0, atses=";".join(ev_u[e] for e in in_top),
                                   lineage_in_top5=lineage, hit=(len(in_top) > 0) and lineage))
    out = os.path.join(vlib.V, "tsv")
    pr = pd.DataFrame(prog_rows); pr.to_csv(os.path.join(out, "c5_programs.tsv"), sep="\t", index=False)
    pd.DataFrame(top_rows).to_csv(os.path.join(out, "c5_top_atses.tsv"), sep="\t", index=False)
    kn = pd.DataFrame(known_rows); kn.to_csv(os.path.join(out, "c5_known_events.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(pr[["program", "replicates", "pi", "live_frac", "top_cell_types"]].to_string(index=False))
    hits = kn[kn["hit"]]
    print(hits[["program", "replicates", "event", "atses"]].to_string(index=False))
    print("events hit (any live program):", hits["event"].nunique(), "/ 4;",
          "in a replicating program:", hits.loc[hits["replicates"], "event"].nunique(), "/ 4")


if __name__ == "__main__":
    main()
