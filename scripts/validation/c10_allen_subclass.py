"""Official Allen subclass for each Allen cell group of check 10 (Karin, 2026-11-01; not part of the pre-registered computation).

    python c10_allen_subclass.py            # writes tsv/c10_Allen_group_subclass.tsv
    python c10_allen_subclass.py export     # adds official_subclass columns to c10/export/catalog_Allen_s0.tsv.gz (after the run)

Check 10 groups Allen cells by "cell groups (cluster names)": the Allen cluster label without its number (as in SpliceVI
STAGE5 s111). The official `subclass_label` of the Allen metadata is not in the model-ready files; each cluster label maps
to exactly one official subclass (382 clusters), so a cell group maps to the subclasses of its TRAIN cells' clusters.
Columns: cell_group, official_subclass (majority over the group's TRAIN cells), official_subclass_fraction, all subclasses.
"""
import os, sys
import numpy as np, pandas as pd, h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

META = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"
OUT = os.path.join(vlib.V, "tsv", "c10_Allen_group_subclass.tsv")


def mapping():
    m = pd.read_csv(META, usecols=["cluster_label", "subclass_label"], low_memory=False).drop_duplicates()
    assert m.cluster_label.is_unique, "a cluster maps to more than one subclass"
    c2s = dict(zip(m.cluster_label, m.subclass_label))
    with h5py.File(vlib.paths("Allen")["train"], "r") as f:
        cl = vlib._strings(f["obs"]["cell_ontology_class"]).astype(str)
    grp = np.array([x.split("_", 1)[1] if "_" in x else x for x in cl])
    sub = pd.Series(cl).map(c2s)
    rows = []
    for g in np.unique(grp):
        v = sub[grp == g].value_counts()
        rows.append(dict(cell_group=g, train_cells=int(v.sum()), official_subclass=v.index[0],
                         official_subclass_fraction=round(v.iloc[0] / v.sum(), 4),
                         all_official_subclasses="; ".join(f"{k} ({n})" for k, n in v.items())))
    D = pd.DataFrame(rows); D.to_csv(OUT, sep="\t", index=False)
    print(f"{len(D)} cell groups; {int((D.official_subclass_fraction < 1).sum())} span more than one official subclass -> {OUT}")
    return D


def export():
    D = pd.read_csv(OUT, sep="\t").set_index("cell_group")
    p = os.path.join(vlib.V, "c10", "export", "catalog_Allen_s0.tsv.gz")
    X = pd.read_csv(p, sep="\t")
    X["official_subclass"] = X.cell_type.map(D.official_subclass)
    X["official_subclass_fraction"] = X.cell_type.map(D.official_subclass_fraction)
    X.to_csv(p + ".tmp.gz", sep="\t", index=False, float_format="%.4g"); os.replace(p + ".tmp.gz", p)
    print(f"added official_subclass to {p}")


if __name__ == "__main__":
    export() if len(sys.argv) > 1 and sys.argv[1] == "export" else mapping()
