"""Read-only (Karin 2026-11-05): is the Allen "mouse" column of check 10 and §176 a true donor, how concentrated are cell classes within it,
and what do genotype and FACS fields say? No model, no new runs. Writes c10_null_diag/allen_donor/*.tsv and prints the tables."""
import os, sys, numpy as np, pandas as pd, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
M = "/gpfs/commons/projects/knowles_singlecell_splicing/allen-brain/mouse_isocortex_hippocampal_2021/METADATA/metadata.csv"
OUT = os.path.join(vlib.V, "c10_null_diag", "allen_donor"); os.makedirs(OUT, exist_ok=True)
P = vlib.paths("Allen")
def obs(path, split):
    with h5py.File(path, "r") as f:
        o = f["obs"]; d = dict(cell=vlib._strings(o[o.attrs["_index"]]).astype(str), mouse=vlib._strings(o["mouse.id"]).astype(str),
                               coc=vlib._strings(o["cell_ontology_class"]).astype(str), cell_id=vlib._strings(o["cell_id"]).astype(str),
                               meta=vlib._strings(o["meta_sample_name"]).astype(str), obs_batch=vlib._strings(o["batch"]).astype(str))
        print(split, "obs columns", list(o.keys()))
    return pd.DataFrame(d).assign(split=split)
O = pd.concat([obs(P["train"], "TRAIN"), obs(P["test"], "TEST")], ignore_index=True)
print(O.head(3).to_string())
m = pd.read_csv(M, low_memory=False, dtype=str)
print("metadata rows", len(m), "platforms", m.platform_label.value_counts().to_dict())
# join key: try each candidate and take the one that matches every cell
best = None
for k in ("sample_name", "exp_component_name"):
    for c in ("cell_id", "meta"):
        n = O[c].isin(m[k]).mean(); print(f"join {c} -> {k}: {n:.4f}")
        if best is None or n > best[0]: best = (n, c, k)
print("join used", best); assert best[0] > 0.999, "no clean join"
J = O.merge(m.drop_duplicates(best[2]), left_on=best[1], right_on=best[2], how="left", validate="m:1", suffixes=("", "_m"))
print("mouse.id == external_donor_name_label:", (J.mouse == J.external_donor_name_label).mean())
print("cell_ontology_class == cell_type_alias_label:", (J.coc == J.cell_type_alias_label).mean())
J["batch"] = J.exp_component_name.str.split("_").str[0]   # library plate prefix (SM-xxxxx) of exp_component_name
J["full_genotype_label"] = J.full_genotype_label.fillna("missing")
J["group"] = [x.split("_", 1)[1] if "_" in x else x for x in J.coc]
cols = ["class_label", "subclass_label", "group", "full_genotype_label", "facs_population_plan_label", "injection_type_label", "injection_method_label", "region_label", "donor_sex_label", "batch"]
print(J[cols].nunique().to_string()); print("missing share\n", J[cols].isna().mean().round(4).to_string())
print("examples: donors", list(J.mouse.unique()[:6]), "exp_component", list(J.exp_component_name[:3]))
for tag, X in (("TRAIN", J[J.split == "TRAIN"]), ("ALL", J)):
    r = []
    for d, x in X.groupby("mouse"):
        vc = x.class_label.value_counts(normalize=True); sc = x.subclass_label.value_counts(normalize=True); gc = x.group.value_counts(normalize=True)
        r.append(dict(donor=d, cells=len(x), n_class=vc.size, top_class=vc.index[0], top_class_share=vc.iloc[0], n_subclass=sc.size, top_subclass_share=sc.iloc[0], n_group=gc.size, top_group_share=gc.iloc[0],
                      n_genotype=x.full_genotype_label.nunique(), genotype=x.full_genotype_label.mode().iloc[0], n_facs=x.facs_population_plan_label.nunique(), facs=x.facs_population_plan_label.mode().iloc[0], n_sex=x.donor_sex_label.nunique(), n_region=x.region_label.nunique(), n_batch=x.batch.nunique()))
    R = pd.DataFrame(r); R.to_csv(os.path.join(OUT, f"per_donor_{tag}.tsv"), sep="\t", index=False, float_format="%.4f")
    print(f"\n== {tag}: {len(R)} donors, {len(X)} cells")
    print(R[["cells", "n_class", "top_class_share", "n_subclass", "top_subclass_share", "n_group", "top_group_share", "n_genotype", "n_facs", "n_sex", "n_region", "n_batch"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).T.to_string())
    print("donors with 1 genotype", (R.n_genotype == 1).mean(), "1 sex", (R.n_sex == 1).mean(), "1 class", (R.n_class == 1).mean(), "top class>=0.9", (R.top_class_share >= .9).mean())
    print("cell-weighted top class share", np.average(R.top_class_share, weights=R.cells), "top subclass", np.average(R.top_subclass_share, weights=R.cells))
    print("batches spanning >1 donor:", (X.groupby("batch").mouse.nunique() > 1).mean(), "of", X.batch.nunique())
    for fld in ("full_genotype_label", "facs_population_plan_label"):
        g = X.groupby(fld); T = pd.crosstab(X[fld], X.class_label, normalize="index"); T.insert(0, "cells", g.size()); T.insert(1, "donors", g.mouse.nunique())
        T = T.sort_values("cells", ascending=False); T.to_csv(os.path.join(OUT, f"class_by_{fld}_{tag}.tsv"), sep="\t", float_format="%.4f")
        print(f"\n{fld}: {len(T)} values"); print(T.head(14).round(3).to_string())
    # how much of class is explained: Cramer's V and same-cell null from donor / genotype / facs
    from scipy.stats import chi2_contingency
    for fld in ("mouse", "full_genotype_label", "facs_population_plan_label", "region_label"):
        ct = pd.crosstab(X[fld], X.class_label); chi = chi2_contingency(ct)[0]; print(f"Cramer V class ~ {fld}: {np.sqrt(chi / ct.values.sum() / (min(ct.shape) - 1)):.3f} ({ct.shape[0]} levels)")
    # within a genotype, do donors share a class mix?  median top-class share by genotype and spread
    B = R.groupby("genotype").agg(donors=("donor", "size"), cells=("cells", "sum"), median_top_class_share=("top_class_share", "median")).sort_values("cells", ascending=False); B.to_csv(os.path.join(OUT, f"top_class_by_genotype_{tag}.tsv"), sep="\t", float_format="%.3f"); print(B.head(12).to_string())
