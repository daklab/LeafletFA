"""10.6 scQuint benchmark (TMS TRAIN; settings from SpliceVI §170: stage5_s170_scquint.py; scQuint env python 3.7, scvi 0.5.0, commit cd255a1). Training only, CPU.
Counts: SpliceVI §170 dump, verified identical to the check 10 TMS TRAIN matrix (2026-11-05). Runtime and memory: sacct on this job, c10_benchmark_record.tsv."""
import os, time, numpy as np, pandas as pd, scipy.sparse as sp, torch, anndata as ad
from scquint.data import group_introns
from scquint.dimensionality_reduction import vae as V
I = "/gpfs/commons/home/kisaev/SpliceVI-utils/manuscript/age_analysis/audit_2026_09/screen_evals/s170/emb/inputs/"; DS = "TMS"
torch.manual_seed(20261170); np.random.seed(20261170); t0 = time.time()
var = pd.read_csv(I + DS + "_var.tsv", sep="\t"); ids = var.iloc[:, 0].astype(str).values; var = var.iloc[:, 1:]; var.columns = [c.split(".")[0] for c in var.columns]; var.index = ids
p = var.junction_id.astype(str).str.rsplit("_", n=3, expand=True); var["chromosome"] = p[0].values; var["start"] = p[1].astype(int).values; var["end"] = p[2].astype(int).values; var["strand"] = p[3].values; var["jpos"] = np.arange(len(var))
a = ad.AnnData(X=sp.csr_matrix((1, len(var)), dtype=np.float32), var=var[["chromosome", "start", "end", "strand", "gene_id", "gene_name", "event_id", "jpos"]].copy())
g = group_introns(a, by="three_prime"); keep = g.var.jpos.values.astype(int); print("junctions kept", len(keep), "of", len(var), "clusters", g.var.intron_group.nunique(), flush=True)
K = sp.load_npz(I + DS + "_TRAIN_counts.npz").tocsr()[:, keep].astype(np.float32); x = ad.AnnData(X=K, var=g.var.copy()); x.obs_names = [str(i) for i in range(x.n_obs)]
t1 = time.time(); lat, model = V.run_vae(x, use_cuda=False); t2 = time.time()
print("cells", x.n_obs, "latent", lat.shape, "prep_s %.1f train_s %.1f total_s %.1f" % (t1 - t0, t2 - t1, t2 - t0), flush=True)
