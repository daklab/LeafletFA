"""Per-step training cost of the new LeafletFA numerics on CPU, at the production batch size.

    python bench_step.py --out bench.json

For TMS and Allen TRAIN: 8,192 random cells (= one production minibatch), K 20, full-batch fit
steps timed after 2 warm-up steps. Arms: sparse + float64 phi at 16 and 8 threads, sparse + float32
phi (isolates float64), dense float32 (the old numerics, CPU). One profiled step lists the top ops.
"""
import argparse, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from run_leafletfa_baseline import read_cells, read_counts, read_var, peak_gb  # noqa: E402

D = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/20260526/"
STEM = "_ONLY_model_ready_combined_ge_splicing_aligned_filtered_minCells75_minJuncs500_2026-06-17.h5mu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--datasets", default="TMS,AB")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arms", default="sparse_f64_t16,sparse_f32_t16,sparse_f64_t8,dense_f32_t16")
    a = ap.parse_args()
    import torch, pyro
    from leafletfa.minibatch import LeafletFAMinibatch
    res = {}
    for ds in a.datasets.split(","):
        path = D + "train_70_30_" + ds + STEM
        _, ev = read_var(path)
        n = len(read_cells(path))
        rows = np.sort(np.random.default_rng(0).choice(n, a.batch, replace=False))
        Y = read_counts(path, rows)
        res[ds] = dict(n_train=n, nnz=int(Y.nnz), arms={})
        arms = [("sparse_f64_t16", 16, "float64", True), ("sparse_f32_t16", 16, "float32", True),
                ("sparse_f64_t8", 8, "float64", True), ("dense_f32_t16", 16, "float32", False)]
        arms = [x for x in arms if x[0] in a.arms.split(",")]
        for name, th, dt, sparse in arms:
            torch.set_num_threads(th)
            pyro.clear_param_store()
            if sparse:
                m = LeafletFAMinibatch.dm(Y, None, K=20, event_id=ev, conc_min=2.0, device=a.device,
                                          sparse=True, phi_dtype=dt)
                res[ds]["entries"] = int(m.sobs.n_entries)
            else:
                Dn = Y.toarray()
                _, inv = np.unique(ev, return_inverse=True)
                import scipy.sparse as sp
                clus = np.asarray((Y @ sp.csr_matrix((np.ones(len(ev)), (np.arange(len(ev)), inv))))[:, inv].todense(), np.float32)
                m = LeafletFAMinibatch.dm(Dn, clus, K=20, event_id=ev, conc_min=2.0, device=a.device)
                del Dn, clus
            m.fixed_dir_conc = torch.tensor(20.0, device=m.device)
            steps = 3 if not sparse else a.steps
            t = []
            orig = m.model
            m.fit(num_steps=2, subsample_size=a.batch, lr=0.025, gamma=0.1, early_stop=False, verbose=False)
            if a.device == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            m.fit(num_steps=steps, subsample_size=a.batch, lr=0.025, gamma=0.1, early_stop=False, verbose=False)
            sps = (time.time() - t0) / steps
            res[ds]["arms"][name] = dict(threads=th, phi_dtype=dt, sparse=sparse, s_per_step=round(sps, 3),
                                         peak_rss_gb=round(peak_gb(), 2),
                                         peak_gpu_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                                      if a.device == "cuda" else None))
            print(ds, name, f"{sps:.2f} s/step", flush=True)
            if name == "sparse_f64_t16":
                from torch.profiler import profile, ProfilerActivity
                acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if a.device == "cuda" else [])
                with profile(activities=acts) as prof:
                    m.fit(num_steps=1, subsample_size=a.batch, lr=0.025, gamma=0.1, early_stop=False, verbose=False)
                tab = prof.key_averages().table(sort_by=("self_cuda_time_total" if a.device == "cuda" else "self_cpu_time_total"), row_limit=15)
                res[ds]["profile_top"] = tab
                print(tab, flush=True)
            del m
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
