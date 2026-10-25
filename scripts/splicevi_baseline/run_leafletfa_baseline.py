"""LeafletFA as a baseline for the SpliceVI paper: fit on training cells, infer factors for new cells.

    python run_leafletfa_baseline.py \
        --train  <train_70_30_..._ONLY_...h5mu> \
        --project real=<test_30_70_...h5mu> \
        --project contentonly=<SPLICINGCONTENTONLY_..._test_30_70_...h5mu> \
        --project popavg=<DS_TEST_popavg.npz> --project planted_d0.1=<DS_TEST_planted_d0.1.npz> \
        --npz-cells <DS_TEST_obs.npz> --npz-var <DS_var.tsv> \
        --K 20 --seed 0 --out <runs>/<dataset>/<run_name>

Fit = the shipped LeafletFA recipe (Dirichlet-multinomial over ATSEs, within-ATSE simplex psi,
per-ATSE concentration floored at 2, Dirichlet concentration on phi pinned at K, 16,200 steps x
8,192 cells, lr 0.025 decaying to 0.1x, no early stopping). Projection = the frozen-global E-step
used for every held-out LeafletFA number to date (psi, pi, delta = K and the per-ATSE
concentration frozen; phi inferred; 3,000 steps, batch 8,192, lr 0.1, gamma 0.1).

Two changes from the code that produced the published fits, both numerical, neither changes the
model:
  --phi-dtype float64  the phi site in float64. Fixes the non-finite E-step restarts (float32
                       stick-breaking underflow under the corner-seeking Dirichlet prior; see
                       LeafletFAMinibatch.__init__).
  --sparse             the DM evaluated on observed events only; equal to the dense likelihood up to
                       summation order (tests/test_sparse_dm.py). ~15-40x less work on CPU and no
                       dense (cells x junctions) matrices.
Pass --phi-dtype float32 --dense to reproduce the old numerics exactly.

--reference-fit <fit.npz> skips fitting and projects against an existing fit (e.g. the TMS70 K20
fits, which were trained on exactly SpliceVI's TMS TRAIN cells); its junction order is checked
against the training file's.

Inputs: .h5mu / .h5ad (splicing layer cell_by_junction_matrix; var.event_id) or .npz (scipy sparse
junction counts, rows ordered as --npz-cells, columns as the training file; --npz-var checks the
column order). Only junction counts are read: the DM derives each event total from them.

Outputs (never overwritten; an existing output is skipped, so a killed run can be resumed):
  fit/fit.npz                  globals (psi, pi, per-ATSE concentration, delta), phi of the
                               training cells, junction/event/cell ids, loss trace
  factors_<name>.tsv.gz        cells x K, first column cell_id
  factors_<name>.npz           phi (float32), cell_id, steps, restarts
  run_record.json              settings, versions, git commit, host, md5 of every input, timings,
                               peak memory, non-finite restarts per stage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import scipy.sparse as sp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

JUNC = "cell_by_junction_matrix"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── input ─────────────────────────────────────────────────────────────────────────────────────
def _splicing_root(f):
    return f["mod/splicing"] if "mod" in f and "splicing" in f["mod"] else f


def _strings(ds):
    import h5py
    if isinstance(ds, h5py.Group):                                   # categorical
        cats = _strings(ds["categories"])
        return cats[ds["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


def read_var(path):
    """(junction_id, event_id) of an h5mu / h5ad, or of a var.tsv written next to an npz."""
    if path.endswith(".tsv"):
        import pandas as pd
        v = pd.read_csv(path, sep="\t", dtype=str)        # first column = the var index (junction id)
        return v.iloc[:, 0].values, v["event_id"].values
    import h5py
    with h5py.File(path, "r") as f:
        var = _splicing_root(f)["var"]
        return _strings(var["junction_id"]), _strings(var["event_id"])


def read_cells(path):
    import h5py
    if path.endswith(".npz"):
        return np.asarray(np.load(path, allow_pickle=True)["cells"]).astype(str)
    if path.endswith(".tsv"):
        import pandas as pd
        return pd.read_csv(path, sep="\t", dtype=str).iloc[:, 0].values
    with h5py.File(path, "r") as f:
        obs = _splicing_root(f)["obs"]
        return _strings(obs[obs.attrs["_index"]])


def read_counts(path, rows=None, npz_cells=None):
    """Junction counts as float32 CSR, optionally only `rows` (sorted int array)."""
    if path.endswith(".npz"):
        Y = sp.load_npz(path).tocsr().astype(np.float32)
        return Y if rows is None else Y[rows]
    import h5py
    with h5py.File(path, "r") as f:
        g = _splicing_root(f)["layers"][JUNC]
        if g.attrs.get("encoding-type", "csr_matrix") != "csr_matrix":
            raise SystemExit(f"{path}: {JUNC} is {g.attrs.get('encoding-type')}, expected csr_matrix")
        shape = tuple(int(x) for x in g.attrs["shape"])
        ip = g["indptr"][:]
        if rows is None:
            return sp.csr_matrix((g["data"][:].astype(np.float32), g["indices"][:], ip), shape=shape)
        data, ind, ptr = [], [], [0]
        for r in rows:                                               # row slices, no full read
            a, b = int(ip[r]), int(ip[r + 1])
            data.append(g["data"][a:b].astype(np.float32)); ind.append(g["indices"][a:b])
            ptr.append(ptr[-1] + b - a)
        return sp.csr_matrix((np.concatenate(data), np.concatenate(ind), np.array(ptr)),
                             shape=(len(rows), shape[1]))


def md5(path, out, key):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    out[key] = dict(path=os.path.abspath(path), bytes=os.path.getsize(path),
                    mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
                    md5=h.hexdigest())


def git_state(repo):
    def run(*a):
        return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


def gpu_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # ru_maxrss is KB on Linux


# ── main ──────────────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="training file (.h5mu/.h5ad); its var fixes the junction order")
    ap.add_argument("--project", action="append", default=[], metavar="NAME=PATH",
                    help="file to infer factors for; repeatable")
    ap.add_argument("--npz-cells", help="cell ids, in row order, for every .npz input (npz with 'cells', tsv, or h5mu)")
    ap.add_argument("--npz-var", help="var.tsv / h5mu giving the column order of the .npz inputs (checked)")
    ap.add_argument("--reference-fit", help="existing fit.npz: skip fitting, project against it")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-steps", type=int, default=16200)
    ap.add_argument("--fit-batch", type=int, default=8192)
    ap.add_argument("--fit-lr", type=float, default=0.025)
    ap.add_argument("--estep-steps", type=int, default=3000)
    ap.add_argument("--estep-batch", type=int, default=8192)
    ap.add_argument("--estep-lr", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.1, help="lr decays to gamma x lr over the run")
    ap.add_argument("--conc-min", type=float, default=2.0)
    ap.add_argument("--max-nan-restarts", type=int, default=40)
    ap.add_argument("--checkpoint-every", type=int, default=500, help="fit snapshot interval (steps)")
    ap.add_argument("--phi-dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--dense", action="store_true", help="dense likelihood (the pre-2026-10 numerics)")
    ap.add_argument("--n-train", type=int, default=0, help="random subset of training cells (0 = all)")
    ap.add_argument("--n-project", type=int, default=0, help="random subset of projected cells, the "
                    "same cell ids in every input (0 = all)")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or None)
    ap.add_argument("--no-md5", action="store_true", help="skip input checksums (smoke tests only)")
    a = ap.parse_args(argv)

    import torch
    import pyro
    import leafletfa
    from leafletfa.minibatch import LeafletFAMinibatch
    if a.threads:
        torch.set_num_threads(a.threads)
    t_start = time.time()
    os.makedirs(os.path.join(a.out, "fit"), exist_ok=True)
    rec_path = os.path.join(a.out, "run_record.json")
    projects = []
    for p in a.project:
        name, path = p.split("=", 1)
        projects.append((name, path))
    if len({n for n, _ in projects}) != len(projects):
        raise SystemExit("duplicate --project names")

    done = (os.path.exists(os.path.join(a.out, "fit", "fit.npz")) or a.reference_fit) and all(
        os.path.exists(os.path.join(a.out, f"factors_{n}.npz")) for n, _ in projects)
    if done and os.path.exists(rec_path):
        log(f"nothing to do: fit and all {len(projects)} projections exist in {a.out}")
        return
    record = dict(argv=sys.argv, settings=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                  host=platform.node(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                  versions=dict(python=platform.python_version(), torch=torch.__version__,
                                pyro=pyro.__version__, numpy=np.__version__,
                                scipy=__import__("scipy").__version__,
                                leafletfa=leafletfa.__version__),
                  leafletfa_git=git_state(REPO), torch_threads=torch.get_num_threads(),
                  gpu=(torch.cuda.get_device_name(0) if a.device == "cuda" and torch.cuda.is_available() else None),
                  stages={}, inputs_md5={})
    if os.path.exists(rec_path):                                     # resumed run: keep history
        with open(rec_path) as fh:
            record["previous_runs"] = [json.load(fh)]

    # checksums run beside the compute; joined before the record is final
    md5_threads = []
    prev_md5 = record.get("previous_runs", [{}])[0].get("inputs_md5", {})
    if not a.no_md5:
        files = [("train", a.train)] + projects
        files += [(k, v) for k, v in (("npz_cells", a.npz_cells), ("npz_var", a.npz_var),
                                      ("reference_fit", a.reference_fit)) if v]
        for key, path in files:
            p0 = prev_md5.get(key, {})
            if (p0.get("path") == os.path.abspath(path) and p0.get("bytes") == os.path.getsize(path)
                    and p0.get("mtime") == time.strftime("%Y-%m-%d %H:%M:%S",
                                                         time.localtime(os.path.getmtime(path)))):
                record["inputs_md5"][key] = p0                       # unchanged file: reuse
                continue
            t = threading.Thread(target=md5, args=(path, record["inputs_md5"], key), daemon=True)
            t.start(); md5_threads.append(t)

    def write_record():
        with open(rec_path + ".tmp", "w") as fh:
            json.dump(record, fh, indent=2, default=str)
        os.replace(rec_path + ".tmp", rec_path)

    junc_id, ev_id = read_var(a.train)
    J = len(junc_id)
    log(f"train var: {J:,} junctions, {len(np.unique(ev_id)):,} ATSEs")
    sparse = not a.dense

    def model_for(Y):
        pyro.clear_param_store()
        if sparse:
            return LeafletFAMinibatch.dm(Y, None, K=a.K, event_id=ev_id, conc_min=a.conc_min,
                                        device=a.device, sparse=True, phi_dtype=a.phi_dtype)
        D = Y.toarray()
        _, inv = np.unique(ev_id, return_inverse=True)
        tot = sp.csr_matrix(Y) @ sp.csr_matrix((np.ones(J), (np.arange(J), inv)))
        clus = np.asarray(tot[:, inv].todense(), np.float32)
        return LeafletFAMinibatch.dm(D, clus, K=a.K, event_id=ev_id, conc_min=a.conc_min,
                                    device=a.device, phi_dtype=a.phi_dtype)

    # ── fit (or load) ──
    fit_path = os.path.join(a.out, "fit", "fit.npz")
    if a.reference_fit:
        z = np.load(a.reference_fit, allow_pickle=True)
        fit_ev = np.asarray(z["event_id"]).astype(str)[np.asarray(z["inv_ev"])]
        if len(fit_ev) != J or not np.array_equal(fit_ev, ev_id):
            raise SystemExit("reference fit's junction -> ATSE map differs from the training file's")
        if "junction_id" in z.files and not np.array_equal(np.asarray(z["junction_id"]).astype(str), junc_id):
            raise SystemExit("reference fit's junction_id order differs from the training file's")
        if int(z["K"]) != a.K:
            raise SystemExit(f"reference fit has K={int(z['K'])}, --K {a.K}")
        pi = np.asarray(z["pi"] if "pi" in z.files else z["trace_pi"][-1], np.float32)
        G = dict(psi=np.asarray(z["psi_mean"], np.float32), pi=pi / pi.sum(),
                 conc=np.asarray(z["bb_conc_vec"], np.float32), dir_conc=float(a.K))
        record["stages"]["fit"] = dict(source="reference", path=os.path.abspath(a.reference_fit),
                                       note="junction -> ATSE map verified equal to --train var")
        log(f"reference fit {a.reference_fit}")
    elif os.path.exists(fit_path):
        z = np.load(fit_path, allow_pickle=True)
        G = dict(psi=z["psi_mean"], pi=z["pi"], conc=z["bb_conc_vec"], dir_conc=float(z["dir_conc"]))
        log(f"fit exists, reusing {fit_path} (not refitted)")
        record["stages"]["fit"] = dict(source="existing", path=fit_path)
    else:
        cells_tr = read_cells(a.train)
        rows = None
        if a.n_train and a.n_train < len(cells_tr):
            rows = np.sort(np.random.default_rng(a.subset_seed).choice(len(cells_tr), a.n_train, replace=False))
        t0 = time.time()
        Y = read_counts(a.train, rows)
        cells_tr = cells_tr if rows is None else cells_tr[rows]
        empty = int((np.diff(Y.indptr) == 0).sum())
        log(f"train counts {Y.shape[0]:,} x {Y.shape[1]:,}, nnz {Y.nnz:,}, {empty} cells with no reads "
            f"({time.time() - t0:.0f}s)")
        m = model_for(Y)
        del Y
        m.fixed_dir_conc = torch.tensor(float(a.K), device=m.device)  # delta = K, the shipped pin
        # CHECKPOINT / RESUME. A full-size CPU fit can outlive a job's wall clock, so the guide state
        # is snapshotted every --checkpoint-every steps. A rerun with the same arguments resumes at
        # the saved step with the remaining budget, the lr decay continuing where it stopped.
        ckpt_path = os.path.join(a.out, "fit", "checkpoint.pt")
        s0, prev = 0, dict(losses=[], nan_steps=[], n_nan_restarts=0, segments=[])
        resume = None
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, weights_only=False)
            s0, resume, prev = int(ck["step"]) + 1, ck["state"], ck["prev"]
            log(f"resuming fit at step {s0:,} from {ckpt_path}")
        lrd = a.gamma ** (1.0 / a.fit_steps)
        remaining = a.fit_steps - s0

        def on_track(step, _psi, _phi, _tr):
            if step == 0:
                return
            ck = dict(step=s0 + step, state=m._param_snapshot(),
                      prev=dict(losses=prev["losses"] + list(map(float, m.losses)),
                                nan_steps=prev["nan_steps"] + [s0 + x for x in m.nan_steps],
                                n_nan_restarts=prev["n_nan_restarts"] + m.n_nan_restarts,
                                segments=prev["segments"]))
            torch.save(ck, ckpt_path + ".tmp"); os.replace(ckpt_path + ".tmp", ckpt_path)
            log(f"  [checkpoint] step {s0 + step:,} ({(time.time() - t0) / max(step, 1):.2f} s/step, "
                f"peak RSS {peak_gb():.1f} GB)")
        t0 = time.time()
        m.fit(num_steps=remaining, subsample_size=a.fit_batch, lr=a.fit_lr * lrd ** s0,
              gamma=a.gamma ** (remaining / a.fit_steps), seed=a.seed + s0, early_stop=False,
              max_nan_restarts=a.max_nan_restarts, checkpoint_every=200, log_every=200, verbose=True,
              track_every=a.checkpoint_every, on_track=on_track, resume_state=resume)
        m.losses = prev["losses"] + list(m.losses)
        m.nan_steps = prev["nan_steps"] + [s0 + x for x in m.nan_steps]
        m.n_nan_restarts += prev["n_nan_restarts"]
        m.stopped_step = s0 + m.stopped_step
        g = m.globals_()
        G = dict(psi=m.psi_mean().numpy().astype(np.float32),
                 pi=np.asarray(g["pi"], np.float32) / np.asarray(g["pi"]).sum(),
                 conc=m.concentration().astype(np.float32), dir_conc=float(a.K))
        phi_tr = m.phi_mean().numpy()
        ev_u, inv = np.unique(ev_id, return_inverse=True)
        np.savez_compressed(fit_path + ".tmp.npz", psi_mean=G["psi"], pi=G["pi"], bb_conc_vec=G["conc"],
                            dir_conc=G["dir_conc"], phi=phi_tr.astype(np.float32), cell_id=cells_tr,
                            junction_id=junc_id, event_id=ev_u, inv_ev=inv.astype(np.int32),
                            K=a.K, seed=a.seed, steps_run=m.stopped_step,
                            losses=np.asarray(m.losses, np.float32), nan_steps=np.asarray(m.nan_steps))
        os.replace(fit_path + ".tmp.npz", fit_path)
        record["stages"]["fit"] = dict(source="fitted", path=fit_path, n_cells=int(len(cells_tr)),
                                       empty_cells=empty, steps_run=int(m.stopped_step),
                                       n_nan_restarts=int(m.n_nan_restarts), nan_steps=list(map(int, m.nan_steps)),
                                       finite_phi=bool(np.isfinite(phi_tr).all()),
                                       minutes=round((time.time() - t0) / 60, 2),
                                       resumed_from_step=s0,
                                       s_per_step=round((time.time() - t0) / max(1, remaining), 3),
                                       peak_rss_gb=round(peak_gb(), 2), peak_gpu_gb=gpu_gb())
        log(f"fit done: {m.n_nan_restarts} non-finite restarts, {(time.time() - t0) / 60:.1f} min")
        del m
        write_record()

    # ── project ──
    npz_cells = read_cells(a.npz_cells) if a.npz_cells else None
    if a.npz_var:
        jv, ev_v = read_var(a.npz_var)
        if not (np.array_equal(jv, junc_id) and np.array_equal(ev_v, ev_id)):
            raise SystemExit("--npz-var junction order differs from the training file's")
    chosen = None                                                     # same cell ids in every input
    if a.n_project and projects:
        sel_path = os.path.join(a.out, "projected_cells.txt")
        if os.path.exists(sel_path):                                  # resumed run: same cells
            chosen = np.loadtxt(sel_path, dtype=str, ndmin=1)
        else:
            p0 = projects[0][1]
            c0 = npz_cells if p0.endswith(".npz") else read_cells(p0)
            chosen = np.sort(np.random.default_rng(a.subset_seed + 1).choice(c0, a.n_project, replace=False))
            np.savetxt(sel_path, chosen, fmt="%s")
    for name, path in projects:
        out_tsv = os.path.join(a.out, f"factors_{name}.tsv.gz")
        out_npz = os.path.join(a.out, f"factors_{name}.npz")
        if os.path.exists(out_npz):
            log(f"{name}: {out_npz} exists, skipped (never overwritten)")
            continue
        if path.endswith(".npz"):
            if npz_cells is None:
                raise SystemExit(f"{name}: .npz input needs --npz-cells")
            cells = npz_cells
            n_rows = sp.load_npz(path).shape[0] if a.n_project else None
            if n_rows is not None and n_rows != len(cells):
                raise SystemExit(f"{name}: {n_rows} rows but --npz-cells has {len(cells)}")
        else:
            cells = read_cells(path)
            jv, ev_v = read_var(path)
            if not (np.array_equal(jv, junc_id) and np.array_equal(ev_v, ev_id)):
                raise SystemExit(f"{name}: junction order differs from the training file's")
        if chosen is not None:
            pos = {c: i for i, c in enumerate(cells)}
            missing = [c for c in chosen if c not in pos]
            if missing:
                raise SystemExit(f"{name}: {len(missing)} chosen cells missing")
            rows = np.sort(np.array([pos[c] for c in chosen]))
        else:
            rows = None
        t0 = time.time()
        Y = read_counts(path, rows)
        cells_p = cells if rows is None else cells[rows]
        if Y.shape[0] != len(cells_p):
            raise SystemExit(f"{name}: {Y.shape[0]} rows but {len(cells_p)} cell ids")
        if Y.shape[1] != J:
            raise SystemExit(f"{name}: {Y.shape[1]} columns, training has {J}")
        empty = int((np.diff(Y.indptr) == 0).sum())
        m = model_for(Y)
        del Y
        m.freeze_globals(psi=G["psi"], pi=G["pi"], alpha_pi=1.0, dir_conc=G["dir_conc"], bb_conc=G["conc"])
        t1 = time.time()
        phi = m.infer_phi(num_steps=a.estep_steps, subsample_size=a.estep_batch, lr=a.estep_lr,
                          gamma=a.gamma, seed=a.seed, early_stop=False,
                          max_nan_restarts=a.max_nan_restarts, verbose=True, log_every=500)
        phi = np.asarray(phi, np.float32)
        import pandas as pd
        df = pd.DataFrame(phi, columns=[f"LFA_{k + 1}" for k in range(a.K)])
        df.insert(0, "cell_id", cells_p)
        df.to_csv(out_tsv + ".tmp", sep="\t", index=False, float_format="%.6g", compression="gzip")
        os.replace(out_tsv + ".tmp", out_tsv)
        np.savez_compressed(out_npz + ".tmp.npz", phi=phi, cell_id=cells_p,
                            steps=m.stopped_step, n_nan_restarts=m.n_nan_restarts)
        os.replace(out_npz + ".tmp.npz", out_npz)
        record["stages"][f"project_{name}"] = dict(
            path=os.path.abspath(path), n_cells=int(len(cells_p)), empty_cells=empty,
            steps_run=int(m.stopped_step), n_nan_restarts=int(m.n_nan_restarts),
            nan_steps=list(map(int, m.nan_steps)), finite=bool(np.isfinite(phi).all()),
            minutes=round((time.time() - t0) / 60, 2),
            s_per_step=round((time.time() - t1) / max(1, m.stopped_step), 3),
            peak_rss_gb=round(peak_gb(), 2), peak_gpu_gb=gpu_gb(), output=out_tsv)
        log(f"{name}: {len(cells_p):,} cells, {m.n_nan_restarts} non-finite restarts, "
            f"{(time.time() - t0) / 60:.1f} min -> {out_tsv}")
        del m
        write_record()

    for t in md5_threads:
        t.join()
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record["total_minutes"] = round((time.time() - t_start) / 60, 2)
    record["peak_rss_gb"] = round(peak_gb(), 2)
    write_record()
    log(f"done in {record['total_minutes']} min, peak RSS {record['peak_rss_gb']} GB -> {rec_path}")


if __name__ == "__main__":
    main()
