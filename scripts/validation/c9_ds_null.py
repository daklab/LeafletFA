"""Check 9b (CPU): calibration of the shipped posterior DS test, and of LeafCutter, on a real-data null (amendment 5).

    python c9_ds_null.py run        # pick types, make splits, run LeafletFA (seed-0 fit), write LeafCutter inputs
    python c9_ds_null.py submit     # LeafCutter array (one task per type x split)
    python c9_ds_null.py collect    # LeafCutter results + summary

Cell types: the 5 TMS tissue-cell types with the most mice having >= 10 training cells (ties: more cells), >= 8 such mice.
Splits: within each mouse the type's cells are shuffled and halved (odd cell to arm A); 20 splits (seeds 0-19). Every
call is false. LeafCutter: one pseudobulk sample per (mouse, arm), leafcutter-ds as in GB 08_leafcutter_ds/05 (min 4
samples per group and per intron), no covariates.
Writes tsv/c9b_types.tsv, c9b_leafletfa.tsv, c9b_leafcutter.tsv, c9b_summary.tsv; LeafCutter inputs in c9_ds_null/.
"""
import os, subprocess, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, c9_lib as L  # noqa: E402

N_TYPES, MIN_CELLS_MOUSE, MIN_MICE, N_SPLITS, MIN_SAMPLES = 5, 10, 8, 20, 4
T = os.path.join(vlib.V, "tsv")
OUT = os.path.join(vlib.V, "c9_ds_null")
LEAFCUTTER_DS = "/gpfs/commons/home/kisaev/miniconda3/envs/LeafletSC/bin/leafcutter-ds"


def pick_types(tct, mouse):
    d = pd.DataFrame(dict(t=tct, m=mouse)).value_counts().rename("n").reset_index()
    d = d[d.n >= MIN_CELLS_MOUSE]
    g = d.groupby("t").agg(n_mice=("m", "size"), n_cells=("n", "sum")).reset_index()
    g = g[g.n_mice >= MIN_MICE].sort_values(["n_mice", "n_cells"], ascending=False).head(N_TYPES)
    return g, d


def splits(rows, mouse, seed):
    rng = np.random.default_rng(seed)
    a, b = [], []
    for m in np.unique(mouse[rows]):
        r = rng.permutation(rows[mouse[rows] == m]); h = (len(r) + 1) // 2
        a.append(r[:h]); b.append(r[h:])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def lc_name(jid, ev):
    c, s, e, st = jid.rsplit("_", 3)
    return f"{c}:{s}:{e}:{ev}_{st}"


def run():
    cells, tct, mouse, Y = L.tms_train()
    G, d = pick_types(tct, mouse)
    G.to_csv(os.path.join(T, "c9b_types.tsv"), sep="\t", index=False)
    print(G.to_string(index=False))
    post = L.load_posterior(0)
    assert np.array_equal(post["cell_id"], cells)
    inv = post["inv"].numpy(); ev = post["event_id"]
    cov = L.coverage(Y, inv, post["n_atse"]).tocsr()
    psi_s = L.draw_psi(post, seed=0)
    names = np.array([lc_name(j, ev[inv[i]]) for i, j in enumerate(post["junction_id"])])
    Yc = Y.tocsr()
    rows_out, jobs = [], []
    for t in G.t:
        mice = d[d.t == t].m.values
        rows = np.flatnonzero((tct == t) & np.isin(mouse, mice))
        for sp_ in range(N_SPLITS):
            a, b = splits(rows, mouse, sp_)
            res = L.test(post, psi_s, a, b, cov, draw_seed=sp_)
            rows_out.append(dict(cell_type=t, split=sp_, n_a=len(a), n_b=len(b), tested=len(res),
                                 calls=int(res.called.sum()), false_call_rate=res.called.mean() if len(res) else np.nan,
                                 max_prob=float(res.prob.max()) if len(res) else np.nan))
            # LeafCutter inputs: one pseudobulk per (mouse, arm)
            dd = os.path.join(OUT, f"{t.replace(' ', '_')}__split{sp_:02d}")
            os.makedirs(dd, exist_ok=True)
            samp, cols = [], []
            for arm, rr in (("armA", a), ("armB", b)):
                for m in np.unique(mouse[rr]):
                    sel = rr[mouse[rr] == m]
                    cols.append(np.asarray(Yc[sel].sum(0)).ravel()); samp.append((f"{m}_{arm}", arm))
            M = np.vstack(cols).T
            keep = M.sum(1) > 0
            df = pd.DataFrame(M[keep].astype(int), index=names[keep], columns=[s for s, _ in samp])
            df.to_csv(os.path.join(dd, "counts.tsv.gz"), sep="\t", compression="gzip", index_label="chrom")
            with open(os.path.join(dd, "groups.txt"), "w") as fh:
                fh.write("".join(f"{s}\t{g}\n" for s, g in samp))
            jobs.append(dd)
            print(f"{t} split {sp_}: LeafletFA {rows_out[-1]['calls']} calls / {rows_out[-1]['tested']} tested", flush=True)
    pd.DataFrame(rows_out).to_csv(os.path.join(T, "c9b_leafletfa.tsv"), sep="\t", index=False)
    with open(os.path.join(OUT, "array_jobs.txt"), "w") as fh:
        fh.write("\n".join(jobs) + "\n")


def submit():
    lst = os.path.join(OUT, "array_jobs.txt"); n = sum(1 for _ in open(lst))
    logs = os.path.join(vlib.V, "logs")
    wrap = (f'D=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" {lst}); '
            f'{LEAFCUTTER_DS} "$D/counts.tsv.gz" "$D/groups.txt" --output_prefix "$D/ds" --baseline_group armA '
            f'--num_threads 4 --min_samples_per_intron {MIN_SAMPLES} --min_samples_per_group {MIN_SAMPLES}')
    r = subprocess.run(["sbatch", "--parsable", "-J", "lfa_c9_lc", "-p", "cpu", "--exclude=ne1dg7-[001-011]", "-c", "4",
                        "--mem=16G", "-t", "6:00:00", f"--array=0-{n - 1}%25", f"--output={logs}/%x_%A_%a.log",
                        "--wrap", wrap], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())


def collect():
    jobs = [l.strip() for l in open(os.path.join(OUT, "array_jobs.txt")) if l.strip()]
    rows = []
    for dd in jobs:
        t, sp_ = os.path.basename(dd).rsplit("__split", 1)
        f = os.path.join(dd, "ds_cluster_significance.txt")
        if not os.path.exists(f):
            rows.append(dict(cell_type=t.replace("_", " "), split=int(sp_), tested=np.nan)); continue
        x = pd.read_csv(f, sep="\t"); x = x[x.status == "Success"]
        rows.append(dict(cell_type=t, split=int(sp_), tested=len(x), frac_p05=(x.p < 0.05).mean(),
                         calls_q05=int((x["p.adjust"] < 0.05).sum()), frac_q05=(x["p.adjust"] < 0.05).mean()))
    C = pd.DataFrame(rows); C.to_csv(os.path.join(T, "c9b_leafcutter.tsv"), sep="\t", index=False)
    Lf = pd.read_csv(os.path.join(T, "c9b_leafletfa.tsv"), sep="\t")
    Lf["key"] = Lf.cell_type.str.replace(" ", "_")
    S = Lf.groupby("key").agg(lfa_tested=("tested", "median"), lfa_false_call_rate=("false_call_rate", "mean"),
                              lfa_frac_splits_any_call=("calls", lambda c: (c > 0).mean())).join(
        C.groupby("cell_type").agg(lc_tested=("tested", "median"), lc_frac_p05=("frac_p05", "mean"),
                                   lc_frac_q05=("frac_q05", "mean"), lc_runs=("tested", "count")))
    S["lfa_pass"] = S.lfa_false_call_rate <= 0.05
    S.to_csv(os.path.join(T, "c9b_summary.tsv"), sep="\t")
    pd.set_option("display.width", 250)
    print(S.round(4).to_string())
    print("9b:", "PASS" if S.lfa_pass.all() else "FAIL", "(LeafletFA mean false-call rate <= 0.05 in every type)")


if __name__ == "__main__":
    {"run": run, "submit": submit, "collect": collect}[sys.argv[1]]()
