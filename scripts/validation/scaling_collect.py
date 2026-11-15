"""Scaling and runtime record (VALIDATION_PREREG, 'Record, scaling and runtime for the supplement', 2026-11-14).
Collects the controlled runs into one table and one plot, as measured.
    python scaling_collect.py   -> $V/scaling/scaling.tsv, $V/scaling/scaling.{pdf,png}
LeafletFA s/step = median over the 500-step checkpoint intervals after step 500, from log timestamps
(intervals 500-1000, 1000-1500, 1500-1999; checkpoint writes included). scQuint epochs = last tqdm epoch count."""
import glob, json, os, re, subprocess, sys
from datetime import datetime
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402
sys.path.insert(0, "/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/figures/_common")
from leaflet_theme import apply_theme, PALETTE, figsize, panel_label, save  # noqa: E402

S = os.path.join(vlib.V, "scaling"); L = os.path.join(vlib.V, "logs"); FULL = 54694


def sacct(job):
    o = subprocess.run(["sacct", "-j", str(job), "-n", "-P", "-o", "JobID,Elapsed,MaxRSS,AllocCPUS,NodeList"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    top = o[0].split("|"); h, m, s = map(int, top[1].split(":")[-3:]); dd = top[1].split("-")
    el = (int(dd[0]) * 86400 if len(dd) > 1 else 0) + h * 3600 + m * 60 + s
    u = {"K": 1e-6, "M": 1e-3, "G": 1.0}
    rss = max((float(v[:-1]) * u[v[-1]] if v[-1] in u else float(v) / 1e9
               for v in (x.split("|")[2] for x in o) if v), default=np.nan)
    return el / 60, rss, int(top[3]), top[4]


def step_rate(log):
    t = {int(m.group(2).replace(",", "")): datetime.strptime(m.group(1), "%H:%M:%S")
         for m in re.finditer(r"\[(\d\d:\d\d:\d\d)\]\s+\[checkpoint\] step ([\d,]+)", open(log).read())}
    k = sorted(x for x in t if x >= 500)
    return float(np.median([(t[b] - t[a]).total_seconds() % 86400 / (b - a) for a, b in zip(k, k[1:])]))


rows = []
for tag in ["gpu_5000", "gpu_10000", "gpu_25000", "gpu_50000", "gpu_full", "cpu_5000"]:
    r = json.load(open(os.path.join(S, f"lfa_{tag}", "run_record.json"))); f = r["stages"]["fit"]
    job = r["slurm_job"]; log = glob.glob(os.path.join(L, f"lfa_scale_{tag}_{job}.log"))[0]; el, rss, cpu, node = sacct(job)
    rows.append(dict(method="LeafletFA", device=tag.split("_")[0].upper(), cells=f["n_cells"], job=job, node=node, cpus=cpu,
                     hardware=r.get("gpu") if tag.startswith("gpu") else f"{cpu} CPUs",
                     steps=f["steps_run"], s_per_step=step_rate(log), fit_min=f["minutes"], job_min=el,
                     peak_gpu_gb=f.get("peak_gpu_gb", np.nan) if tag.startswith("gpu") else np.nan,
                     peak_host_gb=f["peak_rss_gb"], sacct_maxrss_gb=rss))
for n in (5000, 10000, 25000, 50000):
    log = glob.glob(os.path.join(L, f"lfa_scale_scq_{n}_*.log"))[0]; job = re.search(r"_(\d+)\.log$", log).group(1)
    txt = open(log).read(); tr = float(re.search(r"train_s ([\d.]+)", txt).group(1))
    ep = int(re.findall(r"\|\s*(\d+)/300", txt)[-1]); el, rss, cpu, node = sacct(job)
    rows.append(dict(method="scQuint", device="CPU", cells=n, job=job, node=node, cpus=cpu, hardware=f"{cpu} CPUs",
                     epochs=ep, s_per_epoch=tr / ep, fit_min=tr / 60, job_min=el, peak_host_gb=rss, sacct_maxrss_gb=rss))
T = pd.DataFrame(rows); T.to_csv(os.path.join(S, "scaling.tsv"), sep="\t", index=False)
g = T[(T.method == "LeafletFA") & (T.device == "GPU")]; c = T[(T.method == "LeafletFA") & (T.device == "CPU")].iloc[0]
print(T.round(3).to_string(index=False))
print(f"CPU/GPU s-per-step ratio at 5,000 cells: {c.s_per_step / g[g.cells == 5000].s_per_step.iloc[0]:.1f}")

apply_theme(); import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, 2, figsize=figsize("double", h=2.6), constrained_layout=True)
q = T[T.method == "scQuint"]
ax[0].plot(g.cells, g.fit_min, "o-", color=PALETTE["teal"], ms=3, label="LeafletFA, GPU, 2,000 steps")
ax[0].plot([c.cells], [c.fit_min], "s", color=PALETTE["teal"], mfc="white", ms=4, label="LeafletFA, CPU, 2,000 steps")
ax[0].plot(q.cells, q.fit_min, "o-", color=PALETTE["olive"], ms=3, label="scQuint, CPU, to early stop")
ax[0].set(xscale="log", yscale="log", xlabel="Cells", ylabel="Training time (min)", ylim=(1, 400))
ax[1].plot(g.cells, g.peak_gpu_gb, "o-", color=PALETTE["teal"], ms=3, label="LeafletFA, GPU memory")
ax[1].plot(g.cells, g.peak_host_gb, "o--", color=PALETTE["teal"], ms=3, label="LeafletFA, host memory (GPU run)")
ax[1].plot(q.cells, q.peak_host_gb, "o-", color=PALETTE["olive"], ms=3, label="scQuint, host memory")
ax[1].set(xscale="log", xlabel="Cells", ylabel="Peak memory (GB)", ylim=(0, None))
import matplotlib.ticker as mt  # noqa: E402
for a, l, loc in zip(ax, "ab", ("center right", "upper left")):
    a.set_xticks([5000, 10000, 25000, 54694]); a.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    a.xaxis.set_minor_locator(mt.NullLocator())
    a.legend(frameon=False, handlelength=1.2, labelspacing=0.35, borderpad=0.2, loc=loc); panel_label(a, l)
save(fig, os.path.join(S, "scaling")); print("wrote", S)
