"""DEFINITIVE convergence study: recovery(K, epochs, batch_size) with epochs & batch varied INDEPENDENTLY.

The batch sweep at fixed steps confounds batch and epochs (smaller B at fixed steps = fewer epochs).
Here we fix EPOCHS and vary B, so steps = round(epochs * N / B). If recovery-vs-epochs curves for
different B overlap, epochs is the governing variable and B only affects speed. If small-B curves sit
lower, minibatch gradient noise itself costs recovery. Faceted by K to see how difficulty shifts the curve.

Same sim per seed across all (B, epochs) for comparability. Output grid_convergence_K<K>.csv.
Run: grid_convergence.py <K> <seeds> <device>"""
import sys
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate
from minibatch_vs_fullbatch import run
import pandas as pd

K = int(sys.argv[1]) if len(sys.argv)>1 else 20
seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1"])]
device = sys.argv[3] if len(sys.argv)>3 else "cpu"
N=40000; J=max(400,100*K)
B_list = [2048, 4096, 8192, N]           # N = full-batch
epochs_list = [150, 300, 600, 1200, 2400]
rows=[]
for seed in seeds:
    sim = simulate(N=N,J=J,K=K,seed=seed,overdisp_conc=10.0)
    print(f"\n===== K={K} seed {seed}: true psi={sim['psi'].mean():.4f} N={N} J={J} =====",flush=True)
    for B in B_list:
        for ep in epochs_list:
            steps = max(1, round(ep * N / B))
            r = run(sim, K, B, steps, seed, device, likelihood="betabinomial")
            r.update(dict(K=K, batch=("full" if B>=N else B), epochs=ep, steps=steps, seed=seed))
            rows.append(r)
            print(f"  B={str(r['batch']):>5}  epochs={ep:>5}  steps={steps:>6}  "
                  f"r={r['recovery_r']:.3f}  slope={r['beta_slope']:.3f}  "
                  f"perp={r['perplexity']:.1f}/{K}  {r['seconds']}s", flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/grid_convergence_K{K}.csv"
df.to_csv(out,index=False)
print(f"\n===== K={K} recovery_r by (batch,epochs), mean over seeds =====",flush=True)
piv=df.pivot_table(index="batch",columns="epochs",values="recovery_r",aggfunc="mean")
print(piv.to_string(),flush=True)
print(f"\n===== K={K} beta_slope by (batch,epochs) =====",flush=True)
print(df.pivot_table(index="batch",columns="epochs",values="beta_slope",aggfunc="mean").to_string(),flush=True)
print(f"wrote {out}",flush=True)
