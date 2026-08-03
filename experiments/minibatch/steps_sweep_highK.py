"""Does the high-K minibatch gap close with more steps (epochs)?
Full-batch 3000 steps = 3000 epochs. Minibatch B=8192 on N=40k at 3000 steps = only ~600 epochs.
Matched-epoch => steps = 3000 * N/B ~ 14650. Sweep B=8192 at 3000/6000/14650 vs full-batch, K in {20,50}.
Run: steps_sweep_highK.py <K> <seeds> <device>"""
import sys
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA")
sys.path.insert(0,"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch")
from compare_variants import simulate
from minibatch_vs_fullbatch import run
import pandas as pd

K = int(sys.argv[1]) if len(sys.argv)>1 else 20
seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv)>2 else ["0","1","2"])]
device = sys.argv[3] if len(sys.argv)>3 else "cpu"
N=40000; B=8192; J=max(400,100*K)
matched = round(3000*N/B)   # ~14650
arms = [("full",N,3000),("mb_3000",B,3000),("mb_6000",B,6000),(f"mb_{matched}",B,matched)]
rows=[]
for seed in seeds:
    sim = simulate(N=N,J=J,K=K,seed=seed,overdisp_conc=10.0)
    print(f"\n===== K={K} seed {seed}: true psi={sim['psi'].mean():.4f} N={N} J={J} matched_steps={matched} =====",flush=True)
    for name,b,steps in arms:
        r = run(sim,K,b,steps,seed,device,likelihood="betabinomial")
        r["arm"]=name; r["steps"]=steps; r["seed"]=seed; r["epochs"]=round(steps*b/N,1)
        rows.append(r)
        print(f"  {name:>10}  steps={steps:>6}  epochs~{r['epochs']:>6}  r={r['recovery_r']:.3f}  "
              f"slope={r['beta_slope']:.3f}  perp={r['perplexity']:.1f}/{K}  {r['seconds']}s",flush=True)
df=pd.DataFrame(rows)
out=f"/gpfs/commons/home/kisaev/LeafletFA/experiments/minibatch/steps_sweep_K{K}.csv"
df.to_csv(out,index=False)
print(f"\n===== K={K} mean over seeds =====",flush=True)
print(df.groupby("arm")[["recovery_r","beta_slope","epochs","seconds"]].mean().to_string(),flush=True)
print(f"wrote {out}",flush=True)
