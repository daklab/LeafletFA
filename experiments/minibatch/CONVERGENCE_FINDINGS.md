# LeafletFA convergence study — K × epochs × batch size × likelihood × init

One overdispersed simulation (N=40,000 cells, J=100·K junctions, coverage 13.5%, mean 12 reads,
Beta(0.30,0.72) U-shaped ψ, Beta-Binomial generative with `overdisp_conc=10`, known ground truth).
All arms use the corrected Pyro-plate minibatch trainer (`leafletfa/minibatch.py`, subsampling verified
engaged). Recovery = Hungarian-matched corr(ψ̂, ψ_true); β-slope = regression of fitted DS effect size on
truth (1.0 = no attenuation); perplexity = median effective #programs/cell (low = committed representation).

## Headline: convergence has THREE regimes, not one

| Regime | Behavior | Governing factor | Fix |
|---|---|---|---|
| **K=10** | minibatch = full-batch everywhere | nothing — easy | any batch works |
| **K=20** | persistent minibatch gap that **more epochs cannot close** | **batch size** (gradient noise) | larger batch / full-batch |
| **K=50** | Beta-Binomial **fails to commit**; Binomial recovers | **likelihood + init** | fix bb_conc / structured init / Binomial |

The naive expectation ("minibatch just needs enough epochs to match full") is **false** at K≥20.

## Evidence

### K=10 — solved
Minibatch matches full-batch at every batch size and step count tested. r≈0.99, slope≈1.0 across
full/8192/4096/2048 at 3000 steps, all seeds. Both likelihoods fine.

### K=20 — the gap is batch NOISE, not epoch deficit  [steps sweep, 3 seeds, CONFIRMED]
| arm | epochs | recovery r | β-slope |
|---|---|---|---|
| full-batch | 3000 | **0.996** | **1.000** |
| B=8192 | 614 | 0.898 | 0.894 |
| B=8192 | 1229 | 0.905 | 0.876 |
| B=8192 | 3000 (matched to full) | **0.898** | **0.818** |

At *identical* epochs, B=8192 plateaus at r≈0.90 while full hits 0.996; recovery is **flat across 5× the
epochs** (0.898→0.905→0.898) and the **β-slope monotonically worsens** (0.894→0.876→0.818) — minibatch
converges to a *stable but attenuated* solution and more steps just sharpen toward it. So the gap is genuine
minibatch gradient noise → **batch size is the knob at K≥20, not epochs**, and B=8192 attenuates the DS
effect size ~18%. [grid will show whether even larger batches recover K=20]

**Directly relevant to the paper:** the atlas production recipe is B=8192; this is the effect-size
attenuation the review plan §2 flagged, now confirmed on overdispersed data with correct plate minibatching.

### K=50 — Beta-Binomial COLLAPSES, Binomial works  [full-batch, seed 0]
| likelihood | recovery r | β-slope | perplexity |
|---|---|---|---|
| **Beta-Binomial** | **0.142** | 0.001 | 46.9/50 (uncommitted, ≈uniform) |
| **Binomial** | **0.941** | 0.918 | 15.7/50 (committed) |

The *correct* model (BB, matching the overdispersed generator) fails while the *misspecified* Binomial
recovers. Reproducible/deterministic per seed (r=0.142 identical across simOD and steps-sweep). At minibatch
B=8192, Binomial still gives r=0.727 vs BB r=0.083. Random-init BB at K=50 fails at every batch/step tested.

**Hypothesis:** learnable `bb_conc` runs away to high overdispersion, flattening the likelihood so no
gradient pushes factors apart (perplexity ≈ uniform).

**Init experiment — structured init RESCUES K=50 BB [full-batch, seed 0; seed 1 pending]:**
| init | recovery r | β-slope | perplexity |
|---|---|---|---|
| random | 0.142 | 0.001 | 46.9/50 (collapsed) |
| **k-means (waypoint proxy)** | **0.930** | **0.828** | **19.3/50 (committed)** |

Random init lands in the collapse basin; k-means init (distinct data-supported anchors) avoids it entirely.
**The K=50 failure is an initialization problem** — this is the plan §9 "init is load-bearing at high K",
demonstrated. The atlas fix: **K=50 must use waypoint init** (random-init K=50 sweep models are garbage).

- **mechanism** (fix bb_conc at 10/20/100 vs learned): does constraining dispersion *alone* also rescue BB,
  and did the learned bb_conc collapse? [running — pins the exact mechanism + a second independent fix]

## Implications for the atlas retrain (config = K∈{10,20,50}, waypoints∈{T,F}, B=8192, ~264 epochs)
1. **B=8192 attenuates effect sizes at K≥20** — production models likely report compressed DS β. Consider
   larger batch (or full-batch on a GPU-fitting subset) at high K.
2. **K=50 random-init BB models are at risk of collapse** — half the sweep (waypoints=False) uses random
   init. Waypoint init and/or bb_conc control may be mandatory at K=50.
3. **3000 steps ≈ 264 epochs on the full atlas** — far below where even K=20 stabilizes in the sim.
4. Retrain was cancelled pending these results — correct call; the recipe is not yet settled.

## Caveats
- Sim uses **random init**; the atlas uses **waypoints** (init experiment tests whether that closes the K=50
  gap). Sim J (1000-5000) << atlas J (79,768) — more junctions/program may improve identifiability at scale.
- Most single-seed results above will be seed-averaged once grid/init/mechanism finish (2 seeds each).

_Status: simOD K=10/20 done, K=50 finishing; steps-sweep finishing; grid + init running; mechanism queued._
