# Plan — mouse atlas 70% training runs

*Written 2026-07-13. Everything below is measured, not guessed; sources are cited inline.*

---

## ⚠️ STATUS 2026-07-13, 11:30 — array **18850906** (38 runs) is queued

An earlier array (18850569) was **cancelled while still PENDING** — zero GPU time wasted — because a test
run late that morning found that **the ψ prior the paper uses is a shrinkage prior toward the symmetric
saddle, and it had never been tested.** See §0 below and METHODS.md §11.4d. `junc_specific_prior` was
promoted from an ablation to a **core axis**, and the array was resubmitted.

Monitoring: SLURM job **18850911** polls every 15 min → `LEAFLETFA_MODELS/2026-07-13/WATCH.log`.

---

## 0. 🔴 THE ψ PRIOR — the finding that changed the grid

`ψ_kj ~ Beta(a_j, b_j)`. **`a_j`, `b_j` are indexed by junction only — there is no `k`.** So at every
junction, all K programs are drawn from the *same* learned Beta. That is a **shrinkage prior pulling the K
programs toward each other** — and "all K programs are the same" is exactly the symmetric saddle that is
this model's central failure mode. The prior pulls *toward* what training must escape. And because `a_j`/`b_j`
are *learned* (empirical Bayes), early training — when the programs really are all near each other — fits a
sharp Beta around their common value, which pulls them together harder still. Positive feedback into the saddle.

Measured on simulated data with known truth:

```
 psi prior           seed   psi_r   psi_contrast          loss
 junction-specific     0   +0.395      0.1800      9.1559e+04
 junction-specific     1   +0.158      0.0690      4.2330e+05
 junction-specific     2   +0.386      0.1683      6.5399e+04
 global                0   +0.994      0.2648      5.0664e+04
 global                1   +0.994      0.2588      5.0656e+04
 global                2   +0.993      0.2563      5.0653e+04
```

**The global prior recovers the truth AND removes the seed variance.** That second part matters enormously:
the run-to-run instability that motivated "fit three times, keep the best ELBO" (§5 below) may be a *symptom
of this prior*, not a property of the model. The ELBO also prefers the global prior, so best-ELBO selection
would have caught it — if the two priors had ever been in the same comparison. They never were.

**Not settled on one simulation.** So both variants are core axes and the held-out 30% decides. A
confirmation on the full Figure-2 sim is running. **If it reproduces, the atlas models refit with the global
prior and §4.2 of the Methods changes.**

---

## TL;DR — the config

```yaml
# config/mouse_smartseq2.yaml
n_factors:                 [10, 20, 50]     # K=5 dropped
param_grid_learning_rates: [0.05]
param_grid_input_conc:     [null]           # BETA-BINOMIAL ONLY. Never "inf".
param_grid_waypoints_use:  [True, False]
param_grid_num_passes:     20
param_grid_batch_sizes:    [8192]
num_inits:                 3                # 3 seeds, BOTH init arms
```

**Grid: K(3) × ψ-prior(2) × init(2) × seed(3) = 36 core + 2 Binomial ablations = 38 runs.**
Both the ψ-prior arms and the init arms are exactly balanced (18/18) — an unequal number of restarts per arm
would rig the comparison.

```python
# 01_generate_params.py  (already updated)
num_epochs_first: 150      # was 500
num_epochs_later:  60      # was 1000 — FEWER per chunk is better AND faster
gamma:            0.5      # was 0.01 (decayed the LR 100x)
patience:          50
min_delta_rel:   1e-5      # scale-free; the absolute min_delta was inert
```

**28 runs, ~1.8 h each, parallel across the SLURM array. Memory fits at every K** (K=50 needs 39 GB of 95 GB).

---

## What changed and why (all measured on `sim_heart_K5_*`, where ground truth is known)

### 1. 🔴 THE BIGGEST ONE — always Beta-Binomial. Never Binomial.

Swept on data where the **Binomial is the CORRECT model**:

```
 MODEL      lr   psi_r   agree   AUROC
 binom   0.05  +0.758  45.1%   0.914     <- the correct likelihood
    bb   0.05  +0.984  99.9%   0.998     <- the "wrong" one. FAR better.
```

**The Beta-Binomial wins even when the truth is Binomial.** Its concentration parameter absorbs residual
misfit and makes the objective easier to descend. `param_grid_input_conc: [null]`.

*Corollary:* `lr` **interacts with the likelihood** — Binomial (`conc=inf`) is much sharper, so an `lr`
swept on one is not valid for the other. Sweep them together.

*Corollary:* `bb_conc` is a **convergence diagnostic**. When ψ is wrong the model collapses the
concentration to soak up the residual; when ψ is right it recovers the truth (21.2 vs a true 20). Do not
report it as assay overdispersion unless ψ recovery is good.

### 2. Learning rate — the old default sat at chance

```
lr=0.001 (OLD DEFAULT) -> psi_r ~0, agreement = CHANCE, at ANY number of epochs
lr=0.02                -> psi_r 0.97
lr=0.05                -> psi_r 0.98
```

The model starts **on** a symmetric saddle: `AutoDiagonalNormal` initialises every cell's φ at the *prior
median* and all K programs at nearly the same ψ. Escape is driven by gradient noise growing exponentially,
and **its rate scales with `lr`**. At `lr ≤ 1e-3` it never escapes. `gamma=0.01` decayed the LR 100×,
killing it before the escape inflection.

`0.02` and `0.05` are both strong; **both are in the grid** so the atlas (a different regime — sparser,
20× more cells) can pick on held-out data.

### 3. Early stopping killed EVERY previous atlas batch

Across **9,239 batch fits** in the real SLURM logs: **median 10 epochs run** (nominal 500/1000), and
**9,239 / 9,239 ended via early stopping — not one ran to completion.** `min_delta` is absolute against an
ELBO of ~1e9, i.e. a relative tolerance of 1e-7 — inert. Now `min_delta_rel=1e-5`, `patience=50`.

### 4. Fewer epochs per chunk is BOTH faster and better

Counterintuitive but measured (`experiments/qpsi_width_under_chunking.py`): each chunk sees different cells
and disagrees with the others, so `q(ψ)` is pulled around by the disagreement — and training each chunk
*harder* makes it pull *harder*. `num_epochs_later: 60`, not 1000.

### 5. 3 seeds on the random-init arm — the Methods promises this and it was never done

The Methods says the random-init model *"is fit three times, and the one with the best ELBO was selected."*
`run_training.py` called `train(num_initializations=1)` with a **hardcoded seed 0**. Now a `seed` axis in
the grid, threaded through. **This matters:** the same config gave `psi_r = 0.758` (seed 0) and `0.223`
(seed 5973).

### 6. The waypoint arm is in the grid — and random init may now win

With the corrected schedule, **random init recovers simulated ground truth to ψ r = 0.98 / 100% program
agreement.** Waypoints were never the fix; the learning rate was. *"Programs are recovered from random
initialization; waypoints simply converge faster"* is a far stronger answer to the local-optima objection
than *"we needed waypoints"* — and the grid runs both arms, so you'll have the direct comparison.

(The waypoint initializer itself was also badly broken and is now fixed: it seeded ψ with **one random
cell's PSI**, dropped observed zeros, and left **~52% of entries at exactly 1.0** → a saturated logit,
gradient 1.2e-7 → **half the ψ matrix frozen before training started.**)

---

## Tomorrow, in order

### 1. Regenerate the params and sanity-check the JSON
```bash
cd Leaflet-analysis/GenomeBiology_Submission2026
python pipelines/05_leafletfa/01_generate_params.py config/mouse_smartseq2.yaml
# expect 28 rows; check `arm`, `seed`, `lr`, `input_conc: null` on a few
```

### 2. Launch ONE run first, not the array
Pick `K=20, lr=0.05, waypoints=False, seed=0`. Watch wandb for:

- **`psi_contrast`** (logged per batch) — must climb well above ~0.05. **If it sits near 0.003 the model
  has collapsed and the whole array is wasted.**
- `health_ok` — must stay 1.
- **epochs actually run per batch** — should be ~60, not ~10. If it's still ~10, early stopping is still
  firing and something didn't take.

**Do not launch the array until that one run looks right.** That is a 2-hour check against ~50 wasted
GPU-hours.

### 3. Launch the array (28 runs)

### 4. Evaluate on the 30% — and SELECT ON HELD-OUT METRICS, NOT THE ELBO
- `07c_classify_heldout_phi.py` — cell-type decodability from **held-out inferred φ**; the
  **train − heldout gap is the memorization estimate**.
- `04*_evaluate_imputation*.py` — masked-entry imputation vs NMF. **Report calibration, not just Spearman**
  (Reviewer 1): predicted-vs-observed PSI binned across the range, MAE/RMSE in PSI units, check for slope < 1.

**All of these now correctly FREEZE the globals** (`fixed_psi`, `pi_fixed`, `delta_fixed`, trained
`bb_conc`) and infer only φ. They used to pass `pi_init`, which let SVI **re-learn π on the test cells** —
leakage. See METHODS.md §7b.

> ⚠️ **Do not run the E-step against any model trained before 2026-07-12.** The old `bb_conc ~ Gamma(10,10)`
> prior (mean 1) pins the concentration near 1, which makes the likelihood nearly flat and φ uninformative.
> That is *why* `04e` had the comment *"the tiny learned bb_conc that leaves the likelihood ~flat"* and
> forced a Binomial as a workaround. The new `Gamma(2, 0.1)` prior fixes the cause.

### 5. Report the two robustness results the reviewers asked for
- **Initialization consistency** — the 3 seeds per (K, lr) give this directly. Report ψ cosine similarity
  and per-cell imputed-PSI correlation across seeds.
- **Empirical FPR** — already measured on the null simulation: **0 / 4,268 ATSEs called, FPR = 0.0000**
  (signal: realised FDR 0.079 at nominal 0.05, power 0.636). This is Reviewer 1's *"no clear empirical
  estimate of false positives"*. **Lead with it** — do not try to argue from the posterior width, which is
  not characterised and whose error is not even consistent in sign.

---

## Open / not done

- **`q(ψ)` posterior width under chunking is not characterised** (1.5× wider on a toy, 6× narrower on the
  real sim). **Accepted** — the point estimates agree with full-batch (r = 0.881) and that is what the
  figures rest on. Use the empirical FPR for type-I error.
- **`psi_contrast` is necessary but NOT sufficient.** A run scoring 0.045 passed a 0.04 gate while true-ψ
  recovery was only r = 0.22 and AUROC 0.525 (chance). `health_check` now warns in a marginal band, but
  **always confirm on held-out data.**
- **Methods text still to write:** the extra hierarchical level on the ψ prior (4 undeclared latents), the
  ε guards, the τ\* direction ("largest τ" → "smallest τ"), §7b, and §11.
