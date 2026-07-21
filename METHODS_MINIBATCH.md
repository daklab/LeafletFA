# Minibatch SVI for LeafletFA — diagnosis, redesign, and validation

Working document. Started 2026-07-20. Records the failure that motivated the redesign, the
evidence for each hypothesis (including the ones that turned out wrong), the candidate fixes, and
their test results.

---

## 1. The failure

Measured on the 2026-07-18 mouse atlas grid (70% split, 92,894 cells, 79,768 junctions), across
**all 39 completed models** regardless of K (10/20/50), initialisation (waypoint or random), prior
type (global or junction-specific), or likelihood (Beta-Binomial or Binomial):

| quantity | value |
|---|---|
| learned Ψ mean | **0.670 – 0.720** |
| observed splicing rate, read-weighted `sum(junc)/sum(clus)` | **0.2951** |
| observed PSI, unweighted mean over covered entries | 0.2928 |
| Ψ fraction below 0.10 | 2.2% |
| true PSI exactly 0 (covered entries) | **58.7%** |

Ψ is **mislocated by ~ +0.37**. It is not garbage — it ranks junctions correctly
(`corr(π@Ψ, observed per-junction mean) = +0.68`, n = 76,791) — but it sits on the wrong absolute
scale, with regression `observed ≈ 1.15 × model − 0.47`.

Consequences: `lf = φ@Ψ` has bias **+0.28** and MAE **0.338 vs 0.108 for a per-junction mean** —
three times worse than the naive baseline. Any absolute-scale claim (imputation, calibration) built
on these models measures this failure rather than the method.

### Trajectory

Ψ *initialises correctly* and then moves away:

```
init (coverage-aware mean)  0.305
pass 1                      0.7122   (r vs init = 0.301)
pass 2                      0.6954
...
pass 8                      0.6700   (r vs init = 0.409)
```

Identical without waypoints (task 3: 0.644 → 0.696 monotonically; task 33: 0.677 → 0.677). So it is
not an initialisation problem.

---

## 2. Hypotheses tested

Recorded including the failures, because each wrong turn narrowed the search.

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| 1 | Waypoint init samples random **non-zero** values → biased high (audit §4) | ❌ **refuted** | Already fixed; `initialize_psi_phi` uses the coverage-aware mean and the pipeline **does** pass `mask` (waypoints.py:218, :373). No warning in any training log. Waypoints ≡ random (0.6927 vs 0.6931). |
| 2 | Missing N/B likelihood rescaling → prior over-weighted (audit §2) | ❌ **refuted** | `obs_scale = n_cells / len(batch)` is implemented and applied via `_ScaleSites(obs_scale, {"assign","obs"})` — the likelihood **and** local φ terms are scaled, globals at weight 1. Correct. |
| 3 | Beta prior with `a > 1` has zero density at ψ=0, so ψ **cannot** reach 0 | ⚠️ **overstated** | True that density at *exactly* 0 is 0 (11/13 models have a>1, prior modes 0.85–1.0), but ψ = 0.01 is reachable and merely penalised. Not a prohibition. |
| 4 | Junction/cluster **indexing bug** — model fits 1−PSI or misaligned counts | ❌ **refuted** | `corr(model, observed) = +0.678`; the flip gives −0.678. And the likelihood sweep peaks at the observed rate, which is what a *correct* likelihood does. |
| 5 | Held-out artifact — Ψ compared against the wrong data | ❌ **refuted** | Training PSI 0.2928 vs test 0.2929 (gap 0.0001), identical 58.7% zeros and 13.5% coverage. Split is distribution-matched. |
| 6 | Read-weighting — likelihood targets a different rate than the unweighted mean | ❌ **refuted** | read-weighted 0.2951 vs unweighted 0.2928, per-junction corr 0.979. Same target. |
| 7 | **Optimisation failure** — the objective wants low ψ, training never gets there | ✅ **supported** | See §3. |

### The decisive magnitude argument

The ψ-shift test (hold φ fixed, shift ψ in logit space, score the Beta-Binomial log-likelihood of
the real counts):

```
delta   psi_mean  logLik/entry
 0.00    0.6700     -3.25863     <- trained model
-2.50    0.3121     -2.16758     <- BEST
observed data rate = 0.2956
```

The likelihood prefers ψ ≈ 0.31, essentially the observed rate, and the gain is **1.09 nats per
observed entry**. Over ~1.0 × 10⁹ covered entries in the training set that is **~10⁹ nats**. The
prior can contribute at most one Beta log-density term per ψ value — K × J = 4 × 10⁶ values, O(1–5)
nats each, so **~10⁷ nats**.

**The likelihood outweighs the prior by 50–100×. The prior cannot be what holds ψ at 0.67.**
This is an optimisation failure, not a misspecified model.

### Confirmed by decomposing the actual objective (job 19278771)

Sweeping the same logit shift and evaluating **both** ELBO terms — likelihood scaled to the full
training set, prior summed over all K × J values of ψ:

| δ | ψ mean | logLik (full) | logPrior | TOTAL |
|---|---|---|---|---|
| +0.50 | 0.7431 | −3.8418e9 | +9.90e5 | −3.8408e9 |
| 0.00 | 0.6700 | −3.2860e9 | +2.43e5 | **−3.2858e9** ← trained model |
| −1.00 | 0.5134 | −2.5449e9 | −2.10e6 | −2.5470e9 |
| −2.00 | 0.3717 | −2.2307e9 | −5.56e6 | −2.2362e9 |
| **−2.50** | **0.3121** | −2.1858e9 | −7.66e6 | **−2.1935e9** ← best |
| −3.00 | 0.2604 | −2.1939e9 | −9.96e6 | −2.2038e9 |

Moving ψ to 0.31 costs the prior **7.9 × 10⁶ nats** and gains the likelihood **1.10 × 10⁹ nats** —
the likelihood wins by **139×**, and the total objective improves by **1.09 × 10⁹ nats**.

**Verdict: the objective is optimal near ψ = 0.31 and training never reached it. Optimisation
failure, confirmed on the real objective. The prior is exonerated.** The curve is smooth and
unimodal with a clear interior optimum, so this is not a local-minimum artifact of the sweep.

---

## 3. Root cause: no plate, so no real minibatching

`leafletfa/model.py:99`:

```python
assign = pyro.sample("assign", dist.Dirichlet(pi * dir_conc).expand([N]).to_event(1))
```

`.expand([N]).to_event(1)` folds all N cells into a **single event**. There is no `pyro.plate`, so
Pyro cannot subsample. The pipeline works around this by rebuilding a fresh `LeafletFA` — and a
fresh guide — on every chunk: **96 rebuilds** for the atlas (12 chunks × 8 passes).

`AutoDiagonalNormal` makes this unavoidable. From the Pyro docs:

> In `AutoDiagonalNormal`, if your model has N named parameters with dimensions k_i and sum k_i = D,
> you get a single vector of length D for your mean, and a single vector of length D for sigmas.

It flattens every latent into one vector, so it has no plate structure and **does not support local
variables under subsampling**. `AutoNormal` keeps one distribution per site and accepts
`create_plates`, which is what makes subsampling work.

Two consequences of the rebuild loop:

1. **q(ψ)'s scale is reset to the default width on every chunk** — model.py:857 says so outright:
   *"reset to AutoDiagonalNormal's default width on every chunk and never accumulates evidence."*
   96 resets, so the posterior never sharpens.
2. **φ never commits** (measured perplexity 27.2 of K=50). The likelihood gradient on ψ_kj scales
   with φ_ck, so diffuse φ dilutes the likelihood's pull on ψ by ~K× while the prior's pull is
   undiluted. Programs stay similar, cells cannot distinguish them, φ stays diffuse — self-
   sustaining. This is the symmetric saddle described in METHODS.md §11.1, not fully escaped.

---

## 4. The redesign — `leafletfa/minibatch.py`

One model, one guide, one optimiser, alive for the whole run. Nothing is rebuilt.

```python
with pyro.plate("cells", N, subsample_size=B) as idx:
    phi = pyro.sample("phi", dist.Dirichlet(pi * dir_conc))      # batch [B], event [K]
    mu  = torch.clamp(phi @ psi, clip, 1 - clip)
    pyro.sample("obs",
        dist.BetaBinomial(concentration1=mu*bb_conc, concentration0=(1-mu)*bb_conc,
                          total_count=clus[idx]).mask(mask[idx]).to_event(1),
        obs=junc[idx])
```

The plate:
- draws subsample indices **once** in the guide and replays the same ones into the model — *verified*
  on our wiring, not just assumed; see §4a for the index-identity proof and the bug that had to be
  fixed before it actually held,
- **auto-scales** the local terms by N/B — no manual `poutine.scale` bookkeeping,
- leaves globals (ψ, π, a, b, dir_conc, bb_conc) at weight 1, which is correct.

Guide is `AutoNormal(model, create_plates=...)`. Local variational parameters stay full-size
(N × K) and are indexed by the plate — **no amortisation needed**: at N = 92,894 and K = 50 that is
~37 MB for loc and scale together.

Other deliberate choices:
- `learn_ab_hyperprior=False` by default. METHODS.md specifies `a, b ~ Gamma(1,1)` with no extra
  level; the learned level (`a_shape/a_rate/b_shape/b_rate`) is undocumented (audit §8) and is what
  drives the learned prior mean to 0.696 ≈ the mislocated ψ. Kept as a togglable variant to test.
- ψ init clipped to [1e-4, 1−1e-4]. Pyro maps a Beta site through a logit and ψ_init = 1.0 becomes
  a loc of +15.9 where the sigmoid gradient is ~1e-7 — frozen for the whole run.
- Coverage comes from `clus > 0`, never `junc > 0`. An observed PSI of 0 is **data**, not missing;
  58.7% of covered entries are exactly 0.

### First smoke test (tiny sim, N=400, J=60, K=3, CPU, 150 steps)

```
sim: true psi mean=0.3468   observed rate=0.3495
  step   0  psi_mean=0.4920      <- starts at the prior median
  step  50  psi_mean=0.3421
  step 149  psi_mean=0.3455
fitted psi mean=0.3455   TRUE=0.3468   observed=0.3495
```

ψ moves **down** to the truth. The shipped code moves **up** to 0.71 and stays. The location failure
does not reproduce under the plate model.

### 4a. Two bugs found AFTER the first smoke test (fixed in `c8cad3d`)

The tiny smoke above and the whole §5 simulation sweep passed while **subsampling was silently not
engaging**. Both bugs matter for anyone reading this code.

**Bug 1 — the guide traced the full N, so `phi @ psi` was an (N, J) matrix.** SVI optimises the model
*and* the guide, and both trace the plate independently. The original wiring gave the batch size to
only one side:

```python
# BROKEN
svi   = SVI(lambda: self.model(subsample_size=B), guide, ...)   # model side sees B
guide = AutoNormal(self.model, create_plates=...)               # traces raw self.model() -> None -> full N
```

`AutoNormal` wraps the **raw** `self.model` and calls it with no arguments, so on the guide side
`subsample_size=None` and the `"cells"` plate ran at full N. The guide therefore instantiated `phi`
over all 92,894 cells and `phi @ psi` became `(92894, 79768)` = **29.6 GB**, which OOM'd the first
real 70% run (job 19279941) at exactly a 27.61 GiB allocation.

It was invisible on the K=5 / J=400 simulation because a full `(N, J)` matrix there is only ~4.8 MB.
**Every "plate" result in §5 was therefore silently full-batch, not subsampled** — the plate fix is
still correct, but subsampling itself was never exercised until atlas scale.

Fix: the batch size is an instance attribute `self._B`, read by both the model and the guide's
`create_plates`, and the **same** `self.model` object is passed to both `SVI` and `AutoNormal`:

```python
# FIXED
self._B = B
svi   = SVI(self.model, guide, ...)                            # same object, no lambda
guide = AutoNormal(self.model,
                   create_plates=lambda *a, **k: pyro.plate("cells", N, subsample_size=self._B))
```

**Verified, not assumed.** Replaying the model against the guide trace the way `Trace_ELBO` does, the
two `"cells"` plates use the *identical* index set (this is the correctness-critical part — φ for
cell *c* must be scored against cell *c*'s reads):

```
guide 'cells' idx : (256,) [2382, 2336, 884, 3435, 4660, 3142, ...]
model 'cells' idx : (256,) [2382, 2336, 884, 3435, 4660, 3142, ...]   IDENTICAL: True
model phi shape   : (256, 4)      # B, not N
model obs shape   : (256, 300)    # B x J, not N x J
```

The guide subsamples once; the model's same-named plate is replayed onto those exact indices. A
smoke test asserting `phi`/`obs` are batch-sized (256), not N-sized (5000), is now part of the
regression checks — the check the §5 sweep never made.

**Bug 2 — data must not sit dense on the GPU.** The atlas is 92,894 × 79,768 → 29.6 GB per matrix in
float32, 59 GB for `junc` + `clus`, on a 44.5 GB card. `data_on_cpu` (auto-enabled past ~4 GB per
matrix) keeps both matrices in host memory and moves only the `(B, J)` minibatch to the device each
step (~653 MB at B=2048). Verified numerically identical to the all-on-device path on the smoke test.

**Status:** the real 70% run (job 19286437, K=20, 3000 steps, batch 2048) is the first genuine test
of end-to-end minibatching. Pending on the shared `knowles_lab` GPU quota. The number that decides
everything: ψ mean must fall toward the observed rate **0.2948**.

---

## 4b. THE deciding result — real 70% atlas, genuine minibatch (job 19286437)

First run with subsampling actually engaging, on the full 92,894 × 79,768 matrix (K=20, batch 2048,
lr 0.05, gamma 0.5, 3000 steps). **ψ mean fell from the prior median straight to the observed rate:**

| step | ψ mean | | step | ψ mean |
|---|---|---|---|---|
| 0 | 0.4938 | | 900 | **0.2947** ← observed rate = 0.2948 |
| 100 | 0.3016 | | 1500 | 0.2896 |
| 300 | 0.2972 | | 2700 | 0.2807 |

**The mislocation is fixed.** Every shipped chunked model sits at ψ ≈ 0.67; the plate model lands on
the data. Loss dropped 1.89e9 → 9.7e8 and held stable. This is the confirmation the whole
investigation was after.

**Two things this run also exposed:**

1. **A late-training NaN blow-up.** The loss went non-finite at **step 2745** (after 40 min of
   stable training) and corrupted every saved parameter. Fixed with a NaN-guard + checkpoint-best in
   `fit()` (`_param_snapshot`/`_restore_snapshot`): the param store is snapshotted on each stable
   log interval, and a non-finite loss restores the last good snapshot and stops. The returned model
   is always the best finite state, never the blown-up one. Verified on a smoke test.
2. **ψ drifts monotonically past the target** (0.2947 @ 900 → 0.2807 @ 2700) rather than settling.
   mean(ψ) over the K×J matrix need not exactly equal the read-weighted rate, but the continued
   descent + eventual blow-up says the LR/step budget over-cooks it. The model was best around step
   900–1500. Rerun (job 19286831) uses **1500 steps, gamma=0.1** (stronger decay) with the guard on.

---

## 5. Candidate fixes under test

`experiments/minibatch/compare_variants.py`, simulated data with the atlas's characteristics
(13.5% coverage, Beta(0.30, 0.72) ψ so the truth is U-shaped with mass at 0), K=5, 3 seeds.

| variant | what it isolates |
|---|---|
| `chunked_baseline` | the shipped approach — rebuild per chunk, AutoDiagonalNormal, learned a/b. Should reproduce the bug. |
| `plate_fixed_ab` | plate + AutoNormal, `a,b ~ Gamma(1,1)` per METHODS |
| `plate_learned_ab` | as above **with** the learned hyperprior → does that level still matter once the optimiser is fixed? |
| `plate_junc_prior` | per-junction `a_j, b_j` |
| `full_batch` | no subsampling — the reference ceiling |

### Sanity checks (the intuitions accumulated while debugging)

| check | why it exists |
|---|---|
| `psi_location_err` | **the bug**: 0.67 vs 0.29. Nothing in the old health check looked at location. |
| `psi_recovery_r` | Hungarian-matched corr with true ψ — is the structure learned at all? |
| `beta_slope` | regress fitted effect size on true. **Scale-invariant metrics cannot see attenuation**: shrink every factor toward a common per-junction mean and Pearson r, cosine and argmax all stay ≈1 while β collapses. Slope ≈ 1 is the only thing that certifies DS effect sizes (audit §2). |
| `psi_contrast` | do the K programs differ? |
| `phi_perplexity` | near K ⇒ cells hedge ⇒ diffuse φ ⇒ likelihood gradient on ψ diluted ~K×. |
| `phi_recovery_r` | matched corr on cell loadings. |

### Results — simulated data, K=5, J=400, N=3000, 1500 steps (seed 0; seed 1 replicates)

| variant | ψ mean (true 0.3026) | ψ location err | r_psi | **β slope** | φ perplexity (max 5) | seconds |
|---|---|---|---|---|---|---|
| `full_batch` (ceiling) | 0.3014 | 0.0012 | 0.999 | **1.001** | 2.4 | 157 |
| `plate_fixed_ab` | 0.3007 | 0.0019 | 0.998 | **1.012** | 2.6 | 75 |
| `plate_learned_ab` | 0.3018 | 0.0008 | 0.998 | 1.014 | 2.6 | 64 |
| `plate_junc_prior` | 0.3007 | 0.0019 | 0.998 | 1.011 | 2.6 | 54 |
| **`chunked_baseline`** | **0.0266** | **0.2760** | **0.017** | **0.006** | **4.5** | 110 |

Seed 1, `plate_fixed_ab`: ψ err 0.0012, r_psi 0.998, slope 1.016, perplexity 2.6 — replicates.

**Findings**

1. **Plate minibatching matches full-batch and is 2-3x faster.** β slope 1.012 vs 1.001 means no
   effect-size attenuation. This is the number audit §2 asked for as the one thing that decides
   whether minibatch training shrinks DS effect sizes -- for the PLATE implementation the answer is
   no attenuation. (It says nothing about the shipped chunked-rebuild approach, whose slope has
   never been measured on real data.)
2. **The chunked-rebuild pattern recovers nothing** under identical conditions: r_psi 0.017 is
   chance, β slope 0.006 is total effect-size collapse, and φ perplexity 4.5/5 means cells hedge
   across every program -- the same signature as the real atlas (27 of K=50, ~90% of K).
3. **The a/b hyperprior is exonerated.** `plate_learned_ab` performs identically to
   `plate_fixed_ab`. It was a symptom, not a cause -- consistent with the ELBO decomposition. Had
   the prior been "fixed" first, the model would have changed for no reason and possibly masked the
   real bug.
4. **The junction-specific prior is neutral** here. Keep it as an ablation arm, not the default.

**Two caveats on the control**

- **The direction of failure differs.** On the real atlas ψ drifts UP (0.67 vs 0.295); in this
  simulation the chunked baseline collapses DOWN (0.027 vs 0.303). Both are massive mislocation
  with no recovery, but the simulation does not reproduce the specific upward drift, so it shows
  *that* the pattern is broken rather than that it breaks by the identical route.
- **`chunked_baseline` is a simplified reimplementation**, not production `run_training.py`: it
  lacks the waypoint ψ warm-start via `varm`, φ carry-over, and the real pass/epoch schedule. It is
  probably worse than production, which does achieve `corr(π@Ψ, observed) = 0.68` on real data.
  **r_psi = 0.017 must not be read as "production is that bad."**

**Nomination: `plate_fixed_ab`.** All plate variants are statistically indistinguishable, so the
choice is made on other grounds: it uses `a, b ~ Gamma(1,1)` exactly as METHODS.md documents,
removing the undocumented hyperprior level (audit §8) at zero measured cost.

---

## 6. Instrumentation added so this class of bug is caught on run 1

`health_check` (committed, `baed55b`) now reports, and stores in `self.health`:
- ψ mean/median/p1/p25/p75/p99, frac < 0.05, frac > 0.95
- `φ@ψ` (predicted PSI) mean/median/p1/p99/IQR
- `psi_median_minus_observed` when the caller sets `model.observed_psi_rate`, raising a problem past
  a 0.15 gap

Both eval scripts now record `epochs_ran`/`stopped_early` per chunk and warn when most chunks hit
the cap. Measured: **10/10 chunks hit the cap at both 100 and 300 epochs**, so held-out φ is
truncated, not converged — an independent symptom of the same under-optimisation.

wandb consolidated to one project (`LeafletFA-mouse-atlas-70pct-train`) with the array date in
`group`, so arrays can be compared instead of each landing in its own workspace.

---

## 7. Open items

- [x] Variant comparison on simulated data — **plate matches full-batch; chunked recovers nothing** (§5)
      — CAVEAT: subsampling was silently off during this sweep; results were effectively full-batch (§4a)
- [x] Subsampling wiring bug + atlas-scale memory (§4a) — fixed and index-sync verified (`c8cad3d`)
- [x] **Real 70% data with genuine minibatching (job 19286437) — ψ hit 0.2947 at step 900 = the
      observed rate 0.2948. MISLOCATION FIXED (§4b).** NaN blow-up at step 2745 → added NaN-guard +
      checkpoint-best to fit(); gentler rerun (job 19286831, 1500 steps, gamma=0.1) in progress.
- [x] Array trainer `run_training_minibatch.py` written — drop-in for the 48-config grid (reads the
      same parameter_combinations.json; honours K/seed/junc_prior; ignores chunked-only knobs). Emits
      a single continuous ELBO loss curve + psi-location plot per run, and logs health to one wandb
      project. Smoke-tested (job 19286833). **NOT launched** until the gentler single run confirms
      the settings.
- [ ] Launch the 48-config minibatch array once 19286831 confirms ψ settles cleanly with no NaN.
- [x] 07d (job 19278165): **the E-step does not converge, and early stopping fires prematurely**

### 07d result — test-time E-step convergence (run_30, K=50, 4096 held-out cells)

| epoch budget | epochs actually run | mean_L1 vs prev | corr vs longest | argmax agree vs longest | φ perplexity |
|---|---|---|---|---|---|
| 300 | 300 (hit cap) | — | 0.374 | **0.110** | 27.21 |
| 800 | 800 (hit cap) | 0.738 | 0.818 | 0.495 | 17.40 |
| 2000 | **1158 (early-stopped)** | 0.515 | 1.000 | 1.000 | 11.79 |

**At the 300-epoch production budget only 11% of cells have the same dominant program as at 1158
epochs**, and φ perplexity falls monotonically 27.2 → 17.4 → 11.8, i.e. φ is still sharpening
throughout. Every 04e and 07c number is computed on a 100- or 300-epoch E-step, so all of them are
computed on a φ that reorders almost completely with more compute. In particular the 07c result
"φ loses to rank-K NMF on every structure metric" is **not trustworthy** and must be re-run.

Note also that the φ perplexity of 27/50 previously cited as evidence for "diffuse φ / symmetric
saddle" is substantially **truncation**, not saddle behaviour: at 1158 epochs it is 11.8.

#### Root cause: early stopping is inert in the E-step path

The threshold is `max(min_delta, min_delta_rel * |best_loss|)`. The training array sets
`min_delta_rel = 1e-5`, but the E-step (04e:421, 07c:152) passed only `min_delta = 0.001` and
`min_delta_rel` **defaults to 0.0**. Against an E-step ELBO of ~2.87e7 that is a relative tolerance
of ~3.5e-11 — completely inert — so the rule degenerates to "stop after `patience` epochs with no
new global best", which single-particle SVI noise triggers on a still-descending curve. It fired at
epoch 1158 while φ was still moving by mean L1 0.515. This is exactly the audit §6.3 problem, fixed
for training and never fixed for the E-step. **Fixed: both scripts now pass `min_delta_rel=1e-5`.**

Also note `health["stopped_early"]` is defined as `n_ran < num_epochs AND frac_ran < 0.25`, so it
reports False for a genuine early stop past 25% of the budget — it reported False at 1158/2000. It
is a "stopped suspiciously early" flag, not a "stopped early" flag, and should not be read as the
latter.
- [x] ELBO decomposition (job 19278771): **confirmed optimiser, not prior** — see §2
- [ ] Decide whether `learn_ab_hyperprior` is dropped permanently, and reconcile METHODS.md §8
- [ ] All 39 existing models are affected; Figure 3 cannot be built on them
