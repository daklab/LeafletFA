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

Per the Pyro docs, the plate:
- draws subsample indices **once** and uses the same ones in model and guide
  (*"the indices will only be subsampled once in the guide; the Pyro backend makes sure that the
  same set of indices are used during execution of the model"*),
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

**Results: pending (job 19279480).**

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

- [ ] Variant comparison on simulated data (job 19279480)
- [ ] Best variant on real 70% training data, checking ψ lands near 0.295
- [ ] 07d: does the test-time E-step converge at 300/800/2000 epochs (job 19278165)
- [x] ELBO decomposition (job 19278771): **confirmed optimiser, not prior** — see §2
- [ ] Decide whether `learn_ab_hyperprior` is dropped permanently, and reconcile METHODS.md §8
- [ ] All 39 existing models are affected; Figure 3 cannot be built on them
