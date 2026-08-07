# Plan: to 9 trained atlas models, an informative latent space, and calibrated DS

Written 2026-08-06. Each stage has an explicit acceptance test, so "done" is a measurement rather
than a judgement. Stages 1–3 are chained and running; 4–5 need a decision from us.

---

## Stage 1 — Choose the concentration treatment  ·  RUNNING

Four full K=20 seed-0 fits differing only in how the per-event concentration is handled, each
holding out the **same 10% of cells** (split seed fixed at 12345) and scoring them:

| arm | treatment |
|---|---|
| `free` | unbounded — the baseline that collapsed to 0.52 |
| `cmin2` | bounded ≥ 2.0 |
| `cmin5` | bounded ≥ 5.0 |
| `global` | one concentration shared by all 24,475 events |

**Accept:** the arm with the highest held-out likelihood gain over a null of one shared composition.

**Why this stage exists at all.** `conc_min = 2.0` was originally chosen by scoring the Fbln2
panel — one ATSE, and the showcase figure. That is hyperparameter selection on the illustration,
which is a fair referee objection. This replaces it with held-out likelihood over all events.

**Prefer `global` if it is within noise of the best.** It needs no tuned constant, so the recipe has
nothing to defend. It scored 101.3% of the cell-type reference on training likelihood against
`cmin2`'s 97.9%, so it is the likely winner, but that measurement could not detect overfitting.

**Guard:** the chained selector refuses to proceed if held-out numbers are missing rather than
falling back on the Fbln2-chosen value. It already caught one bug this way.

---

## Stage 2 — Train the 9 models  ·  CHAINED

K ∈ {10, 20, 30} × seeds {0, 1, 2}, from ONE commit, with the Stage-1 setting.

    16,200 steps (1,000 epochs) · batch 8192 · lr0 0.025, gamma 0.1 · early stopping OFF
    per-ATSE alpha floor 1e-2 · 40 NaN restarts · checkpoint every 2,000 steps
    --holdout-frac 0.10 so every fit reports its own held-out likelihood
    trajectory tracked every 200 steps, phi UMAP + silhouette logged to wandb

**Accept, per fit:**
- reaches 16,200 steps (the previous round managed 7 of 9; the two failures were a slow node and an
  exhausted restart budget, both now addressed by a 24h limit and 40 restarts)
- `psi_valid = 1.00`, max deviation < 1e-3
- `psi_contrast` settled — < 0.5% change over the last tracked interval
- held-out likelihood gain > 0

**Note on comparability.** The previous batch straddled a commit, so fits launched together ran
different code and picked up different restart budgets. Anything compared across K or seed must come
from a single submission.

---

## Stage 3 — Is the latent space informative?  ·  CHAINED (A46 + A54)

Three distinct questions, each with its own number:

| question | measure | target |
|---|---|---|
| are the programs compositional? | `psi_valid`, max deviation | 1.00, < 1e-3 |
| do they carry biology? | silhouette + ARI vs held-out `broad_cell_type` | ARI ≥ 0.38 (the best so far at K=20) |
| do they explain data better than labels? | held-out likelihood vs a 38-cell-type-pooled reference | ≥ 100% |
| are they discriminative between groups? | Fbln2 `‖Δφ̄‖₁` | ≥ 1.2 (released model: 1.44) |
| do independent seeds agree? | matched psi r on LIVE programs; share of live programs shared | r ≥ 0.95; share reported honestly |

**Two things must be reported, not buried:**

- **Effective K is 5–11 regardless of nominal K.** The extra programs carry no cells and drift to a
  common average composition, so the K sweep is a robustness check, not model selection. There is no
  principled K-selection criterion in place and we should not imply one.
- **Cross-seed agreement must be quoted live-only.** Dead programs match each other at r ≈ 0.93
  because they all converge to the same average, which inflates a naive "100% reproduced" to
  meaninglessness.

---

## Stage 4 — Calibrated DS on the final fits  ·  CHAINED, but see the caveat

Three arms, three different claims. They are not interchangeable and none subsumes another.

| arm | design | claim it supports |
|---|---|---|
| A49 `--design phi` | learned loadings, df = (m−1)·K | **the model-specific one**: DS across structure found without labels. LeafCutter cannot run this. |
| A49 `--design group` | binary cell-type indicator, df = (m−1) | a matched head-to-head. Isolates single-cell vs pseudobulk with the test fixed. |
| A51 posterior | `P(TV > δ)` on the denoised ψ | uses the part of the model the LRT discards. |

Plus A52 and A53, which run the LRT and the posterior test on **LeafCutter's own null
construction** — donors shuffled within one cell type, arm sizes matched to their manifest.

**Everything currently reported for A51/A52/A53 must be re-run.** All three used the collapsed-φ
model (conc median 0.520). For A52 that probably does not matter — its covariate is a binary
indicator and φ only supplies a concentration initialisation. For A51 and A53 it matters a great
deal: the statistic is `Δ_j = dφ · ψ`, so a collapsed φ makes every Δ small and few events clear δ.
The "0.53% called on a true null" may be an artefact of the collapse rather than calibration.

**Accept:**
- A52 ratio at α = 0.05 near 1 (currently 1.42 against LeafCutter's 4.48, on the old fit)
- A53 calls ≤ nominal on the true null at the chosen δ
- A49 `--design phi` produces calls that overlap the `--design group` calls at better than chance

---

## Stage 5 — What still needs a decision

**1. What the LeafCutter comparison actually claims.** The `group` arm is a clean matched
comparison, but what it shows is that testing across thousands of single cells beats 7-vs-8
pseudobulk samples. That is a data-design result, not evidence the model is better — LeafCutter on
equally-sized pseudobulks would presumably do the same. The φ arm is the model claim, and it cannot
be a head-to-head because LeafCutter cannot run it. It should be framed as *recovering the labelled
contrasts without the labels*, i.e. agreement, not superiority.

**2. δ for the posterior test on real data.** 0.05 transfers from simulation on the old fit
(A53: 0.53% called at a nominal 5%). It must be re-checked on the final fit, and δ = 0.02 was already
too permissive (2.6% mean, 13.8% worst case).

**3. Whether K = 30 earns its compute.** It yields 8–11 live programs for 3× the cost of K = 10.
Keeping it is defensible as a robustness demonstration; presenting it as model selection is not.

**4. A held-out imputation benchmark.** Everything reported so far is reconstruction on training
cells. The A3 held-out evaluation is the separate, stronger claim and has not been re-run on these
models.

---

## Known failure modes, so they are not rediscovered

- **`num_steps` sets the learning-rate schedule** via `lrd = gamma**(1/num_steps)`, not just the
  budget. Any run compared to production must use 16,200 steps and be bounded by wall-clock. This
  invalidated three separate experiments before it was noticed.
- **Slurm does not reliably flush `.out`.** `result.json` is the source of truth; a launcher that
  parsed `.out` once concluded "no result" from a completed run.
- **Array tasks need a real index.** Passing "all contrasts" to every task means N-fold redundant
  work, and concatenating `*_*/` then stitches duplicates.
- **The ELBO cannot judge convergence here** — its whole drop lands in the first ~50 steps and the
  tail slope flips sign between fits. Use `psi_contrast`.
