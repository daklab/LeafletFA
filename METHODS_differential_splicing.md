# Differential splicing in LeafletFA: two tests, and what each is for

Status 2026-08-06. Supersedes the fixed-threshold rule described in `METHODS.md` (§ DS), which was
`prob_de = P(TV > 0.10)` scored with `bayesian_fdr` and measured at **2.9x over nominal FDR** at
alpha=0.05 on simulations with known truth. Do not cite that rule.

There are now two tests. They answer different questions, use different parts of the model, and
have different failure modes. Neither replaces the other.

---

## Test 1 — Dirichlet-multinomial likelihood-ratio test (`A45`, `A49`)

Per ATSE, fit the DM twice on that event's **raw counts** and compare:

    H0:  alpha_j  = conc * softmax(b_j)                   one composition for every cell
    H1:  alpha_ij = conc * softmax(b_j + x_i . B_j)       composition varies with covariate x

    stat = 2 * (logL1 - logL0)  ~  chi2 with df = (m-1) * ncol(x)

Then Benjamini-Hochberg.

**Concentration is estimated per event, in BOTH hypotheses**, so it profiles out of the ratio. This
is not a refinement. With conc fixed at a global value the permuted-null calibration drifts with
effect size (ratio 0.65 / 1.75 / 3.82 at effect 0.25 / 0.5 / 1.0); estimating it per event collapses
that to 0.46 / 0.49 / 0.62 -- a 5.9x span becoming 1.3x. On real data there are no truth labels to
filter by, so that robustness is the only reason the null is trustworthy there.

**Identifiability.** The last junction's coefficients are pinned at zero in both fits. softmax is
invariant to adding a constant across junctions, so leaving all m free makes the optimum a ridge
rather than a point and inflates the apparent degrees of freedom.

**The covariate is the choice that matters:**

| design | x | what it tests |
|---|---|---|
| `--design phi` | learned program loadings | DS across structure discovered WITHOUT labels. LeafCutter cannot do this at all. |
| `--design group` | binary cell-type indicator | LeafCutter's own design. A CONTROL: isolates single-cell vs pseudobulk with the test held fixed. |

The control arm exists because LeafCutter's null is inflated largely by its sample size (7 vs 8
pseudobulk samples). Without it we would claim a modelling win for what is partly a data-volume win.

**What it takes from the trained model: phi only.** psi is re-estimated from raw counts inside every
event and the model's denoised psi never enters. That is conservative -- a DS call cannot be an
artefact of the model's own smoothing -- but it means this test does not exercise the model's main
product. That is what Test 2 is for.

**Validation** (simulations, known truth, 4 effect sizes x 3 seeds, all 4,268 ATSEs): realised FDP at
or below nominal everywhere, power 0.84-0.99.

---

## Test 2 — Posterior effect-size test on group contrasts (`A51`)

Uses the model's **denoised psi**, which Test 1 discards.

    Delta_j^(s) = dphi . psi_:,j^(s)          per posterior draw s, dphi = phibar_1 - phibar_2
    TV^(s)      = 0.5 * sum_{j in e} |Delta_j^(s)|
    report        P(TV > delta)  ->  local fdr = 1 - P  ->  BH

phi is averaged over the group FIRST, because rho is linear in phi; otherwise the [S, C, J] tensor
is 3 PB at atlas scale.

**delta is not a tuning knob.** It states the effect size of interest. On simulations it is 0.05
because that is how the simulator defines a differential event.

**Why this is an effect-size test and not a point null.** Three null constructions were built and all
failed:

| null | result | cause |
|---|---|---|
| permute group labels | 4,265 of 4,268 called, FDP = pi0 | dphi collapses to 0, so the null has no spread |
| permute phi | ratio rises WITH cell count | real between-program variation becomes an overdispersed mixture |
| collapse guide loc across programs, keep scale | 12-14x anti-conservative even at 2x width | -- |

Three failures with three distinct causes meant the null was not the variable. The hypothesis was.
`Delta = 0` exactly is essentially never true: every event carries some program dependence, psi is
denoised so its posterior noise is small, and with thousands of cells an arbitrarily small Delta is
detectably non-zero. A point-null test then calls nearly everything and is technically right each
time. Truth is defined by magnitude.

So the ORIGINAL shape -- `P(TV > delta)` -- was correct. Its faults were fixable ones: a biased
statistic and an unjustified delta.

**The bias fix.** Use `|mean_s Delta_j|`, not `mean_s |Delta_j|`. By Jensen, `E|X| >= |EX|`, so
mean-of-absolute cannot cancel estimation noise and is positively biased under the null -- truly null
ATSEs scored 0.017-0.025 instead of 0 under the old rule.

**Validation** (simulations, contrast-specific truth, no posterior width inflation):

| effect | FDP | ratio | power |
|---|---|---|---|
| 0.25 | 0.000 | 0.00 | 0.253 |
| 0.50 | 0.018 | 0.35 | 0.717 |
| 1.00 | 0.016 | 0.31 | 0.850 |

Conservative at every effect size.

**Scoring caveat that changes the numbers.** Power must be judged against the truth for THIS contrast
-- the true TV between the two groups being compared -- not against "differential across any program
pair". Scoring a 0-vs-1 contrast against the latter understated power by roughly 1.5x.

**Known limitation.** The variational posterior is 1.3-1.7x too narrow (mean-field VI minimises
KL(q||p), which is mode-seeking), so any probability read off its tail is over-confident. The
effect-size threshold absorbs most of this in practice -- calibration is conservative with NO width
inflation -- but the width itself is not fixed, and a sharper posterior would let delta be lowered.

---

## Calibration reference: LeafCutter on the same data

LeafCutter's own `permutation_null/` (60 runs, 453,715 null p-values, donor labels shuffled within a
single cell type -- a clean null with no signal to contaminate it):

| nominal | LeafCutter ratio | LeafletFA ratio |
|---|---|---|
| 0.50 | 1.36 | 1.09 |
| 0.20 | 2.15 | 1.55 |
| 0.05 | 4.48 | 3.27 |
| 0.01 | 11.25 | 9.72 |

By event size, LeafCutter degrades sharply (3.23 at m=2 to 11.35 at m=9) while LeafletFA stays near 3
across all m. Sample size explains it: 7 vs 8 pseudobulk samples against 218-2,870 single cells.

**Both caveats favour LeafletFA and should be stated.** LeafCutter's null is clean, so 4.48 is an
honest measurement. LeafletFA's is contaminated by real signal, so 3.27 is an UPPER BOUND that
already sits lower. However the two permutations are not the same construction, so the exact ratios
should not be quoted as a matched benchmark until both are run on one null -- `A49 --design group`
on LeafCutter's own permuted arms would make it exact.

---

## Which to use

- **Programs, no labels available** -> Test 1 with `--design phi`. Only option, and the novel claim.
- **Two named groups of cells** -> Test 2. Uses the denoised psi and states an effect size.
- **Head-to-head against LeafCutter** -> Test 1 with `--design group`, which holds the test fixed and
  varies only single-cell vs pseudobulk.
- Report both when comparing to LeafCutter. Test 1 is more powerful; Test 2 is more conservative and
  exercises the part of the model that is actually novel.
