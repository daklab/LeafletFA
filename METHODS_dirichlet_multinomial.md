# The Dirichlet-multinomial + simplex model

Verified 2026-08-04. Every claim below is measured, and the check that produced each number is
named so it can be rerun.

## The generative model

```
bb_conc        ~ Gamma(2, 0.1)                 <- LeafCutter: conc ~ gamma(concShape, concRate)
psi_logit[k,j] ~ Normal(0, 1)
psi[k, .]      = softmax within each ATSE      <- LeafCutter: softmax(beta . x)
alpha_pi       ~ Gamma(1, 1);  pi ~ Dirichlet(alpha_pi / K)
dir_conc       ~ Gamma(1, 1)
phi[c]         ~ Dirichlet(pi * dir_conc)      <- LeafletFA-specific: the low-rank mixture
mu[c, j]       = phi[c] @ psi                  <- the composition for cell c
alpha[c, j]    = bb_conc * mu[c, j]            <- LeafCutter: a = conc * softmax(...)
y[c, .]        ~ DirichletMultinomial(alpha)
```

Implemented in `leafletfa/minibatch.py` via `likelihood="dirichletmultinomial"` and
`psi_parameterization="simplex"`, with the segment machinery in
`leafletfa/dirichlet_multinomial.py`.

## Why this is the same statistics as LeafCutter

LeafCutter's `dm_glm.stan` is:

```stan
conc ~ gamma(concShape, concRate);
a = conc * softmax(beta * x[n]);
target += lgamma(suma) + sum(lGaPlusY) - lgamma(suma + sum(y[n])) - sum(lGaA);
```

Three correspondences, each checked numerically:

1. **The likelihood is term-for-term identical.** Our `dm_log_prob` evaluates
   `sum_e [lgamma(A_e) - lgamma(N_e + A_e)] + sum_j [lgamma(y_j + a_j) - lgamma(a_j)]`, which is
   the same expression rearranged. On matched inputs: LeafCutter's Stan target -114.362812, ours
   -114.362793. SpliceVI's independent implementation (`splicevae.dirichlet_multinomial_likelihood`,
   as `per_cell_atse - per_cell_junc` with `nbetaln`) agrees to -8.906084 on a separate test.
2. **The concentration prior matches.** `Gamma(2, 0.1)` against their `gamma(concShape, concRate)`,
   playing the same role: `alpha = conc * composition`.
3. **The composition is a within-group softmax in both.** This is why `psi_parameterization="simplex"`
   is the LeafCutter-equivalent parameterisation and a Dirichlet prior on psi would NOT be --
   LeafCutter does not place a prior on the composition, it constructs it deterministically.

The one structural difference is what generates the composition. LeafCutter uses a linear
predictor per pseudobulk sample; LeafletFA uses a low-rank mixture `phi_c @ psi` over K programs.
That difference is the method, and it does not touch the likelihood.

The one remaining statistical difference is the prior on the logits: LeafCutter reparameterises
`beta` through a simplex with a free scale, which is effectively weak, while we impose a hard
`Normal(0, 1)`. Measured, that prior is not flat -- at m=3 it implies a mean maximum usage of 0.60
against 0.33 for uniform, and at m=11 it implies 0.305 against 0.091. Whether that matters is
under test.

## The properties that make it coherent

**A mixture of compositions is a composition.** `sum_{j in e} mu[c,j] = sum_k phi[c,k] * 1 = 1`,
because psi sums to 1 within the event by construction and phi sums to 1 over programs. Verified
across 200 cells: sums 1.000000 throughout. This is what lets the low-rank structure sit inside a
Dirichlet-multinomial at all.

**A_e equals conc exactly**, which is what `a = conc * softmax(...)` requires. Measured maximum
deviation 7.6e-6, i.e. float32 noise. `check_A_e_is_conc()` reports it, and is the cheap runtime
diagnostic that the simplex is active.

**N_e is derived, never read.** A Dirichlet-multinomial is only defined when the count vector sums
to its total. That identity holds exactly on the mouse atlas but NOT on independent-Binomial
simulations, where junction counts sum to the stored ATSE total only in expectation (measured gap
up to 205 reads, 7.1% of cell-ATSE entries). `dm_log_prob` therefore computes
`N_e = sum_{j in e} y_j` rather than reading `cell_by_cluster_matrix`, which removes the
assumption and matches LeafCutter's own convention of summing a cluster's junctions.

## Identifiability

The DM depends on `alpha = conc * psi` and only on alpha, so rescaling psi within an event by c
while scaling conc by 1/c leaves the likelihood unchanged. Only the composition
`psi_j / sum_j psi_j` is identified; the raw array is not. Under the simplex this is moot, since
psi already sums to 1. Under unconstrained psi the raw array must be normalised on read-out --
`_identified_psi()` does this, and without it the DS statistic TV, which is computed from raw psi,
is rescaled arbitrarily per event. Measured: normalising moves psi recovery from r=0.926 to
r=0.9956 and within-ATSE validity from 17.5% to 100%.

## Cost

The DM is CHEAPER than the BetaBinomial it replaces. At atlas shape (B=2048, J=77,114, 24,475
ATSEs, 4 threads): DM 2,997 ms against BetaBinomial 14,005 ms, a ratio of 0.21. pyro's
BetaBinomial evaluates about five `lgamma` passes over `[B, J]` including the binomial
coefficient; the DM needs two, plus two `index_add` scatters that together cost 447 ms (15% of
its total). This holds only because `atse_idx` is a flat length-J integer array, so grouping is
one pass over a contiguous tensor -- no loop over events, no ragged tensors, no padding to
`max(m)`, which would have cost `[B, 24475, 11]` against the `[B, 77114]` already held. Pinned by
a regression test in `tests/test_dirichlet_multinomial.py`.

## What is omitted

The multinomial coefficient `log(N_e! / prod_j y_j!)` is dropped. It carries no parameter
dependence and therefore no gradient. It does mean the reported ELBO carries a constant offset
and cannot be compared numerically against a BetaBinomial run -- those are different models, so
that comparison was never meaningful.

---

# Concentration: why it is bounded, and how that was found (2026-08-06)

## The problem

The DM+simplex atlas fits fixed psi -- a program's junction usages sum to 1 within every ATSE,
100% of 24,475 events, against the released BetaBinomial model's 86.9% (whose sums range 0.000 to
4.426, so 13% of program-ATSE pairs are not compositions at all). But they broke phi.

Measured on the Fbln2 heart endothelial vs fibroblast contrast, the worked example in Figure 3:

| | psi sums valid | ||dphi||_1 | predicted skip PSI (endo/fibro) |
|---|---|---|---|
| released BetaBinomial | 86.9% | 1.44 | 0.83 / 0.08 |
| DM+simplex | 100% | **0.23** | 0.53 / 0.53 |
| LeafCutter (measured) | -- | -- | 0.92 / 0.02 |

Both cell types loaded on the SAME program at ~0.74, so their contrast had almost nothing to
differ by. Systematic across seeds and K: ||dphi||_1 was 0.23-0.47 for every DM+simplex fit against
1.25-1.44 for every BetaBinomial fit.

It was not confined to one ATSE. Scored under an identical DM likelihood with mu normalised within
ATSE (so the old model's unconstrained psi scale cannot flatter it), against a null of one shared
composition and a reference giving each cell its broad_cell_type's pooled composition:

| model | % of the cell-type reference |
|---|---|
| released BetaBinomial | 101.2% |
| DM+simplex | 88.9% |

## The cause

The trajectory tracking added for this investigation (`fit(track_every=N)`) shows concentration
collapsing from 0.975 to 0.518 by step 1,600 -- BEFORE any program structure has formed -- and never
moving again. 67% of events end below conc=1. Meanwhile psi_contrast peaks around step 4,000 and
flatlines for the remaining 12,000 steps.

A Dirichlet-multinomial with conc<1 says the composition is nearly unconstrained, so the likelihood
barely rewards a cell for loading on the RIGHT program. That is the pressure which spreads phi
across programs, and without it the model concentrates: top-3 programs hold 61% of all cell mass
against the released model's 42%, and cells commit to 2.6 programs rather than 3.4.

This is the documented "a free bb_conc is an early-optimisation escape valve" failure that freezing
bb_conc fixed at K=50, reappearing through the per-event parameterisation. The released model used a
single global concentration and it sat at 1.97.

## The fix, and its limits

`conc_min` bounds the per-ATSE concentration from below rather than freezing it, keeping the
per-event flexibility the DS test needs. Measured at K=20 seed 0, 8,000 steps:

| setting | psi valid | ||dphi||_1 | top-3 mass | % of cell-type reference |
|---|---|---|---|---|
| free (baseline) | 1.00 | 0.23 | 0.60 | 88.9% |
| conc_min 1.0 | 1.00 | 0.11 | 0.54 | 95.5% |
| **conc_min 2.0** | **1.00** | **1.40** | 0.51 | 97.9% |
| conc_min 5.0 | 1.00 | 1.32 | 0.43 | 96.1% |
| **global conc** | **1.00** | 0.98 | 0.51 | **101.3%** |

conc_min=1.0 is NOT enough; 2.0 works, which is where the released model's concentration
independently sat. Both conc_min>=2 and the global-conc arm restore compositional psi AND
discriminative phi.

**A caveat that must not be dropped.** conc_min=2.0 was selected by scoring the Fbln2 panel -- one
ATSE, chosen as the showcase example. Selecting a hyperparameter on the illustrative figure is
exactly the objection a referee should raise, and the likelihood column above is TRAINING
likelihood, which cannot detect overfitting. `--holdout-frac` was therefore added to the trainer and
the setting is being re-selected on 10% of cells held out of training, with a fixed split seed so
every arm is scored on identical cells. The global-conc arm is the cleaner outcome if it holds,
since it requires no threshold at all.

Note the tension this creates with the DS test: per-event concentration was introduced BECAUSE one
global value cannot span the observed 0.012-4,750, and it is what makes the DS null robust without
truth labels (effect-size drift 5.9x -> 1.3x). The DS test estimates concentration per event
internally regardless of how the model was trained, so a global-conc MODEL and a per-event DS TEST
are compatible -- but that should be stated rather than assumed.
