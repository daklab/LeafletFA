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
