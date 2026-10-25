# LeafletFA — the current model (single reference description)

**This is the current description of LeafletFA.** It supersedes `METHODS.md`, `METHODS_MINIBATCH.md`
and `METHODS_dirichlet_multinomial.md` wherever they differ; those files are kept as history. Manuscript
Methods should be written from this file. Written 2026-10-25. Code: `leafletfa/minibatch.py`
(`LeafletFAMinibatch`), `leafletfa/dirichlet_multinomial.py`, `leafletfa/estep.py`, `leafletfa/train.py`.

## 1. Data

- Input per cell c and ATSE (alternative transcript structure event) e: the junction read counts
  y_c^(e) = (y_cj), j ∈ e. Every ATSE has ≥ 2 junctions.
- The ATSE total is N_ce = Σ_{j∈e} y_cj. It is derived from the junction counts, never read from the
  stored `cell_by_cluster_matrix`.
- An ATSE with no reads in a cell contributes nothing to the likelihood.
- AnnData / MuData layer `cell_by_junction_matrix` (cells × junctions); `var.event_id` gives each
  junction's ATSE.

## 2. Generative model (the shipped "DM simplex" model)

```
alpha_pi        ~ Gamma(1, 1)
pi              ~ Dirichlet(alpha_pi / K, ..., alpha_pi / K)        program proportions
delta           = K   (fixed)                                      Dirichlet concentration on phi
phi_c           ~ Dirichlet(delta * pi)                            cell c's program loadings
eta_kj          ~ Normal(0, 1)                                     logits, program k, junction j
psi_k,(e)       = softmax(eta_k,(e))  within each ATSE e           program k's composition of e
                  (clipped to [1e-4, 1 - 1e-4])
gamma_e         ~ Gamma(2, 0.1), floored at 2 (clamped to [2, 1e6])  per-ATSE concentration
rho_cj          = sum_k phi_ck psi_kj                              cell c's expected usage
y_c^(e) | N_ce  ~ DirichletMultinomial(N_ce, gamma_e * rho_c^(e))
```

- **Simplex.** Σ_{j∈e} ψ_kj = 1 for every program and ATSE, so Σ_{j∈e} ρ_cj = 1 and the Dirichlet's
  total pseudo-count equals γ_e.
- **Likelihood terms.** The DM is evaluated as lgamma terms, with the α = γρ floor at 1e-2
  (`dm_alpha_floor`). The multinomial coefficient is omitted in training: it is constant in the
  parameters.
- **m = 2.** At m = 2 this is the Beta-binomial; for larger m it is the exact multivariate
  generalisation, the same likelihood as LeafCutter's DM GLM.
- **Concentration floor.** γ_e is the effective number of independent molecules per cell and event. It is
  floored at 2, the measured 1.7–3.1 molecules behind a Smart-seq2 read pile (no UMIs; reads are PCR
  copies).
  - The floor binds for most events: 75 % of the 24,475 per-ATSE γ_e of the atlas K20 fit sit exactly
    at 2. For those events the overdispersion is set by the floor, not estimated.
- **Pinned δ.** δ is pinned at K. Learned δ fell to δπ_k ≈ 0.19–0.37 and collapsed the loadings.
  - Even pinned, the fitted π leaves most δπ_k < 1: 17 of 20 on the TMS70 K20 fit, minimum 0.06.
  - That prior is unbounded at the simplex edge, so cells are pushed toward a few programs; see §6
    and §8.
- **Code defaults.** `LeafletFAMinibatch.dm()` sets likelihood "dirichletmultinomial",
  psi_parameterization "simplex", conc_per_atse True and conc_min 2. It does **not** pin δ: the
  training drivers do (`train.py`, A44 `--dir-conc K`, `run_leafletfa_baseline.py`), via
  `model.fixed_dir_conc = K`.

## 3. Inference

- **Guide.** Stochastic variational inference in Pyro with an `AutoNormal` guide: mean-field Gaussian
  over the unconstrained parameters.
  - φ uses a full-size (cells × K−1) location and scale, indexed by one subsampling plate.
  - Initialisation is `init_to_feasible`: uniform compositions, φ = 1/K.
- **Minibatching.** Each step draws a minibatch of cells; the local ELBO terms are scaled by N/B.
- **Optimiser.** `ClippedAdam` with gradient-norm clip 10; lr decays exponentially to γ·lr over the run.
- **Non-finite guard.** A non-finite loss restores the best (lowest-EMA) snapshot and clears Adam's
  state, halving the lr only if the failures recur within 100 steps; the run stops after
  `max_nan_restarts`. With float64 φ (§5) this guard no longer fires in practice.
- **Convergence.** Judged on the ψ contrast trace (between-program spread), not the ELBO, which is
  noise-dominated at this scale.
  - Note: training logs for simplex models print `psi_contrast`. The older `psi_mean` is always
    n_ATSE / J (0.3068 on the atlas) and cannot show learning.

## 4. Training recipe (shipped)

| setting | value |
|---|---|
| K | 20 for all reported analyses; fits at K ∈ {10, 20, 30} × 3 seeds as robustness checks |
| steps × minibatch | 16,200 × 8,192 cells, no early stopping |
| learning rate | 0.025, exponential decay to 0.1× (γ = 0.1) |
| K = 30 on datasets < 60,000 cells | lr 0.015 (TMS-only, mouse Allen-only, human Allen fits); 0.025 diverged repeatedly |
| simulations (6,986 cells) | 3,000 steps, minibatch 2,048, lr 0.05 → 0.5× |
| δ | pinned at K |
| γ_e floor | 2 |
| φ precision | float64 (recommended; §5) |

- **Hardware.** Fits run on GPU: 0.05 s/step for TMS and 0.14 s/step for mouse Allen on an RTX PRO 6000
  (b6k), i.e. 15–40 min per fit with the sparse likelihood.
- **CPU is ~10× slower** (6 s/step for TMS on 16 cores); the step is bound by gathers and scatters.
- **Driver:** `python -m leafletfa.train` or `scripts/splicevi_baseline/run_leafletfa_baseline.py`
  (resumable, writes a run record with input md5s).

### Choice of K (validation 2026-10-25, TMS; `LeafletFA_validation/VALIDATION_REPORT.md`)

| | K 5 | K 10 | **K 20** | K 30 |
|---|---|---|---|---|
| kNN, held-out cells | 0.393 | 0.479 | **0.476** | 0.429 |
| programs in use | 5 | 8 | 7 | 7 |

- K = 20 performed as well as K = 10 and better than K = 30.
- Higher K adds unused programs, not information.

## 5. Options added 2026-10 (all opt-in; defaults reproduce the published fits bit-for-bit)

| option | default | recommended | what it does |
|---|---|---|---|
| `phi_dtype` | `"float32"` | `"float64"` | φ site and its guide in float64 (likelihood stays float32). Removes the non-finite restarts (§8). No measurable speed cost. |
| `sparse` | `False` | `True` | DM evaluated on observed events only (`SparseObs`, `dm_log_prob_sparse`); equal to the dense likelihood up to summation order (tests/test_sparse_dm.py); no dense cells × junctions matrices. On GPU 3.4× faster for TMS, 1.2× for Allen (denser); GPU memory 4 GB (TMS) / 21 GB (Allen) vs 34–37 GB dense. |
| `fit(resume_state=…)` | `None` | — | resume from a mid-run snapshot (checkpoint/resubmission). |
| `dir_alpha_min` | `None` | not recommended | floors the φ prior's Dirichlet parameters at the given value (validation only). At 1 it raised kNN but made the representation read more from detection (content-only drop 0.19 vs 0.23). |

Published / Genome Biology fits used the defaults: float32 φ, dense likelihood. Results with the new
options are equivalent, not bit-identical: float64 draws a different random stream.

## 6. Applying a fitted model to new cells (E-step)

- **Frozen globals:** ψ (`psi_mean`), π, α_π, δ = K, and the per-ATSE concentration γ_e.
  - All of them are frozen, not only ψ: π and δ form the prior on φ, and re-learning them on new cells
    would infer φ under a different prior.
- **Inferred:** φ, by `LeafletFAMinibatch.freeze_globals(...)` then `infer_phi(...)`, or
  `leafletfa.estep.infer_phi_minibatch`.
- **Settings:** 3,000 steps, minibatch 8,192, lr 0.1, γ 0.1, float64 φ, sparse likelihood.
- **Output:** the posterior median of φ, renormalised to the simplex.
- **Downstream embedding:** centred log φ (clipped at 1e-6). Embeddings that compare models should use
  the programs in use (mean training φ > 0.01): unused programs carry near-zero loadings that a log
  transform magnifies.

## 7. What the model is validated to do (2026-10-25 validation, pre-registered)

- **Held-out fit.** On held-out cells it beats fair nulls. Each test cell's events are split in half:
  φ is inferred on one half and the other half is scored.
  - It scores higher per read than a population ψ, a per-cell-type ψ and a K = 1 model, all fitted on
    training cells (TMS and mouse Allen; all 95 % CIs > 0).
  - The earlier "held-out below null" number (A44) scored one average φ for every cell and is not a
    test of the model.
- **Per-cell representation.** It is stable across seeds (kNN range ≤ 0.025) and reads splicing ratios:
  kNN falls by ~0.2 when ratios are redrawn with detection kept.
- **Simulation recovery.**
  - At **TMS depth**, data simulated from a fitted model are recovered: ψ r 0.98, 6 of 7 used programs,
    per-cell φ r 0.98.
  - Recovery **fails at EasySci-like depth** (counts thinned to 1.1 %): ψ centred cosine 0.34, 0 of 7
    programs recovered, kNN 0.20 vs 0.50 on the true φ.
  - Do not use LeafletFA programs fitted on data that sparse.
- **Individual programs are mostly not reproducible across seeds** (§8). Program-level claims are
  restricted to reproduced programs.

## 8. Known properties and limitations

- **Program reproducibility.**
  - **Rule.** Match programs across seeds by Hungarian assignment on the cosine of ψ centred by its
    π-weighted mean. A program is reproduced if it is used by a cell type and matches at ≥ 0.8 in both
    other seeds.
    - Do not use uncentred ψ: it is ~0.98 similar for every program, so "r ≥ 0.9" passes nearly
      all of them.
  - **Counts.**
    - Atlas K20: 4 programs are reproduced (excitatory and inhibitory neuron programs,
      keratinocyte/basal epithelium).
    - TMS: 3. Mouse Allen: 1–2.
    - Lower or higher K does not increase the count.
  - **Consensus.** The consensus ψ of a reproduced program is its members' mean across seeds,
    renormalised within each ATSE.
- **Few programs are used.** 5–8 programs are used regardless of K, because most δπ_k < 1.
- **The concentration floor dominates.** 75 % of γ_e sit at the floor of 2 (§2).
- **Depth.** See §7: programs are not recoverable at EasySci-like depth.

### The non-finite restarts, and the earlier "φ-corner mechanism is refuted" finding

- **The restarts.** Every float32 fit and E-step hit occasional non-finite losses: 1–17 restarts per
  production fit, and repeated restarts in the SpliceVI §170 E-step.
- **Cause.**
  - With δπ_k < 1, q(φ) moves cells toward the simplex edge.
  - In float32 the stick-breaking transform then rounds a component of φ to exactly 0.
  - log 0 enters both the Dirichlet log-density at the φ site and the transform's Jacobian, and the
    ELBO is inf − inf = NaN.
- **Fix.** float64 φ removes it: even 19 saturated sticks (~1e-297) stay above the float64 floor. Zero
  restarts in every fit and projection since.
- **The two findings are compatible.** `ATLAS_TRAINING_DECISIONS.md` §9 reports flooring φ at 1e-4 left
  the divergence step byte-identical and calls the φ-corner mechanism refuted. That ablation floored φ
  *after* the sample, where it enters the likelihood (`phi_floor`). It did not touch the Dirichlet
  log-density or the Jacobian of the sampled value, which is where the NaN arises. So it refuted a
  corner effect *through the likelihood gradient*, not the corner mechanism itself.
- **The learning-rate effect.** Lowering the lr delays the edge drift without removing it: production
  fits at lr 0.025 still had 1–17 restarts.

## 9. Superseded

- **Beta-binomial model.** Independent per-junction Beta ψ with a BetaBinomial likelihood:
  `likelihood="betabinomial"`, `psi_parameterization="beta"`, and the legacy `LeafletFA` class in
  `leafletfa/model.py`.
  - It treats an ATSE's m junctions as m trials against a shared total and over-counts the evidence by
    m/(m−1).
  - It remains available as an option. No reported result uses it.
- **Learned δ** (`dir_conc ~ Gamma(1, 1)`), as written in `METHODS_dirichlet_multinomial.md`: replaced by
  δ = K.
- **Older descriptions.** Waypoint initialisation, multi-pass chunked training and the factor-specific
  junction DS test (`METHODS.md`, `METHODS_MINIBATCH.md`) are not part of the shipped model.
