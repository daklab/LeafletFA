# LeafletFA — source review, manuscript alignment, and fixes

> Review dated 2026-07-11. Code findings were audited against `leafletfa/` **and** against the real
> training pipeline in `Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/`.

## STATUS  (last updated 2026-07-11, end of session)

### DONE — `leafletfa/` package

| § | Fix | Where |
|---|---|---|
| 1 | **CPU gradient bug.** `torch.no_grad()` in the sparse dot detached psi/assign from the likelihood, so they were trained by the prior alone and the model returned its own initialization. | `model.py` `predict_rho` |
| — | **Triton dense mask deleted.** It allocated a dense C×J mask (**47.6 GB** at atlas scale, twice over), **never read the mask** (bounds-masking only), and ran `tl.dot` in **TF32** (**7.7e-4** error vs float64). Replaced by a sparse gather on both devices: **1.5 GB**, **9e-8** error, no Triton. **This is why the atlas "didn't fit".** `initialize_triton_mask()` is a no-op kept for script compatibility. | `model.py` `predict_rho` |
| — | **Likelihood rewritten in `lgamma`.** `dist.BetaBinomial.log_prob` built ~15–20 nnz-sized intermediates that autograd retained **per ELBO particle** — this is what OOM'd/crashed on the RNA-atlas run. `log(n choose k)` is constant in the parameters and is now cached **once as a scalar** instead of rebuilt as an nnz tensor every step; `lgamma(alpha+beta)` collapses to `lgamma(conc)`. Numerically identical to Pyro (rel. diff **0.00e+00**), **~2.4× faster**, 9 full-size lgammas → 5. | `model.py` `log_prob_calc_sparse` |
| 2 | **`obs_scale`** — rescales the likelihood **and** the `assign` prior by N/batch, global KL stays at weight 1. Defaults to 1.0. | `model.py` |
| 2b | **`get_pi` no longer mutates `self.dir_conc`** — that mutation is what silently froze δ at the first chunk's estimate. Learned value → `self.dir_conc_post`. | `model.py` `get_pi` |
| 7 | **`pi_fixed` / `alpha_pi_fixed`** — genuinely freezes pi (`pi_init` only *initialises* a still-learned pi). Plus a hard error if `fixed_psi`'s junction axis does not match the target AnnData. | `model.py` |
| 10 | **`prune_K`** no longer indexes the per-junction `a`/`b` (shape `[J]`) with **factor** indices. | `model.py` `prune_K` |
| 10 | **`from_anndata`** unbound-local `device` when `float_type` is passed. | `model.py` |
| — | **`clip_norm` decoupled from `enable_cpu_optimization`.** Gradient clipping was ON only when that flag was True — so CPU runs were clipped and **GPU runs (which pass it False) were not**. Now an explicit `clip_norm=10.0` on every device. | `model.py` `fit` |
| — | **Index caching + coalescing on every device.** Was gated on `enable_cpu_optimization and cpu`, so **GPU runs re-extracted indices and ran `torch.equal` over two 2×nnz tensors on every SVI step.** Now done once in `from_anndata`. | `model.py` |
| — | **Progress bar shown on all devices** (was suppressed exactly when `enable_cpu_optimization=False`, i.e. on GPU). | `model.py` `fit` |
| 5 | **`stratified_max_min_sampling` removed** — cell-type-label-driven waypoints conflict with the "without prior cell-type labels" claim. | `waypoints.py` |
| — | Regression tests: gradient reaches psi+assign (sparse **and** dense); psi moves off its init and factors don't collapse; `obs_scale` scales only local terms; `pi_fixed` really is fixed; junction-axis mismatch raises. **15/15 pass.** | `tests/test_basic.py` |

### DONE — `Leaflet-analysis`

- **`run_training.py`** — passes `obs_scale = n_cells/len(batch)`; reads `dir_conc_post`; **stopped feeding the
  learned δ back in as `delta_fixed`** (which had frozen δ at chunk 1 for the remaining ~185 chunks);
  unstratified `max_min_sampling`; dead `initialize_triton_mask()` call removed.
- **`07c_classify_heldout_phi.py`** — `pi_init=` → **`pi_fixed=`** (pi is the prior on phi; re-learning it on
  the test cells confounded the train−heldout memorization gap), and the E-step now uses the **trained
  `bb_conc`** instead of forcing `input_conc_prior=inf` (Binomial where training was Beta-Binomial made
  held-out phi over-confident and flattered the decoding number).
- **`08_compare_batch_equivalence.py`** — added the **β slope** and `psis_scale` comparison. Its existing
  metrics (Hungarian Pearson r, cosine, argmax) are **all scale-invariant** and cannot detect effect-size
  attenuation: on a run deliberately shrunk 3×, they still report *"programs agree, r = 0.886"*, while the
  new slope correctly reports **0.333 → "attenuates effect sizes by 3.0×"**.
- **`rna_atlas_explore.ipynb`** — `MAX_JUNCTIONS = None` was silently selecting **zero** junctions
  (`series <= None` is all-False in pandas); `None` now means "keep all". Dead triton call removed; save cell
  now reads `dir_conc_post`.

### ENVIRONMENT (not a code bug — cost a lot of debugging time)

`LeafletSC` had **torch 2.4.1**, which has no kernels for the **RTX PRO 6000 Blackwell (sm_120)**. Every
`torch.lgamma` on CUDA failed with a kernel JIT error. Fixed by upgrading to **torch 2.11.0+cu128** (arch list
now includes sm_120); scanpy/anndata/pyro all still work.

---

## REMAINING

### Experiments / decisions (block the paper)

- **§2 — run the N/B equivalence test.** `obs_scale` is now wired in, but **we still do not know whether it
  changes anything at K=20 / J=79k with waypoint init.** Full-batch vs chunked on the same cells; report the
  **β slope** (see Appendix). This decides whether §2 is a footnote or a refit of Figs. 4E/5A.
- **§2 — the guide's `q(psi)` scale is still re-initialised every chunk.** `obs_scale` fixes the *objective*,
  so the scale is now aiming at the right target, but `run_training.py` still builds a fresh guide per chunk
  and warm-starts only psi's **mean** — `global_psi_scale` is extracted and never fed back. Whether 20 epochs
  from a cold start reaches the right width is an empirical question. Belt-and-braces fix = carry the guide
  scale forward (needs `fit()`/`train()` refactor).
- **§1b — date the 2025 training runs.** `f0aca06` (9 Jun 2025) → `09c1505` (14 Oct 2025) shipped a
  **misspecified Beta-Binomial** (`alpha = success_probs`, missing the concentration) **on all devices**.
  Anything carried over from the 2025 submission is suspect.

### Code — not yet fixed

- **§3 DS.** `compute_psi_effect_size` still ignores Φ (Methods define a φ̄-weighted ρ). `num_samples`
  default is still **10** (sweep used 100, Methods say 500). The two FDR functions still disagree with each
  other and neither implements max-detections. `analyze_all_junctions_groups` still **missing**.
- **§4 waypoints.** `initialize_psi_phi_with_random_non_zero` still takes `np.random.choice(non_zero_vals)` —
  **one random cell's PSI**, where the Methods says *"average across the 50 nearest neighbours (to reduce
  noise)"*. `assign_nearest_cells` still defaults to `num_nearest=5`. ψ_init still isn't clipped to
  `[ε, 1−ε]`, so an exact 1.0 saturates the logit and freezes that entry. **No package code writes the
  `psi_init_{K}_waypoints` keys** — the glue lives only in `run_training.py`, so `waypoints_use=True` (the
  default!) makes the documented quickstart raise.
- **§6 early stopping.** Still `loss < best_loss - min_delta` with an **absolute** `min_delta` against an ELBO
  of 1e6–1e8 — i.e. a relative tolerance of ~1e-9, numerically inert. Rule degenerates to "stop after
  `patience` epochs without a new global best". Needs a relative + smoothed criterion, and ideally a held-out
  convergence metric.
- **§8 priors.** Code still has `bb_conc ~ Gamma(10,10)` (Methods text says Gamma(2,2), Methods table says
  Gamma(1,1)) and `dir_conc ~ Gamma(2,2)` (Methods says Gamma(1,1)). The **extra learned hierarchical level**
  (`a_shape`, `a_rate`, `b_shape`, `b_rate`) is still absent from the Methods entirely.
- **§10 dead/broken code.** `utils.compute_and_plot_albf` still calls `differential_splicing.compute_albf`,
  which **does not exist**; `EMDifferentialSplicing` still consumes `albf_scores` nothing produces;
  `utils.calculate_and_plot_correlations` still imports `cost_correlation_assign`, absent from the repo;
  all of `leafletfa/analysis_utils/` is still unreferenced.
- **§10 docs.** Still tell users to read `adata.varm["psi_learned"]` / `adata.obsm["X_PHI"]` — **the package
  writes neither**. `docs/outputs/differential.md` still calls `test_differential_activity` (does not exist).
  `num_initializations` still documented as a constructor arg (it's a `train()` arg). README quickstart still
  raises (`waypoints_use` defaults True). **This is Reviewer 2's complaint #4.**

### Manuscript — not yet written

Perplexity (Fig. 3C/D — Results says "see Methods"; Methods never defines it), PVE (Fig. 3A), `prune_K`, the
transfer/E-step procedure, and the Notion-listed text fixes (restore the `s` index in the β equation; correct
the FDR rule to max-detections; soften "ensures at most 5% false positives").

---

## Context

Full audit of `leafletfa/` against the manuscript Methods **and** against the real training pipeline
(`Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/run_training.py`), which turned out
to matter enormously: several things that look like paper-invalidating bugs in the package are **not**,
because the pipeline explicitly opts out of the broken code path. Conversely, the pipeline has a
statistical problem the package does not.

Headline:

- **One critical bug makes CPU training a silent no-op for Ψ and Φ.** It does **not** affect the paper
  (`run_training.py` passes `enable_cpu_optimization=False` and runs on CUDA), but it will silently
  destroy results for every user who pip-installs this and trains on CPU with defaults.
- **The chunked multi-pass training never rescales the likelihood by N/batch.** This is a real
  statistical issue that *does* affect the paper: the Ψ posterior width reflects 5,000 cells, not 92,894 —
  which is exactly the posterior the DS/FDR analysis samples from.
- **Several Methods↔code mismatches**, most notably the waypoint ψ initializer, which does the *opposite*
  of what the Methods claims, and `num_samples` (100 actual vs 500 stated).
- `stratified_max_min_sampling` **is** used by the current sweep (Notion confirms: "Includes the phi
  carry-over fix and stratified-by-cell-type waypoints"), contrary to the initial assumption — see §5.

---

## 1. CRITICAL (package, not paper) — CPU training silently learns nothing for Ψ and Φ

[leafletfa/model.py:337-347](leafletfa/model.py#L337-L347), `sparse_dot_cpu`:

```python
if self.enable_cpu_optimization:
    with torch.no_grad():                                    # <-- detaches assign and psi
        assign_rows = assign.index_select(0, cell_idx.long())
        psi_selected = psi.index_select(1, junc_idx.long())
    pred = torch.einsum('nk,kn->n', assign_rows, psi_selected)
    pred.requires_grad_(True)                                # <-- pred becomes a NEW leaf
    return pred
```

`no_grad` severs the graph; `requires_grad_(True)` makes `pred` a fresh leaf. The likelihood gradient
terminates at `pred` and **never reaches `assign` or `psi`**. They still drift under the prior/entropy
terms, so the model **returns a shrunk version of its own initialization** — and under waypoint init that
initialization already encodes cell-type structure, so the output looks biologically plausible.

Verified two ways: gradients w.r.t. `assign`/`psi` are `None` (the non-optimized branch gives finite
gradients); and on simulated data with the same seed, forced sparse path, the optimized path converges to a
**worse ELBO (121,635 vs 119,000)** — a 2,635-nat gap that is exactly the likelihood signal it cannot use.

**Blast radius — the bug requires all three of these to be true simultaneously:**
1. `self.device.type == 'cpu'` — `sparse_dot_cpu` is called *only* from the CPU branch at
   [model.py:467-468](leafletfa/model.py#L467-L468). The GPU branch
   ([:478-487](leafletfa/model.py#L478-L487)) goes through `compute_pred_triton` → `masked_matmul`, whose
   custom backward (`grad_A = grad_out @ B.T`, `grad_B = A.T @ grad_out`) is the correct gradient of `A @ B`.
   **On CUDA this code is unreachable, regardless of the `enable_cpu_optimization` flag.**
2. `enable_cpu_optimization=True` — the **default**.
3. Sparse mode — `detect_data_density` routes to the dense (correct) path above 30% density. Real splicing
   data is far below 30%. **The tiny test fixtures sit near that boundary and take the safe path; real data
   takes the broken one.**

So: **strictly a CPU bug**, and on CPU it is the default configuration. `tests/test_basic.py` asserts only
shapes and absence of NaN, so an untrained model passes.

**The paper is safe:** `run_training.py:337,375` passes `enable_cpu_optimization=False` *and* the run is on
CUDA — either alone would have been sufficient.

**Fix** — drop the `no_grad` wrapper and the `requires_grad_` call; keep `index_select`:

```python
assign_rows = assign.index_select(0, cell_idx.long())
psi_selected = psi.index_select(1, junc_idx.long())
return torch.einsum('nk,kn->n', assign_rows, psi_selected)
```

Introduced by `09c1505` "update oct 25" (14 Oct 2025), which added the flag and the `no_grad` block
together. The CPU path was **correct before that commit** (that code survives as the now-unreachable
`else:` branch). The same commit added a cosmetic `with torch.enable_grad():` around `svi.step`, which
cannot help — the detachment is *inside* the model.

**Regression tests to add** (a shapes-only test cannot catch this class of bug):
- Gradient-flow test: sparse mode + `enable_cpu_optimization=True`, one `svi.step`, assert the Ψ and Φ
  guide `loc` gradients are finite and non-zero.
- Factor-recovery test: simulate from the generative model (`φ~Dir`, `ψ~Beta`, `y~Binomial(T, φψ)`), fit,
  assert best-match `corr(ψ̂, ψ_true)` clears chance.
- Parity test: `enable_cpu_optimization` True vs False reach the same ELBO within MC noise.

---

## 2. HIGH (affects the paper) — the chunked training never rescales the likelihood by N/batch

`run_training.py` implements the Methods' multi-pass scheme by **re-instantiating a fresh `LeafletFA` on
each 5,000-cell chunk** and warm-starting Ψ/π from the previous chunk. That is a legitimate design
(streaming / incremental VI), and the ELBO-per-cell and π-evolution logging is genuinely good. But it is
missing the one thing that makes minibatch VI correct.

**Note this is NOT the same thing as transfer.** In transfer the globals are *frozen*; here they are
**updated** every chunk (warm-started from the previous chunk). The only thing the two share is that each
chunk *initializes* from the previous chunk's globals. The distinction matters, because it is precisely the
"updated using only B cells' likelihood" part that goes wrong.

**The problem — stated exactly.** A correct minibatch SVI objective is an unbiased estimate of the *full*
ELBO, which requires scaling the likelihood **and the local terms** by `N/B` while leaving the **global** KL
at weight 1:

```
L̂_b = (N/B)·[ E log p(y_b | φ_b, Ψ) + E log p(φ_b | π,δ) − E log q(φ_b) ]      # local, scaled
      +        [ E log p(Ψ) − E log q(Ψ) + E log p(π) + E log p(a,b) + … ]        # global, weight 1
```

What `run_training.py` optimizes per chunk is **everything at weight 1** — i.e. literally the ELBO of a model
that contains only 5,000 cells. Its stationary point is therefore `q(Ψ) ≈ posterior over Ψ given 5,000 cells`,
not given 92,894.

A second, compounding problem: a **fresh `LeafletFA` (hence a fresh guide) is built on every chunk**, warm-
starting only Ψ's *mean* (plus π/α). `global_psi_loc`/`global_psi_scale` are extracted at
[run_training.py:403-405](../Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/run_training.py#L403-L405)
and **never fed back in**, so every variational *scale* and the `a`/`b`/`conc` locations are re-initialized
from scratch ~186 times, each with only 20–50 epochs to recover.

### What a simulation shows — and what it does NOT

Simulated from the generative model with known Ψ (N=4,000, K=4, J=90, B=250 → N/B=16, 6 passes, **random
init**). "Factor contrast" = `psi.std(dim=0).mean()`, the spread of ψ across factors at each junction —
**this is exactly the DS effect size β**:

```
TRUE factor contrast                                            0.3336
1. full-batch (reference)            q(ψ) width=0.2399   contrast=0.3372   corr w/ truth=1.000
2. chunked, NO N/B scaling           q(ψ) width=0.1840   contrast=0.0825   corr w/ truth=0.704
3. chunked, WITH N/B scaling         q(ψ) width=0.0886   contrast=0.1134   corr w/ truth=0.786
4. chunked + N/B + persistent guide  q(ψ) width=0.2213   contrast=0.0195   corr w/ truth=0.539
```

**Robust:** full-batch recovers the truth exactly; **every chunked variant attenuates β by 3–17×**. The `N/B`
scaling is a real improvement (contrast +37%, recovery 0.70→0.79, posterior sharpens 2.1×) but **is not a
rescue on its own**.

**NOT established — do not act on these magnitudes:**
- Condition 4 was expected to *close* the gap and instead **collapsed** (all factors converged to nearly the
  same ψ). It is **confounded** — it differs from the pipeline in three ways at once (no warm-start, no LR
  decay, persistent optimizer state). Plausibly the per-chunk teardown is *accidentally protective*: re-seeding
  ψ from the previous chunk re-anchors the factors apart and prevents exactly this collapse.
- The sim uses **random init**; the pipeline uses **waypoint init**, which is precisely the anti-collapse
  mechanism the sim lacks (§9). Real runs may be far better protected. The toy is K=4/J=90; the model is
  K=20/J=79,768. **None of these numbers transfer.**

### Why the December full-vs-batch check came back clean — and why that is not reassurance

The 2025 training script
(`LeafletFA_Submission2025/Mouse_Splicing_Foundation/model_train/MOUSE_FOUNDATION/full_workflow/02_run_leaflet.py`)
is the direct ancestor of `run_training.py`: same multi-pass loop, `BATCH_SIZE=5000`, `NUM_PASSES=10`,
50/20 epochs, and **also no N/B scaling**. So the issue, whatever it is worth, was equally present in December.

**But `08_compare_batch_equivalence.py` cannot detect it.** Its metrics are Hungarian-matched **Pearson r** on
ψ, per-cell **cosine** on φ, and **argmax agreement** — every one of them **scale-invariant**. If minibatch
shrinks all K factors toward a common per-junction mean, each program's profile *across junctions* is
preserved (r ≈ 0.99, cosine ≈ 1, argmax unchanged) while the factor-to-factor **contrast — i.e. β — collapses**.
"Programs agree" is exactly what you would see under 4× attenuation. **The December check was structurally
incapable of detecting this**, so it is not evidence of safety (just as the toy above is not evidence of harm
at real K/J with waypoint init).

### The one number that settles it — do this before anything else in §2

Add a **scale-sensitive** metric to `08_compare_batch_equivalence.py`: compute β for both runs, then **regress
β_batch on β_full and report the SLOPE, not the correlation.**

- **slope ≈ 1** → no attenuation; minibatch is fine; question closed permanently.
- **slope ≪ 1** → effect sizes are being shrunk, and Figs. 4E / 5A need a refit.

Also report `psi.std(axis=0).mean()` per run, the `psis_scale` magnitude, and the number of significant DS
junctions. ~20 lines on a script that already exists. Run it against the **current** code (post-π-fix,
post-cluster-count-fix), on a subset that fits full-batch in GPU memory (~15–20k cells), with the **same
waypoint init** for both arms.

### What changed between the December (2025) and current (2026) training code

| | 2025 (`02_run_leaflet.py`) | 2026 (`run_training.py`) |
|---|---|---|
| Multi-pass minibatch, B=5000, 10 passes, 50/20 epochs | ✅ | ✅ same |
| **N/B likelihood scaling** | ❌ absent | ❌ **still absent** |
| **π / α carried across chunks** | passed, but **silently discarded** — `AutoDiagonalNormal` ignores `infer={"initial_value":…}`; every chunk's guide started **near-uniform** | ✅ **fixed in `4b2a714`** (17 Jun 2026) via `init_loc_fn` |
| Waypoints | unstratified `max_min_sampling`, computed **per batch** | **cell-type-stratified**, computed on the **full dataset** (§5) |
| Cluster-count extraction | buggy indexing | ✅ fixed in `2431d13` — **changes the likelihood's inputs**, so December's models were fit to subtly different data |

So December's models differ from today's on **three** axes beyond the N/B question. Any comparison against them
is confounded; re-run the equivalence check on current code.

### The fix, regardless

Add an `obs_scale` and apply it to the likelihood **and the local `assign` terms** (not the global sites) —
scaling only the likelihood is a common half-fix and is still wrong:

```python
with poutine.scale(scale=self.obs_scale):        # obs_scale = N_total / batch_size, default 1.0
    assign = pyro.sample("assign", dist.Dirichlet(pi * dir_conc).expand([N]).to_event(1))
    ...
    pyro.factor("obs", log_prob)
```

The deleted `LeafletFAminibatch.py` did this correctly with a real plate —
`git show 6f9ce6f:src/BetaDirichletFactor/LeafletFAminibatch.py`. Resurrecting that is the principled option
and sidesteps the teardown/warm-start question entirely, since a plate keeps one guide alive by construction.

**Recency bias** is a second-order worry: the final Ψ is the endpoint of a sequential chain rather than a
symmetric average. Multi-pass mitigates it; `psi_mac_pass_to_pass` (already logged) is the diagnostic — if it
has plateaued, this is not live.

### 2b. `δ` is accidentally frozen after the first 5,000 cells

`get_pi` ([model.py:964-965](leafletfa/model.py#L964-L965)) **mutates** `self.dir_conc` from `None` to the
learned posterior mean. `run_training.py:396-400` then reads it into `global_dir_conc`, and
[:362](../Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/run_training.py#L362) passes
it back as `delta_fixed=`. From chunk 2 onward the model takes the *fixed*-δ branch and stops sampling it.
**δ is therefore learned on the first 5,000 cells and frozen for the remaining ~185 chunks.** That was not a
design decision — it is the `get_pi` mutation bug leaking into the pipeline. Store the posterior as
`self.dir_conc_post` and leave `self.dir_conc` as the user's input.

(`bb_conc` has the opposite problem: `global_bb_conc` is captured but never fed back, so it is re-learned
from scratch on every chunk. Pick one convention.)

### 2c. Also worth knowing

- `leaflet_model.best_elbo = np.mean([r['elbo'] ...])` ([:577](../Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/run_training.py#L577))
  averages **raw, unnormalized** ELBOs over *all* chunks and *all* passes, including the bad early ones. It
  is not a meaningful model-selection number. Use the **last pass's mean ELBO/cell**.
- `patience=5` against `num_epochs_later=20` means early stopping can kill a chunk after ~6 epochs. See §6.
- The package cannot reproduce the paper on its own: multi-pass training lives only in the pipeline script.
  Either ship a `fit_chunked()` in `leafletfa`, or state plainly in the README that atlas-scale training is
  driven by the analysis repo.

---

## 3. Differential splicing — reconcile with the Methods (and with the reviewers)

Your Notion notes already settle the direction: **group-vs-group (cell type) DS becomes the primary marker
analysis; factor-level DS is reframed as program characterization.** Given that, the fixes are:

**Factor-level DS (`compute_psi_effect_size`) — fix the Methods, not the code.** The Methods define
`ρ_{s,kj} = (1/C) Σ_c φ_{s,ck} ψ_{s,kj}` (a φ̄-weighted difference, shrinking low-usage factors), but
[differential_splicing.py:61-111](leafletfa/differential_splicing.py#L61-L111) computes the **unweighted**
`ψ_{s,kj} − mean_{k'≠k} ψ_{s,k'j}` and says so in its own docstring. Since this analysis is being demoted to
"program characterization", the unweighted form is the right one to keep (it is the natural "used more in
this program than others", independent of π) — **rewrite the equation to match the code.** Reviewer 1
separately objects that "difference versus averaged other programs" oversimplifies when a junction is
shared across programs; the Fig. 5A pleiotropy histogram is the honest answer and should be cited there.

**Also fold in the Methods-text fixes you already listed in Notion:**
- restore the missing `s` index in the β equation (R4);
- correct the FDR rule to "most permissive threshold under α / max detections", not "largest τ";
- soften "ensures at most 5% false positives" → "controls the posterior expected FDP".

**`num_samples`: Methods say 500, the sweep used 100** (`01_generate_params.py:71` → `"num_samples": [100]`;
the package default is `10`). 100 is defensible (P_kj resolves to 0.01), **but the Methods must say 100**,
and the package default of 10 should be raised — at S=10 the FDP estimator reports 0 for any junction where
all 10 samples clear δ, which is an artifact, not evidence.

**The two FDR functions disagree with each other and neither matches the (corrected) rule:**
- `compute_junctions_significance_psi` ([:151](leafletfa/differential_splicing.py#L151)) calls a junction
  significant if the cumulative FDP **at its own rank** is ≤ threshold — not a step rule, unsafe if the FDP
  curve is non-monotone.
- `compute_junctions_significance_groups` ([:326-333](leafletfa/differential_splicing.py#L326-L333)) uses
  `len(np.where(fdrs <= t)[0])` and then takes the top-n by probability. That is only correct if the set is
  a prefix; it should be `max(index) + 1` — which is precisely the "max detections" rule your Notion note
  says to adopt.
- Both then apply `|β| > min_effect_size` **after** computing the FDP, so the reported FDR is not the FDR of
  the reported set. δ is already inside P_kj; drop the redundant second filter.

Unify on one function implementing max-detections, and **build the missing `analyze_all_junctions_groups`
wrapper** (already your own TODO) since group DS is now the primary analysis. Vectorize it while you are
there: draw S posterior samples of (φ, ψ) once, compute `φ @ ψ → [S, C, J]`, and take group means across all
J junctions in one shot instead of looping per junction.

---

## 4. HIGH — waypoint ψ initialization does the opposite of what the Methods says

Methods: "The initial ψ for factor k is set to the **average** non-missing junction usage ratio across the
**50 nearest neighbors (to reduce noise)** of waypoint k."

The code does neither, and this **did** affect the paper:
- `initialize_psi_phi_with_random_non_zero` ([waypoints.py:81-92](leafletfa/waypoints.py#L81-L92)) takes
  `np.random.choice(non_zero_vals)` — **one random cell's value.** The function name says it. This
  *maximizes* noise where the Methods claims to reduce it.
- The pipeline builds metacells of **200** cells (`metacell_size=200` →
  `assign_nearest_cells(num_nearest=200)`), then throws 199 of them away per junction.
- Sampling only non-zeros **biases ψ_init upward**: in sparse data a true PSI of 0 is indistinguishable from
  no coverage, and this estimator can never return 0.
- Exact `PSI = 1.0` is common (a cell with reads on one junction of an ATSE). Pyro unconstrains a Beta site
  through a logit and `logit(1.0)` saturates to ≈15.9 — that entry starts on the flat tail of the sigmoid
  with a vanishing gradient and is **effectively frozen at 1.0 for the whole run**. φ is clipped to
  `[ε, 1−ε]`; **ψ is not.** Clip it.

**Fix:** coverage-weighted mean over the metacell (not a random draw), clip ψ_init to `[ε, 1−ε]`, and make
the neighbor count an explicit parameter. Then either set it to 50 to match the Methods, or update the
Methods to the value you actually use.

**Also:** nothing in the package writes `adata.varm["psi_init_{K}_waypoints"]` / `obsm["phi_init_{K}_waypoints"]` —
`train()` only *reads* them and raises if absent. Since `waypoints_use` **defaults to True**, the documented
quickstart (`LeafletFA(adata, K=20)` → `train()`) raises `ValueError` on any fresh AnnData. The glue exists
only inside `run_training.py`. Lift it into the package as a public function and test it.

---

## 5. HIGH (claim risk) — stratified waypoints use cell-type labels, and they *are* being used

[waypoints.py:106-193](leafletfa/waypoints.py#L106-L193) stratifies waypoint selection **by cell-type
label**. `run_training.py:220-230` calls it whenever a `broad_cell_type` / `cell_type` /
`cell_ontology_class` column exists — which it always does. Your Notion notes confirm the mouse sweep
"includes the phi carry-over fix and **stratified-by-cell-type waypoints**".

The Introduction claims LeafletFA finds programs "**without prior cell-type labels**", and Reviewer 2's
summary repeats that back ("without requiring prior cell type annotations or metadata"). If cell-type labels
enter through the initialization of the model whose programs are then shown to recover cell types
(Fig. 3E, Fig. 6B), that claim is compromised — and it is exactly the kind of circularity R1 already
attacked in the transfer-learning arm.

**DECIDED (2026-07-11): the final paper models will NOT use cell-type-stratified waypoints.** Stratification
was a test, not the shipped configuration. Concrete actions:
- Final models train with unstratified `max_min_sampling`, so the "without prior cell-type labels" claim holds.
- Add a guard so this cannot happen by accident: `run_training.py:220-230` currently opts *into* stratification
  implicitly, just because a cell-type column exists. Make it an **explicit parameter** (default off) rather
  than an incidental consequence of the AnnData schema.
- Keep the stratified run and report the comparison as a **robustness result** — "programs are unchanged
  whether or not the initialization is cell-type aware" is a strong answer to the circularity line of attack,
  and it feeds the "Initialization consistency" check already in the Notion plan.

---

## 6. Loss tracking and convergence (your question)

Your pipeline already does the two most important things right — **ELBO per cell** (`batch_elbo_per_cell`)
and a **pass-level average** (`pass_avg_elbo_per_cell`), plus `psi_mac_pass_to_pass` and π-evolution. Keep
those. The gaps:

1. **`batch_elbo` is `leaflet_model.best_elbo`, i.e. the *best* ELBO seen in that chunk, not the final one** —
   and it is normalized by *cells*, while the likelihood is over *observed (c,j) entries*. Chunks with
   deeper cells have more nnz and so a larger-magnitude ELBO for reasons that have nothing to do with fit.
   **Normalize by `nnz` in the batch, not by cell count.**
2. **Once you add the `N/B` likelihood scaling (§2), per-chunk ELBOs become directly comparable** — each is
   an unbiased estimate of the *full-data* ELBO. That is the number to track, and the one that should
   decrease monotonically-in-expectation across passes.
3. **The early-stopping rule is effectively broken.** `min_delta=0.01` is *absolute*, against an ELBO of
   1e6–1e8 — a relative tolerance of ~1e-9, i.e. numerically inert. So the rule degenerates to *"stop after 5
   consecutive epochs without a new global best"*, and with 1-particle `Trace_ELBO` noise, five consecutive
   non-record epochs happen easily on a still-descending curve. Against `num_epochs_later=20` this can end a
   chunk after ~6 epochs. **Make it relative and smoothed:** compare an EMA of the ELBO against
   `min_delta_rel * |best|` (≈1e-4). It has been absolute in every version of the code since day one.
4. **The real convergence criterion is held-out likelihood, not the ELBO.** You already mask (c,j) entries for
   the imputation eval — reuse that split *during* training: after each pass, compute held-out log-likelihood
   and imputation RMSE/Spearman. Stop when held-out stops improving. The ELBO alone will not show you
   overfitting, and this doubles as R1's requested calibration evidence.
5. **Judge convergence on the globals, not the loss:** `psi_mac_pass_to_pass → 0` and `pi_final_stability → 0`
   are the right signals, and you are already logging both. Add "number of factors with π above the prune
   threshold" per pass to catch **factor collapse** (a factor whose π → 0 is dead and cannot recover — see §9).
6. **Note the ELBO you print is not always an ELBO.** `log_prob_calc_sparse_optimized`
   ([model.py:531-532](leafletfa/model.py#L531-L532)) **drops the binomial coefficient** in the Binomial
   branch ("we can ignore it for gradient computation" — true for gradients, not for the value). Harmless for
   you (the pipeline uses `enable_cpu_optimization=False` and Beta-Binomial), but it means CPU-opt loss values
   are not comparable to GPU ones. Fix it so cross-config ELBO comparison (K selection!) is valid.

---

## 7. Transfer learning — the right way to apply a trained model to a new dataset

### The principle

LeafletFA has a clean **global / local** split, and that split *is* the recipe:

- **Global** (shared by all cells): Ψ, π, α, δ, the Beta hyperparameters a/b, and the likelihood concentration.
- **Local** (one per cell): φ_c.

Applying a trained model to new data means **posterior inference over the local variables only, conditioned on
the frozen globals.** Nothing else. This is the standard "E-step" / partial-VI projection, and it has a very
useful property: with Ψ fixed, the per-cell problems become **independent and K-dimensional**, so the
optimization is near-convex. No local-optima worry, no multiple restarts, no waypoints — all the §9
initialization anxiety simply does not apply on the transfer step. It is cheap and it parallelizes over any
number of cells.

The moment you re-learn *any* global on the target, you are no longer projecting — you are fitting a new
model that happens to be warm-started, and the resulting Φ is **not comparable to the source Φ**.

### The 70/30 held-out-cell projection — what to change in `07c` / `04f`

**The operation is correct and it is the right thing to test.** "Freeze the globals learned on 70%, infer φ on
the raw 30%" is exactly what the global/local split is for — and it is literally what happens when anyone
downloads the released model and applies it to their own cells. **The 70/30 test is testing the product.**

Because Ψ is frozen, each held-out cell is conditionally independent:

```
p(φ_c | y_c, T_c, Ψ, π, δ, conc) ∝ Dirichlet(φ_c ; δ·π) · Π_j BetaBinomial(y_cj | T_cj, (φ_c·ψ_:j)·conc, …)
```

so this is N_test independent K-dimensional problems — well-conditioned, near-convex, no local optima, no
waypoints, no restarts. None of the §9 initialization concerns apply on this step.

`infer_phi_chunk` ([07c_classify_heldout_phi.py:85-93](../Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/07c_classify_heldout_phi.py#L85-L93),
same machinery in `04f`/`04e`) gets two of the four globals right:

| Global | What the E-step does | Correct? |
|---|---|---|
| Ψ | `fixed_psi=psi_tensor` | ✅ genuinely frozen |
| δ | `delta_fixed=delta` | ✅ genuinely frozen |
| **π, α** | `pi_init=…`, `alpha_pi_init=…` | ❌ **only *initialized* — still re-learned** |
| **conc** | `input_conc_prior=inf` | ❌ **Binomial, but the model was trained Beta-Binomial** |

**1. π is not frozen — it is warm-started and then re-optimized.** `pi_init` only feeds `init_to_value` in the
guide ([model.py:801-807](leafletfa/model.py#L801-L807)); `pi` remains a **sampled site with its own guide**,
so SVI keeps optimizing it on the test cells. Since `φ_c ~ Dirichlet(δ·π)`, **π is the prior on φ**. If π
re-fits on the 30%, φ_test is inferred under a *different prior* than φ_train, and the train−heldout gap that
07c measures is confounded — it can move because π moved, not because the representation memorized.
→ **Add a real `pi_fixed` / `alpha_pi_fixed`**, mirroring exactly what `fixed_psi` already does (do not sample
the site; use the tensor). ~15 lines in `model()` plus the guard in the guide construction.

**2. The E-step uses a different likelihood than training.** The sweep trains with `input_conc=None` → learned
`bb_conc` → **Beta-Binomial**. The E-step passes `torch.tensor(float("inf"))` → **Binomial**. The comment
("Binomial: counts inform phi") suggests this was deliberate, but the consequence is that reads are treated as
*more* informative than the trained model believes: no overdispersion, sharper likelihood, φ pulled harder onto
the data and away from the prior. φ_test therefore ends up **over-confident and fit harder than φ_train was**,
which if anything **inflates the held-out decoding number** — i.e. it biases the memorization-gap test in the
flattering direction.
→ **Freeze `bb_conc` to the trained value** instead of switching to Binomial. If the Binomial E-step is wanted
for a specific reason, use it for *both* train and held-out φ so the comparison is like-for-like, and say so.

(`a`/`b` need no attention: they enter only through `p(ψ|a,b)`, and ψ is frozen, so they never touch φ.)

**3. Keep this distinct from the Fig. 2 imputation eval.** Entry masking asks "can `φ@ψ` fill in a missing
count for a cell it has already seen?" The 70/30 cell split asks "does the dictionary work on a cell it has
**never** seen?" Only the second speaks to whether the released model is usable, and it is the within-species
dress rehearsal for cross-species transfer. 07c's docstring already identifies the memorization confound in
`07` correctly — this is the right design, it just needs the two leaks above closed.

### What the current `fixed_psi` path gets wrong (general case)

1. **It freezes Ψ but re-learns π, α, and δ on the target.** This is the important one. π is the *prior* on Φ.
   Re-fitting it on the target means a program that is rare there gets its prior mass shrunk, and its inferred
   activity is pulled down **for prior reasons, not data reasons** — which confounds exactly the Δ-activity
   comparisons transfer exists to enable. **Add `pi_fixed` / `dir_conc_fixed` alongside `fixed_psi`** and infer
   Φ under the *source* prior.
   The one global you can defensibly re-learn is the **Beta-Binomial concentration**: dispersion is a property
   of the assay, not the biology. Re-learn it if the target's technical noise differs, but do it explicitly and
   say so.
2. **Nothing checks the feature axis.** `fixed_psi` is a bare `[K, J]` tensor; nothing asserts
   `fixed_psi.shape[1] == adata.shape[1]`, let alone that the junctions are in the **same order**. A silent
   misalignment produces confident garbage. Carry junction IDs on the model and assert on **identity, not
   length**.
3. **It plugs in Ψ's posterior mean, discarding Ψ's uncertainty**, which makes Φ overconfident. Freeze the
   variational posterior (loc **and** scale) and keep sampling Ψ from it so Φ inherits that uncertainty. At
   minimum, state in the Methods that Ψ is fixed to its posterior mean.
4. **Low-coverage cells are silently prior-driven.** With few reads, φ_c is dominated by `Dirichlet(δπ)` —
   which is *correct* shrinkage behaviour, but it means those Φ values carry no data. Report per-cell coverage
   alongside Φ so prior-driven cells can be flagged rather than interpreted.

### The three evaluations are different things — don't let them substitute for each other

You already have two of these. Keeping them straight is what makes the claims defensible.

| | What is held out | What it tests | Status |
|---|---|---|---|
| **(a) Entry masking** (Fig. 2B/2C) | random **(cell, junction) entries**, 25/50/75% within the test cells | *imputation* — can `φ@ψ` fill in missing counts for cells and junctions it has already seen? | ✅ have it |
| **(b) Cell-level held-out projection** | **30% of cells entirely** (`train_h5ad` / `test_h5mu`), then frozen-Ψ E-step to infer their φ | *within-species transfer* — does the dictionary generalize to cells never seen in training? | ✅ have it — `07c_classify_heldout_phi.py` does exactly this, and its docstring correctly identifies that decoding *training* φ was a memorization confound |
| **(c) Cross-dataset transfer + baselines** | the **entire target dataset** (human) | *is the structure shared* — or did I just assume it? | ❌ **missing the baselines** |

**Your 70/30 mouse split is (b), and it is the right control — but it cannot carry the conservation claim.**
Held-out mouse cells are drawn from the *same distribution* the dictionary was fit to. Success there shows the
dictionary is not memorizing; it says nothing about whether it survives a **distribution shift**. Conservation
is a claim about distribution shift, so it needs its own baselines.

### Making transfer *evidence* rather than an assumption

This is what R1 attacked, and it generalizes beyond species. **Projection onto a fixed dictionary cannot, by
itself, show the structure is shared — you assumed it when you fixed Ψ.** Reconstruction error on the target
has no scale: "mouse Ψ explains human data with RMSE 0.14" means nothing without something to compare to. Two
baselines give it a scale:

- **Baseline A — a target-native model** with the same K, evaluated on held-out *target* cells. If source Ψ
  reconstructs held-out human data nearly as well as a human-native Ψ does, that gap **is** the conservation
  measurement. A small gap is a strong, quantitative claim; a large one is an honest negative.
- **Baseline B — a scrambled dictionary**: permute Ψ's junction axis, or use Ψ from a mismatched tissue. This
  is the null. If real source Ψ does not beat it, the "transfer" was just K free parameters per cell fitting
  noise.

Report A and B and transfer becomes a defensible claim with an effect size. Without them it is circular.

### Two notes specific to your setup

- **Chunked training is *not* transfer** (I said this loosely before — it is worth being precise). Transfer
  *freezes* the globals; chunked training **updates** them every chunk. They share only the warm-start. The
  `N/B` problem in §2 is a property of the *updating*, and would not arise in a true frozen-Ψ transfer step.
- **For the conservation claim specifically, don't transfer at all.** Your Notion §4 already settled this and I
  agree: learn per species, match programs post hoc (Hungarian matching on ψ correlation over orthologous
  junctions), report the matching quality. Transfer stays the right tool for held-out cells, new batches, and
  the RNA Atlas bulk→mouse-latent projection — where baselines A and B still apply.

Finally, `get_psi_variational_params` sets `psis_loc = None` when `fixed_psi` is set
([model.py:928-929](leafletfa/model.py#L928-L929)), so a transferred model has no Ψ posterior samples —
`analyze_all_factors_psi` and `prune_K` will fail on it. Handle explicitly.

---

## 8. Priors in the code do not match the Methods table

| Parameter | Methods text | Methods table | Code | |
|---|---|---|---|---|
| Beta-Binomial conc. `c` | `Gamma(2,2)` | `Gamma(1,1)` | **`Gamma(10,10)`** | [model.py:173](leafletfa/model.py#L173) |
| Dirichlet conc. `δ` | `Gamma(1,1)` | — | **`Gamma(2,2)`** | [model.py:434](leafletfa/model.py#L434) |
| `a_j`, `b_j` | `Gamma(1,1)` | `Gamma(1,1)` | `Gamma(a_shape,a_rate)` with **learned** `a_shape,a_rate,b_shape,b_rate ~ Gamma(1,1)` | [model.py:381-388](leafletfa/model.py#L381-L388) |

Three different values for the Beta-Binomial concentration across text, table, and code — it looks like the
`Gamma(2,2)` was attached to the wrong variable somewhere. And the ψ prior has an **extra learned
hierarchical level** the Methods never mentions (4 additional latents). Reconcile all three and add the
hyperprior level to the Methods and the parameter table.

Also missing from the Methods entirely: **perplexity** (Fig. 3C/3D — the Results says "see Methods" but the
Methods never defines it — and R2 separately asks for it to be explained), **PVE** (Fig. 3A), **`prune_K`**,
and the ε-renormalization of `assign` ([model.py:445-447](leafletfa/model.py#L445-L447)).

---

## 9. Why initialization is load-bearing (for the Methods / rebuttal)

Worth stating explicitly, because a reviewer *will* press on SVI and local optima, and your Notion §3 already
lists "Initialization consistency" as a robustness check to surface.

The objective is **non-convex and non-identifiable** — Φ and Ψ are determined only up to a permutation of the
K factors, and mean-field SVI in the unconstrained (stick-breaking / logit) space converges to whichever local
optimum its basin contains. **The initialization picks the basin.** The paper's two guards (3 random restarts
selecting best ELBO; waypoint init) are the right ones.

But the Dirichlet-on-π hierarchy makes init **more** consequential here, not less. π is a shared global usage
vector with an `α/K` Dirichlet prior — an ARD-style shrinkage. A factor that is not immediately useful has
π_k driven toward 0, and a factor with π_k ≈ 0 receives almost no gradient and **can never recover**. So a
badly-initialized factor is not merely slow, it is *eliminated*. That is precisely what waypoint init defends
against: seeding Ψ from K maximally distant regions of the data starts every factor at a distinct,
data-supported anchor so that none is born useless. It is why you can report K=20 with meaningful π rather
than a handful of survivors.

Three concrete consequences in this codebase:

- **§1 interacts catastrophically with initialization.** With the likelihood gradient severed, the model
  *returns its own initialization* — which, under waypoint init, already looks biologically sensible and
  would pass every downstream sanity check. (Paper unaffected; users very much affected.)
- **Saturated init entries are dead on arrival** (§4): ψ_init = 1.0 → unconstrained ≈ 15.9 → vanishing
  gradient → frozen for the run.
- **Best-ELBO restart selection is currently unsound** (§6): with `patience=5` on a noisy 1-particle ELBO,
  restarts stop after different numbers of epochs, so `get_best_initialization()` compares runs of unequal
  length rather than converged optima.

---

## 10. Other confirmed bugs

- **`prune_K` corrupts `a`/`b`** ([model.py:1048-1052](leafletfa/model.py#L1048-L1052)). With
  `junc_specific_prior=True`, `a`/`b` are shape `[num_junctions]`, but they are indexed with `pruned_indices`,
  which are **factor** indices. Since `num_junctions >> K` this raises no `IndexError` — it silently
  overwrites them with the posterior means of a few arbitrary *junctions*. Junction-specific priors are global
  w.r.t. factors, so **`a`/`b` must not be pruned at all.** Present since the first commit. (The pipeline
  dodges it by hand-rolling its own `pi > 0.01` pruning, so the paper is unaffected — but any package user
  hits it.) `prune_K` also sets `psi_learned` while leaving `psi` unpruned.
- **Historical: the Beta-Binomial `alpha` lost its concentration factor for ~4 months.** `f0aca06` (9 Jun
  2025) changed `alpha = success_probs * input_conc` → **`alpha = success_probs`**; `09c1505` (14 Oct 2025)
  silently restored it. This affected the **GPU path too**. The current sweep (2026) is clear, but **any
  result carried over from the 2025 submission is not** — check before reusing anything.
- **`self.current_log_prob` is never set** — the setter was removed in `c712d04`, the reader survives at
  [model.py:605](leafletfa/model.py#L605). Every wandb `"log_likelihood"` has been `None` since.
- **`from_anndata` has an unbound local**: `device` is defined only inside `if float_type is None`
  ([:191](leafletfa/model.py#L191)) but used at [:228](leafletfa/model.py#L228) — passing `float_type`
  explicitly raises `NameError`.
- **`convertr` drops the device** ([:179](leafletfa/model.py#L179)). Works today only because a 0-dim CPU
  tensor broadcasts against CUDA tensors.
- **The Triton kernel never reads its mask** — `masked_matmul_kernel` takes `M_ptr` but masks only by array
  bounds ([_ops.py:38-39](leafletfa/_ops.py#L38-L39)), so `masked_matmul ≡ A @ B` with extra memory traffic,
  and `Out = torch.zeros_like(M)` allocates a second dense C×J. Numerically fine (so the new `A @ B` fallback
  is equivalent) but the "masked" optimization was never real. Relatedly, `initialize_triton_mask` allocates a
  dense C×J mask — fine at 5,000-cell chunks (~1.8 GB), impossible at full atlas scale (~51 GB). Another
  reason chunking is load-bearing.
- **`except Exception: return A @ B`** ([_ops.py:76-80](leafletfa/_ops.py#L76-L80)) swallows everything
  including OOM and retries the failing call every step. Catch narrowly and latch after first failure.
- **`clip_norm=10.0` is applied only when `enable_cpu_optimization=True`** ([:560-562](leafletfa/model.py#L560-L562)) —
  gradient clipping silently differs between the two flag settings.
- **Dead / broken code:** `utils.compute_and_plot_albf` calls `differential_splicing.compute_albf`, which
  **does not exist** → `AttributeError`; `EMDifferentialSplicing` consumes `albf_scores` nothing produces —
  the whole ALBF → Bayesian-FDR path is dead, yet `quickstart.ipynb` still advertises it.
  `utils.calculate_and_plot_correlations` imports `cost_correlation_assign`, absent from the repo and from
  `pyproject.toml`. All of `leafletfa/analysis_utils/` is unreferenced.
- **Docs describe an API that does not exist** (this is R2's complaint #4, so it matters):
  `docs/models/leafletfa.md` and `docs/outputs/*.md` tell users to read `adata.varm["psi_learned"]` and
  `adata.obsm["X_PHI"]` — **the package writes neither**. `docs/outputs/differential.md` calls
  `test_differential_activity`, which does not exist. `num_initializations` is documented as a constructor arg
  but is a `train()` arg. The README quickstart raises `ValueError` (see §4).

---

## Order of work

1. **§1** — fix `sparse_dot_cpu`; add the gradient-flow, factor-recovery, and parity tests.
2. **§2 — FIRST: run the scale-sensitive equivalence check.** Add the β_batch-vs-β_full **slope** to
   `08_compare_batch_equivalence.py` and run full-batch vs chunked on a ~15–20k-cell subset with identical
   waypoint init. **This single number decides whether §2 is a footnote or a refit.** The December check used
   only scale-invariant metrics (Pearson r, cosine, argmax) and could not have detected effect-size
   attenuation, so it does not settle the question. Then, regardless of the answer: add `obs_scale`, carry the
   Ψ guide *scale* across chunks, and fix the `get_pi`/`dir_conc` mutation that freezes δ.
3. **§5** — decide on stratified waypoints. This is a *claim* risk, not a code bug, and it gates the
   resubmission framing.
4. **§4** — fix the waypoint ψ initializer (mean not random draw; clip to `[ε, 1−ε]`); lift the init glue out
   of `run_training.py` into the package.
5. **§3** — unify the FDR functions on max-detections, build `analyze_all_junctions_groups` (vectorized),
   raise the default `num_samples`, and apply the Methods-text fixes from Notion.
6. **§6/§8** — relative+smoothed early stopping, held-out convergence criterion, reconcile the prior table,
   add perplexity/PVE/prune_K/transfer to the Methods.
7. **§7** — add a real `pi_fixed` / `alpha_pi_fixed` (currently π is only *initialized*, then re-learned) and
   freeze `bb_conc` in the E-step instead of switching to Binomial. **Re-run 07c after this** — the
   train−heldout memorization gap is currently confounded by both leaks, and the Binomial swap biases it
   flatteringly. Add the junction-axis assertion. Then add baselines A and B for any cross-dataset claim.
8. **§10** — sweep the remaining bugs and fix the docs (closes R2.4).

---

## Appendix — spec for the real-data batch-equivalence test (the immediate next action)

**Purpose:** decide whether chunked training attenuates the DS effect size β on *real* data with *waypoint*
init. Nothing else in §2 should be acted on until this returns.

**Setup.** Two runs on the **same cells**, differing **only** in batch size:
- Arm A: **full-batch** (`batch_size = n_cells`, so one chunk per pass)
- Arm B: **chunked** (`batch_size = 5000`, the production setting)

Use a subset that fits full-batch in GPU memory. `initialize_triton_mask` allocates a dense C×J mask, so at
J≈79,768 the budget is roughly `C × 79768 × 4 bytes`: **~15k cells ≈ 4.8 GB, ~20k cells ≈ 6.4 GB.** Size to
the card. Both arms must use **identical waypoint init** (pre-compute waypoints once on the subset and feed
the same `psi_init`/`phi_init` to both) and the same seed, K, priors, and total epoch budget.

**The metric that matters — add to `08_compare_batch_equivalence.py`:**

```python
# beta = the DS effect size, per (factor, junction): psi_kj - mean_{k'!=k} psi_k'j
def beta(psi):                                  # psi: (K, J)
    K_ = psi.shape[0]
    return psi - (psi.sum(0, keepdims=True) - psi) / (K_ - 1)

bA, bB = beta(psi_full), beta(psi_batch[col])   # col = Hungarian match from existing code
slope = np.polyfit(bA.ravel(), bB.ravel(), 1)[0]   # <-- THE NUMBER
```

Report, for both arms:
1. **`slope` of β_batch on β_full** — ≈1 means no attenuation (**question closed**); ≪1 means effect sizes are
   shrunk and Figs. 4E/5A need a refit. *Report the slope, not the correlation.*
2. `psi.std(axis=0).mean()` — factor contrast, per arm.
3. `psis_scale.mean()` — posterior width, per arm.
4. Number of significant DS junctions at the production FDR/δ, per arm.

**Why the existing metrics cannot answer this:** `psi_match_r` (Hungarian + Pearson), `phi_cell_cosine`, and
argmax agreement are **all scale-invariant**. They measure the *shape* of each program across junctions, not
the *amplitude* of the differences between programs. Shrink every factor toward a common per-junction mean and
all three stay near 1.0 while β collapses. Keep them — they answer "are these the same programs?" — but they
cannot answer "are the effect sizes the right size?"

**Run against current code** (post-`4b2a714` π fix, post-`2431d13` cluster-count fix). The December models
differ on three axes beyond N/B scaling, so comparing to them is confounded.

**Decision rule:**
- slope ≈ 1 → add `obs_scale` for posterior calibration, note it, move on. §2 is a footnote.
- slope ≪ 1 → the production models need refitting with `obs_scale` (or full-batch, or a real
  `pyro.plate(subsample=…)` loop), and the DS figures must be regenerated.

---

## Verification

- `pytest tests/ -v` after each change.
- **Gradient-flow test** (sparse mode, `enable_cpu_optimization=True`): Ψ/Φ guide `loc` gradients finite and
  non-zero after one `svi.step`.
- **Factor-recovery test**: simulate from the generative model, assert best-match `corr(ψ̂, ψ_true)` beats
  chance. This is the test that would have caught §1.
- **Parity test**: `enable_cpu_optimization` True vs False reach the same ELBO within MC noise (they currently
  differ by ~2,600 nats).
- **§2 sanity check**: with `obs_scale` on, one chunk's ELBO should approximate the full-data ELBO; and
  `psis_scale` should shrink by roughly `√(N/B)` relative to the current runs. `08_compare_batch_equivalence.py`
  looks like the right place to assert this.
- **Transfer control**: fit Ψ on 80% of mouse cells, freeze, infer Φ on the held-out 20%; confirm cell-type
  recovery and agreement with the jointly-fit Φ.
- Re-run `quickstart.ipynb` end-to-end after the waypoint fix; rebuild the docs.
