# LeafletFA — Methods (working copy, reconciled against the source code)

> **Purpose.** This is the paper's Methods section converted to Markdown, with every place the
> **code disagrees with the text** marked inline, plus a new section (§10) on the failure modes that
> actually determine whether a run learns anything.
>
> Markers used throughout:
> - 🔴 **CODE BUG** — the Methods is right, the code is wrong. Fix the code.
> - 🟡 **DOC BUG** — the code is right (or defensible), the Methods is wrong. Fix the text.
> - 🔵 **UNDOCUMENTED** — a real modelling choice the Methods never mentions. Add it.
> - ✅ **FIXED** — corrected in the code. Commit note given inline.

---

## FIX STATUS (running log — commit notes)

### Fixed in `leafletfa/` ✅

| # | Fix | Commit note |
|---|---|---|
| 1 | **CPU gradient bug.** `torch.no_grad()` in the sparse dot detached psi/assign from the likelihood, so they trained on the prior alone and the model returned its own initialization. | `fix: keep index_select inside the autograd graph in predict_rho` |
| 2 | **Triton dense mask deleted.** Allocated a dense C×J mask (**47.6 GB** at atlas scale, twice over), **never read the mask**, and ran `tl.dot` in **TF32** (7.7e-4 error vs float64). Replaced by a sparse gather on both devices: 1.5 GB, 9e-8 error. **This is why the atlas "didn't fit".** | `perf: replace triton masked-matmul with sparse gather (30x less memory, 4 orders more accurate)` |
| 3 | **Likelihood rewritten in `lgamma`.** `dist.BetaBinomial.log_prob` built ~15–20 nnz-sized intermediates retained by autograd **per ELBO particle** — this is what OOM'd the RNA-atlas run. `log(n choose k)` is constant in the parameters and is now cached once as a **scalar**. Identical to Pyro (rel. diff 0.00e+00), **~2.4× faster**. | `perf: hand-roll the beta-binomial log-prob; cache the binomial coefficient` |
| 4 | **`obs_scale`** — N/batch rescaling of the local terms. **NB: my first attempt was wrong** — `poutine.scale` inside `model()` cannot reach the guide, so it scaled `log p(assign)` but not `log q(assign)`, under-weighting the entropy that stops the per-cell posteriors collapsing. Now a `_ScaleSites` messenger wraps **model AND guide**. | `fix: scale local ELBO terms on both model and guide (poutine.scale in model() never reached the guide)` |
| 5 | **`get_pi` no longer mutates `self.dir_conc`** — that mutation is what silently froze δ at the first chunk's estimate for the remaining ~185 chunks. | `fix: get_pi writes dir_conc_post; stop mutating the input` |
| 6 | **`pi_fixed` / `alpha_pi_fixed`** — genuinely freezes π for transfer (`pi_init` only *initialises* a still-learned π). Plus a hard error if `fixed_psi`'s junction axis doesn't match. | `feat: pi_fixed for projecting new cells onto a trained model` |
| 7 | **`prune_K`** no longer indexes the per-junction `a`/`b` (shape `[J]`) with **factor** indices. | `fix: prune_K must not slice the per-junction a/b by factor index` |
| 8 | **`clip_norm` decoupled from `enable_cpu_optimization`.** Gradient clipping was ON only when that flag was True — so CPU runs were clipped and **GPU runs (which pass it False) were not**. | `fix: clip_norm is an optimizer choice, not a device choice` |
| 9 | **Index caching + coalescing on every device.** Was gated on `enable_cpu_optimization and cpu`, so **GPU runs re-ran `torch.equal` over two 2×nnz tensors every SVI step**. | `perf: cache sparse indices once, on every device` |
| 10 | **All priors and init tensors built on the model's device.** `dist.Gamma(1.0, 1.0)` builds CPU tensors → Pyro's prototype trace ran on CPU → **every AutoGuide param was allocated on the CPU while the model was on CUDA**, so each step sampled a 545k-dim Normal on CPU and shipped it to the GPU. **23× speedup** (4.0 → 0.17 s/epoch). | `fix: build priors on the model device (guide params were being created on CPU)` |
| 11 | **`min_delta_rel`** — scale-free early-stopping tolerance, plus a warning when early stopping fires in the first 25% of `num_epochs`. | `feat: relative early-stopping tolerance; warn on suspiciously early stops` |
| 12 | **Waypoint ψ init rewritten** (`initialize_psi_phi`). See §5.2 — coverage-aware **mean** (not one random value), **includes observed zeros**, **clipped away from the logit boundary**, and no 85 GB densification. Verified: init mean 0.677 → **0.501** against a true 0.502. | `fix: waypoint psi init — mean over covered cells, keep observed zeros, clip the logit` |
| 13 | `assign_nearest_cells` default `num_nearest` **5 → 50** (the Methods' value). | `fix: default metacell size to the documented 50` |
| 14 | Stale `_log_binom_coeff` cache across datasets; `get_bb_conc` reported `"infinity"` for *any* non-None prior; `convertr` device-rebuild broke on `Binomial`/`Categorical`. | `fix: assorted correctness (cache invalidation, bb_conc reporting, convertr)` |
| 14b | **`num_samples` default 10 → 100.** This is S in the DS FDR. At S=10, `P_kj` can only take 11 distinct values and the FDP is exactly 0 for any junction where all 10 samples clear δ — an artefact, not evidence. | `fix: raise the posterior-sample default to 100 (S=10 makes the FDP unusable)` |
| 15 | `stratified_max_min_sampling` **removed** — cell-type labels must not enter the init of a model claimed to work "without prior cell-type labels". | `fix: drop cell-type-stratified waypoints (circularity)` |

| 16 | **`best_elbo` was the LAST loss, not the best.** Under early stopping the last epoch is by definition a *non-improving* one (it is what tripped `patience`), and it is a single noisy MC estimate. This number drives model selection across the whole grid. Now `min(losses)`; the final-epoch loss is kept as `final_elbo` (a large gap between the two means the run was still descending). | `fix: select on the best loss, not the last (early stopping makes the last epoch non-improving by construction)` |
| 17 | **`model.health_check()`** — computes `psi_contrast`, `phi_spread` and `stopped_early`, stores them on `self.health`, and is called automatically at the end of `get_all_variables()`. See §11.4: the existing diagnostics ("0/K dead", "0/K never dominant") report *healthy* on a model that has learned nothing. | `feat: health_check — detect symmetric collapse, which the existing diagnostics cannot` |

### Fixed in `Leaflet-analysis/.../pipelines/05_leafletfa/run_training.py` ✅

Full path: `/gpfs/commons/home/kisaev/Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/run_training.py`

| # | Fix | Commit note |
|---|---|---|
| 18 | **PCA space was double-scaled.** `TruncatedSVD.fit_transform` **already returns `U·Σ`**; the code multiplied by `singular_values_` again → `U·Σ²`. Max-min sampling then ran in a space weighting each PC by σ² instead of σ, collapsing waypoint selection toward PC1 (PC1:PC5 weighting 3.6× → 12.7×) — the opposite of "chosen to capture major subspaces of variation". | `fix: TruncatedSVD.fit_transform already returns U*Sigma; stop squaring it` |
| 19 | **PC count fixed at 10**, per the Methods. Was `min(K, 50)`, so the waypoint space silently changed dimension with K and only K=10 matched the paper. A fixed budget also keeps the geometry comparable across the K sweep. | `fix: use a fixed 10 PCs for waypoints, independent of K` |
| 20 | **`metacell_size` 30 → 50**, per the Methods. | `fix: 50 nearest neighbours per waypoint, as documented` |
| 21 | **Coverage mask threaded into the waypoint init** (`psi_mask`, else `cell_by_cluster_matrix > 0`), so an observed PSI of 0.0 counts as data instead of being dropped as missing. | `fix: pass the ATSE coverage mask to the waypoint init` |
| 22 | **ELBO-per-cell divided by `len(batch)` instead of `n_cells`.** But `obs_scale = n_cells/len(batch)` had already put the loss on the full-data scale, so this inflated it 16.2×, and the short last batch of each pass (1,634 cells) looked ~5× larger than every other batch. Corrupted `pass_avg_elbo` and every wandb curve. | `fix: ELBO is already on the full-data scale after obs_scale; normalise by n_cells` |
| 23 | **Pruning misaligned the saved arrays.** `psi`/`assign_post`/`pi` were pruned; **`psis_loc`/`psis_scale` were not** — and all four were pickled. Saved `psis_loc` had `original_K` rows against `psi`'s `pruned_K`. **`psis_loc`/`psis_scale` are exactly what the DS analysis samples from.** Now pruned together, with an assertion on the factor axis. (`a`/`b` are per-junction and correctly left alone.) | `fix: prune every factor-indexed array together; assert the factor axis agrees` |
| 24 | `torch.set_default_tensor_type` removed (deprecated in torch 2.x, and no longer needed now that LeafletFA builds its tensors on its own device). | `chore: drop deprecated set_default_tensor_type` |

### New — group-vs-group differential splicing ✅

New module `leafletfa/differential_splicing_groups.py`. This is the **primary marker analysis**;
factor-level DS is reframed as *program characterisation*.

| # | Item | Commit note |
|---|---|---|
| 25 | **scVI change-mode DS between two groups of cells**, with **ΔPSI** as the effect size rather than a log-fold-change (PSI is a proportion, so a ratio is the wrong scale and is undefined at 0). Same posterior-sampling scaffold, same δ threshold, same Bayesian FDP. | `feat: group-vs-group differential splicing (scVI change-mode, DeltaPSI effect size)` |
| 26 | **Vectorised over all junctions.** ρ is *linear* in φ, so `mean_{c∈A} ρ_scj = (mean_{c∈A} φ_sck) @ ψ_skj` — the `[S, C, J]` tensor never needs to exist (at S=100, C=100k, J=80k it would be 3 PB). Replaces the per-junction Python loop. | `perf: average phi over the group first; never materialise [S,C,J]` |
| 27 | **ATSE-level score = total variation, per posterior sample.** `TV_se = ½ Σ_{j∈e} |Δ_sj|`. The junctions of an ATSE are a composition, so the question is "how much splicing mass moved within the event", not "did junction j move". Reduces exactly to `|ΔPSI|` for a 2-junction ATSE. **TV is computed per sample and only then thresholded** — TV of the posterior mean would discard the uncertainty the whole scaffold exists to propagate. | `feat: ATSE-level DS via per-sample total variation` |
| 28 | **Dropped the redundant `|β| > δ` filter applied AFTER the FDP.** δ is already inside `P_kj`; filtering afterwards changes the composition of the reported set after its FDP was computed, so the reported FDR no longer describes the set you report. | `fix: do not re-filter on effect size after computing the FDP` |
| 29 | `atse_simplex_check` — see the caveat below. | `feat: check whether imputed PSI sums to 1 within an ATSE` |

**CORRECTION to an earlier claim in this document.** A previous revision said the two existing FDR
functions "disagree with each other and neither implements max-detections". **That was wrong.**
`FDP(n)` is a running mean of `(1 − P)` over *descending* `P`, i.e. a running mean of an *ascending*
sequence, so it is **monotone non-decreasing in n by construction**. Verified over 20,000 random
draws: `len(where(fdp <= α))` and `argmax + 1` **always coincide**. Both existing functions were
already correct on that point.

What *is* wrong is the **Methods text**: it says *"τ\* is the **largest** τ satisfying FDP(τ) ≤ α"*.
FDP is non-increasing in τ, so the largest such τ is trivially 1.0 and selects nothing. It must be the
**smallest** τ — equivalently, **max detections**. 🟡

### 🔴 CAVEAT — the simplex is LEARNED, not ENFORCED

LeafletFA does not constrain `Σ_{j∈e} ρ_cj = 1` within an ATSE: each `ψ_kj ~ Beta(a_j, b_j)`
independently, and the likelihood scores each junction as its own Binomial with `T_cj` trials.
Measured on the K=5 simulation:

```
sum of imputed PSI within an ATSE (should be 1.0):
  mean 0.841   median 0.989   5-95% [0.217, 1.044]
```

The median says the model **does** learn it — but a minority of ATSEs violate it badly. TV assumes
the composition, so run `atse_simplex_check` before trusting ATSE-level scores, and state this in the
Methods. (A per-ATSE multinomial/Dirichlet likelihood would enforce it by construction — a real
modelling improvement available later.)

### ✅ FDR CALIBRATION — the empirical false-positive rate (Reviewer 1)

Run on `sim_heart_K5_binom_eff0.0` — a **pure null**, in which every ATSE has `atse_delta_psi = 0.0`
exactly, so **any call is a false positive by construction**. Model trained with the corrected
schedule (`lr=0.01, gamma=0.5`, S=100 posterior samples); `psi_contrast = 0.054`, health check passed.

```
[junction DS]  0 / 12,804 junctions called at FDP <= 0.05  (delta = 0.1)
[ATSE DS]      0 /  4,268 ATSEs     called at FDP <= 0.05  (delta_TV = 0.1)

  ==> empirical FPR = 0 / 4,268 = 0.0000
```

**This is the number Reviewer 1 asked for** ("the manuscript does not provide a clear empirical
estimate of false positives"). Report it.

### Still to fix 🔴

- 🔴🔴 **THE ψ PRIOR (§11.4d) — the biggest open question in the model.** `ψ_kj ~ Beta(a_j, b_j)` indexes
  `a_j`/`b_j` by **junction only**, so the same learned Beta is shared across all K factors at a junction.
  That is a **shrinkage prior pulling the K programs toward each other** — i.e. *toward the symmetric
  saddle* that is this model's main failure mode (§11.1). **The paper uses it. It was never tested.** On
  simulated data the *global* prior recovers ψ to **r = 0.994 with zero seed variance**; the
  junction-specific one scores **r = 0.31 with huge seed variance**. **It may be the cause of the seed
  variance we have been treating with restarts.** Both variants are now core axes of the atlas grid; the
  held-out 30% decides. **Do not report the junction-specific prior as the model until this returns.**
- **The LR schedule** — **`lr = 0.05`, `gamma = 0.5`** (swept against ground truth, §11.4b) in
  `01_generate_params.py` / `config/*.yaml`, and `patience`/`min_delta` must stop killing every batch at
  ~10 epochs (§11.3; use `min_delta_rel`). **This is the fix that decides whether the models learn anything
  at all.** The shipped `lr=0.001, gamma=0.01` sits at chance.
- `num_initializations=1` — the Methods says random init is "fit 3 times, keep best ELBO".
- **Factor-level DS** (§6): `compute_psi_effect_size` ignores Φ; the Methods defines a φ̄-weighted ρ.
  Now that group DS is the primary analysis, the simplest resolution is to keep the unweighted form and
  **fix the Methods equation** — but decide deliberately.
- `num_samples` default is **10** (the sweep used 100, the Methods says 500). At S=10, `P_kj` can only
  take 11 distinct values and the FDP is 0 for any junction where all 10 samples clear δ.
- Priors: `bb_conc ~ Gamma(10,10)` and `dir_conc ~ Gamma(2,2)` vs the Methods.
- **Methods text**: the extra hierarchical level on ψ (§4.2), the ε guards, the τ\* direction, and §11
  all need writing in.
>
> Source of truth for code claims: `leafletfa/model.py`, `leafletfa/waypoints.py`,
> `leafletfa/differential_splicing.py`, and
> `Leaflet-analysis/GenomeBiology_Submission2026/pipelines/05_leafletfa/`.

---

## 1. Single-Cell Smart-seq2 Data Processing

We processed publicly available datasets to generate inputs for LeafletFA model training and evaluation
(**Fig. 1A**). All measurements were taken from distinct individual cells or nuclei; no samples were measured
repeatedly. For mouse, GENCODE vM19 (GRCm38) was used for all alignments and annotations. For human,
GENCODE v45 (GRCh38). Four data sources:

1. **Tabula Muris Senis (TMS).** Smart-seq2 BAMs from the TMS consortium, obtained from the CZ Biohub S3
   bucket `s3://czb-tabula-muris-senis/`. Processed count matrices used for gene expression analysis.
2. **Allen Brain Atlas — Mouse.** Smart-seq2 single-nuclei RNA-seq (ssv4), GEO `GSE185862` and the Allen
   Brain Map portal. Exon-only counts, or adjusted intron+exon counts, used for expression analysis.
3. **Tabula Sapiens.** Smart-seq2 data from the Tabula Sapiens portal.
4. **Allen Brain Atlas — Human.** Human brain single-nuclei Smart-seq2 from the Allen Brain Map portal.

## 2. Gene Expression Normalization and Integration

To jointly analyse single-cell and single-nuclei data, raw gene counts were normalized for transcript-length
bias (divide by each gene's mean transcript length, multiply by the median transcript length across genes),
following best practice for Smart-seq2. For Allen Brain nuclei (which contain intronic reads) the same
normalization was applied to intronic counts using mean intron lengths. Length-normalized counts then
underwent library-size normalization and `log1p`.

QC: batch-wise robust statistics (median and MAD) for gene count and total count; cells retained within ±3 SD
of the batch-specific median. Additional filters: >10% ribosomal RNA content removed; cells lacking cell-type
annotation removed; lncRNAs removed from both the expression and splicing objects.

Dimensionality reduction used **Linear scVI** on the length-normalized raw integer counts, with **K = 20**.

## 3. Alternative Splicing Quantification and ATSE Mapping

Splice junctions were extracted from BAMs with **regtools** and processed with the **ATSEmapper** pipeline.
Each junction was validated against the species reference FASTA for a canonical splice-site motif (GT-AG,
GC-AG, AT-AC). GTF annotations classified each splice site as annotated or novel, retaining high-confidence
novel events while filtering alignment artefacts.

A splice graph is built per gene: nodes are unique splice sites, edges are observed junctions. An
**Alternative Transcript Structure Event (ATSE)** — analogous to a LeafCutter intron cluster — is a set of
two or more junctions interconnected through shared splice sites, forming one connected component of the
gene's splice graph. ATSEs capture cassette exons, alternative 5'/3' splice sites, and alternative
first/last exon usage (**Fig. 1B**).

Junctions were grouped into ATSEs **within each species separately**. Filters: minimum 10 reads and 5 cells
per junction; intron length between 50 bp and 500,000 bp. Read counts per junction, per ATSE, per cell were
aggregated into sparse `AnnData` objects.

---

## 4. The LeafletFA Beta-Dirichlet Factor Model

LeafletFA models junction read counts `y_cj` conditional on total ATSE read counts `T_cj`, for cell `c` and
junction `j`. Each cell is a weighted combination `φ_c:` of `K` latent factors, each with a usage ratio
`ψ_kj` per junction:

```
E[ y_cj / T_cj ]  =  ρ_cj  =  Σ_k  φ_ck · ψ_kj
```

### 4.1 Prior on φ

```
α      ~ Gamma(1, 1)
π      ~ Dirichlet(α/K, …, α/K)
δ      ~ Gamma(1, 1)                     ✅ FIXED: code now matches
φ_c|π,δ ~ Dirichlet(δ · π)               for c = 1 … C
```

> ✅ **FIXED** (`fix: dir_conc prior Gamma(2,2) -> Gamma(1,1) to match the Methods`). The code sampled
> `Gamma(2,2)` — same mean (1) but **half the variance**, i.e. *more* informative than the paper claims. Of
> the two, `Gamma(1,1)` is the more diffuse and lets the data speak.
>
> 🟡 **BUT THERE IS A DEEPER ISSUE, AND IT TOUCHES A HEADLINE RESULT.** δ scales `Dirichlet(δ·π)`, so with
> π ≈ 1/K the **per-component concentration is δ/K**. At δ ≈ 1 and K = 20 that is **0.05** — a prior that on
> its own already implies a median φ perplexity of **~2.5**:
>
> | K | δ | conc/component | median perplexity from the PRIOR ALONE |
> |---|---|---|---|
> | 20 | 0.8 | 0.040 | **2.26** |
> | 20 | 1.0 | 0.050 | **2.54** |
> | 20 | 5.0 | 0.250 | 7.07 |
>
> **The paper reports a median SP perplexity of 2.46** (Fig. 3C/3D), and fitted δ lands at ≈0.8. So the
> reported complexity is almost exactly what the prior gives you for free. **That does not make the finding
> wrong** — the data may genuinely support ~2.5 mixing — **but perplexity alone cannot distinguish a
> biological result from a restatement of the prior**, and a reviewer will ask.
>
> `health_check` now reports **fitted vs prior-alone perplexity** so the gap is visible. To *claim* the
> complexity result you need to show φ is informative **about cell type**, not merely that its perplexity is
> 2.46. (I first made this a hard failure and was wrong: a model that recovers the true ψ to r > 0.4 still
> shows perp_fit 3.09 vs perp_prior 2.96, because the simulated φ was Dirichlet too. It is a caveat, not a
> bug.)

### 4.2 Prior on ψ

Two configurations were explored; the **junction-specific** variant is used in the final model.

*Global:*
```
a ~ Gamma(1,1),  b ~ Gamma(1,1)
ψ_kj | a,b ~ Beta(a, b)
```

*Junction-specific (used):*
```
a_j ~ Gamma(1,1),  b_j ~ Gamma(1,1)      🔵 UNDOCUMENTED: code adds a hyper-prior level above this
ψ_kj ~ Beta(a_j, b_j)
```

> 🔴 **THE JUNCTION-SPECIFIC PRIOR IS A SHRINKAGE PRIOR TOWARD THE SYMMETRIC SADDLE, AND IT WAS NEVER
> TESTED.** See §11.5 — this is now the single largest open question in the model, and both variants are
> being run on the atlas. On simulated data the *global* prior recovers ψ to **r = 0.994** with **zero seed
> variance**; the junction-specific one — the paper's default — scores **r = 0.31** with huge seed variance.

> 🔵 **The code has an extra hierarchical level the Methods never mentions.** It does not sample
> `a_j ~ Gamma(1,1)` directly. It samples four *learned* hyper-parameters and then draws `a`/`b` from them:
> ```python
> a_shape, a_rate, b_shape, b_rate  ~ Gamma(1, 1)          # 4 extra global latents
> a_j ~ Gamma(a_shape, a_rate)      # per junction
> b_j ~ Gamma(b_shape, b_rate)      # per junction
> ψ_kj ~ Beta(a_j + ε, b_j + ε)
> ```
> This is an empirical-Bayes layer that lets the ψ prior adapt to the data. It is arguably *better* than
> what the paper describes — but it is four undeclared latent variables and it **must be added to the
> Methods and to the parameter table.** Note `a_j`/`b_j` are shared across all K factors (they are indexed
> by junction only), which is what makes them a *shrinkage* prior pulling the K programs toward each other
> — see §10.1.
>
> 🔵 The `+ ε` (`eps = 1e-6`) added to the Beta parameters for numerical stability is also undocumented.

### 4.3 Likelihood

The final model uses the **Beta-Binomial**.

*Beta-Binomial:*
```
conc ~ Gamma(2, 0.1)                     ✅ FIXED: mean 20, sd 14 (was Gamma(10,10), mean 1)
y_cj | T_cj, ρ_cj, conc  ~  BetaBinomial( T_cj,  ρ_cj·conc,  (1 − ρ_cj)·conc )
```

*Binomial:*
```
y_cj | ρ_cj ~ Binomial(T_cj, ρ_cj)       equivalent to conc → ∞
```

> ✅ **FIXED** (`fix: bb_conc prior Gamma(10,10) -> Gamma(2, 0.1) (mean 1 -> mean 20)`).
> **The concentration prior was wrong in all three places, and it biased the estimate.**
> - Methods **text** says `Gamma(2,2)` → mean 1, sd 0.71
> - Methods **table** says `Gamma(1,1)` → mean 1, sd 1.00
> - **Code** uses `Gamma(10,10)` → mean 1, **sd 0.32**
>
> All three have **mean 1**. But the concentration is an *inverse-overdispersion* parameter whose true value
> is typically 10–100. A prior with mean 1 shrinks it hard toward 1, and `Gamma(10,10)` is by far the
> tightest. **Measured on the K=5 simulation with a known true concentration of 20, the model recovered
> 11.75 — biased ~40% low.** Recommend a weakly-informative prior on the right scale (e.g. `Gamma(2, 0.1)`,
> mean 20, sd 14) and reporting the learned value.

### 4.4 Parameter table (corrected)

| Parameter | Distribution (as coded) | Description |
|---|---|---|
| `a_shape, a_rate, b_shape, b_rate` | `Gamma(1,1)` | 🔵 **undocumented** hyper-priors on the ψ prior |
| `a_j` | `Gamma(a_shape, a_rate)` | junction-specific shape (shared across all K factors) |
| `b_j` | `Gamma(b_shape, b_rate)` | junction-specific rate (shared across all K factors) |
| `ψ_kj` | `Beta(a_j + ε, b_j + ε)` | factor–junction usage ratio (PSI) |
| `α` | `Gamma(1,1)` | Dirichlet concentration for π |
| `π` | `Dirichlet(α/K, …, α/K)` | global program usage |
| `δ` | `Gamma(1,1)` ✅ | Dirichlet concentration for φ. **NB: δ/K is the per-component concentration — see §4.1** |
| `φ_c` | `Dirichlet(δ · π)` | per-cell program activities |
| `conc` | `Gamma(2, 0.1)` ✅ (mean 20) | Beta-Binomial concentration (inverse overdispersion) |
| `ρ_cj` | `Σ_k φ_ck ψ_kj` | deterministic |
| `y_cj` | `BetaBinomial(T_cj, ρ·conc, (1−ρ)·conc)` | observed junction counts |
| `T_cj` | — | total ATSE reads (trials) |

> 🔵 **Also undocumented in the code:** after sampling, `φ` is ε-renormalized
> (`if any(φ < ε): φ += ε; φ /= φ.sum()`), and `ρ` is clamped to `[1e-6, 1 − 1e-6]` before the likelihood.
> Both are numerical guards, but they perturb the model and should be stated.

---

## 5. Training

### 5.1 Multi-pass mini-batch training

The atlas is trained over **10 full passes**, cells in mini-batches of **5,000**. Global parameters `Ψ` and
`Π` evolve continuously across passes:

1. **Batch training.** Each batch fits the local `Φ` for its 5,000 cells while refining the globals `Ψ`, `Π`
   against the ELBO.
2. **Global update and re-initialization.** After each mini-batch, the refined `Ψ`/`Π` are extracted and used
   to initialize the *next* batch.
3. **Learning-rate decay.** A factor of 0.95 is applied to the base LR at the start of each new pass. First
   pass 50 epochs; later passes 20 epochs.

> ✅ **FIXED — but note the first attempt was WRONG.** `poutine.scale` inside `model()` cannot reach the
> guide, so it scaled `log p(assign)` but not `log q(assign)`, under-weighting the entropy that stops the
> per-cell posteriors collapsing. A `_ScaleSites` messenger now wraps **model AND guide**, and the test
> asserts the guide side. Original problem: **the likelihood was never rescaled by N/batch.** Each chunk's ELBO charged the **full** global prior
> (over ψ across *all* junctions, plus a, b, π) against the likelihood of **only 5,000 cells** — i.e. it
> optimized the ELBO of a 5,000-cell dataset, not an unbiased estimate of the full-data ELBO. **Fixed:** an
> `obs_scale = N_total / batch_size` now multiplies the likelihood **and the local `φ` prior/entropy**,
> leaving the global KL at weight 1. The Methods should state this explicitly.
>
> 🔴 **`Ψ`'s posterior width is wrong, and the obvious fixes DON'T WORK. The real fix is architectural.**
>
> Two hypotheses were tested and **both were wrong** — recording them so nobody re-tries them:
>
> **(a) "Carry the guide's scale forward across chunks."** Measured against a full-batch reference on
> simulated data (1,200 cells, K=3, 4 chunks × 3 passes):
>
> ```
> full-batch (the truth)       mean q(psi) width = 0.173    psi_r = 0.997
> chunked, scale RESET         mean q(psi) width = 0.265    (1.5x too wide)
> chunked, scale CARRIED fwd   mean q(psi) width = 0.411    (2.4x too wide)  <- WORSE
> ```
> Each chunk's optimisation *widens* from its narrow start (0.1 → 0.265), so carrying that width forward
> **compounds the widening** instead of accumulating evidence. The capability exists
> (`train(psi_scale_init=...)`) but **`run_training.py` deliberately does not use it.**
>
> **(b) "Drop chunking — the 47.6 GB dense mask is gone, so full-batch should fit now."** It does not. The
> real training matrix is **13.8% dense — 1.46 BILLION nonzeros** (132,706 × 79,768). Full-batch needs
> **~305 GB** (the `[nnz, K]` gather alone is 218 GB) against a 95 GB GPU. **Chunking is genuinely required.**
>
> **(c) "Just run each chunk longer — obs_scale already targets the full-data width."** Also wrong, and
> **backwards**:
>
> ```
> REFERENCE full-batch:  q(psi) width = 0.1731
>  epochs/chunk   q(psi) width   vs reference   psi_r
>            60        0.2652        1.53x      0.986
>           200        0.3372        1.95x      0.993
>           600        0.3721        2.15x      0.992    <- MORE epochs = WORSE width
> ```
>
> **Why:** `obs_scale` gets the *magnitude* right (scaling the log-likelihood by N/B multiplies the Fisher
> information by N/B, so one chunk's psi posterior *should* have the full-data width). But **each chunk sees
> different cells and disagrees with the others.** Cycling chunks pulls `q(psi)` back and forth, and its width
> **inflates to cover the between-chunk disagreement**. Training each chunk *harder* makes it pull *harder*,
> so the inflation grows. **This is structural — no tuning fixes it, because the chunks fight each other.**
>
> **PRACTICAL CONSEQUENCE — a free win:** use **fewer** epochs per chunk. 60 is both faster *and* gives the
> least-inflated posterior. Once early stopping is fixed (§11.3), do NOT raise `num_epochs_later` to 1000 —
> cap it around 50–100.
>
> **The actual problem is the per-chunk TEARDOWN, not the scale.** A fresh `LeafletFA` — and so a fresh
> `AutoGuide` and a fresh `ClippedAdam` — is built for every one of the ~186 chunks. That throws away the
> optimizer state and re-initialises `q(ψ)` every time, so it can **never** accumulate evidence. Warm-starting
> the mean papers over this; warm-starting the scale makes it worse.
>
> **The fix is one persistent guide + optimizer for the whole run, with `pyro.plate(subsample_size=B)` inside
> the model.** Then `q(ψ)` is a single set of parameters refined by *every* minibatch, so its width converges
> to the full-data posterior by construction; only B cells of likelihood ever materialise (at B=8,192 the
> gather is ~14 GB, comfortably inside 95 GB); and the plate applies the N/B scaling to model **and** guide
> automatically — which is exactly what the `_ScaleSites` messenger currently hand-rolls. **This is what the
> deleted `LeafletFAminibatch.py` did** (`git show 6f9ce6f:src/BetaDirichletFactor/LeafletFAminibatch.py`).
>
> **Impact meanwhile — and this is the bottom line:** the ψ/φ/π **point estimates are FINE**, and this
> replicates on the real Figure-2 simulation (chunked-vs-full-batch agreement **r = 0.881**, `psi_contrast`
> 0.126 vs 0.119). Everything Figs. 3/4/6 rest on is sound. `q(ψ)`'s **width** does not match full-batch, and
> **the direction of the error is not consistent** (1.5× wider on a small toy, 6× narrower on the real sim), so
> **do not make any claim about DS being conservative or anti-conservative from it.** Use the **direct
> empirical FPR** on the null instead (0 / 4,268 ATSEs). See §11.4c.
>
> ✅ **Guarded by a test:** `test_chunked_training_recovers_the_same_psi_as_full_batch` asserts the chunked
> scheme lands where full-batch lands, on a simulation small enough that full-batch is feasible. It runs on
> every commit, so the minibatch scheme cannot silently drift again.
> (`test: pin chunked training against a full-batch reference`)
>
> Original finding: a fresh guide is built per batch and only
> ψ's **mean** is carried forward; `psis_scale` is extracted and never fed back. Since the DS/FDR analysis
> samples from that posterior, its uncertainty currently reflects one 5,000-cell batch, not the atlas.
>
> ✅ **FIXED** (`fix: get_pi writes dir_conc_post; stop mutating the input`). **`δ` was silently frozen after the first chunk.** `get_pi()` overwrote `self.dir_conc` with the learned
> value, which `run_training.py` then passed back as `delta_fixed` — so δ was estimated on the first 5,000
> cells and frozen for the remaining ~185 chunks. **Fixed** (learned value now goes to `dir_conc_post`).
>
> 🔴 **Epoch counts contradict the Methods.** The param grid uses `num_epochs_first = 500` and
> `num_epochs_later = 1000`, not 50 and 20.
>
> 🔴 **There are TWO learning-rate decays, and the Methods only mentions one.** The per-pass `0.95 ** pass_idx`
> *and* an epoch-wise `lrd = gamma ** (1/num_epochs)` inside `fit()`. With the hardcoded `gamma = 0.01`, the
> LR decays **100× within every pass**. See §10.2 — this is the single most damaging setting in the pipeline.

### 5.2 Model fitting and initialization using waypoints

We fit LeafletFA with **Stochastic Variational Inference** in **Pyro**, optimizing the ELBO. Two
initialization approaches: random (fit 3×, keep the best ELBO) and **waypoint-based**, inspired by Palantir,
to reduce run-to-run variability from local optima.

1. **Initial waypoint (first batch).** Number of waypoints = `K`. Waypoints capture the major subspaces of
   variation in the observed ratios `ρ̂_cj = y_cj / T_cj`. Non-missing values are mean-centered to reduce the
   influence of total expression. **Ten PCs** are extracted by sparse SVD. **Max-min sampling** in PC space
   selects `K` diverse cells. The initial `ψ` for factor `k` is the **average non-missing junction usage
   ratio across the 50 nearest neighbours** (to reduce noise) of waypoint `k`.
2. **Sequential initialization (subsequent batches).** Initialize from the accumulated global `Ψ`/`Π`.

> ✅ **FIXED (`fix: waypoint psi init — mean over covered cells, keep observed zeros, clip the logit`).**
> Rewritten as `waypoints.initialize_psi_phi`: coverage-aware **mean** over the metacell, **observed zeros
> included** (via a `mask`), **clipped to `[1e-4, 1-1e-4]`**, and no dense densification. Verified: the init
> mean goes from 0.677 (biased) to **0.501** against a true 0.502. The old function is kept as a deprecated
> alias that warns. What it used to do:
> [`waypoints.py`](leafletfa/waypoints.py) `initialize_psi_phi_with_random_non_zero` takes
> `np.random.choice(non_zero_vals)` — **one random cell's PSI value per junction** — not the *average* over
> neighbours. The function name says so. This *maximizes* noise where the text claims to reduce it, and it
> weakens exactly the symmetry-breaking that waypoints exist to provide (§10.1).
>
> ✅ **FIXED.** `assign_nearest_cells` now defaults to **50** and the pipeline passes **50**
> (`fix: default metacell size to the documented 50`). It was 5 in the package and 30 in the pipeline.
>
> ✅ **FIXED.** `run_training.py` now uses a **fixed 10 PCs**, independent of K
> (`fix: use a fixed 10 PCs for waypoints, independent of K`). It was `min(K, 50)`, so the waypoint space
> silently changed dimension with K and only K=10 matched the paper.
>
> ✅ **FIXED.** ψ_init is now clipped to `[1e-4, 1 − 1e-4]`. Previously an exact `1.0` (which was **51.7%**
> of the values the old sampler could draw) unconstrained to a guide loc of **+15.94**, where
> `sigmoid'(loc) ≈ 1.2e-7` — so **about half the ψ matrix was frozen before training started**.
>
> ✅ **FIXED.** A coverage `mask` (`psi_mask`, else `cell_by_cluster_matrix > 0`) now distinguishes an
> *observed* PSI of 0 from *missing* data, so genuine zeros count. **58.6% of stored `junc_ratio` values are
> exactly 0.0** and were all being discarded.
>
> ✅ **Cell-type-stratified waypoints have been REMOVED.** An earlier version stratified waypoint selection
> by cell-type label. That would have let labels enter the initialization of the very model whose programs
> are then shown to recover cell types — contradicting the claim that programs are found "without prior
> cell-type labels." Final models use unsupervised `max_min_sampling`.

---

## 6. Bayesian Factor-Specific Differential Splicing

Given the inferred latents, factor-specific differential splicing compares factor `k`'s contribution to
junction usage against the remaining factors, using posterior samples from the variational approximation.

**Posterior sampling and effect size.** With `S` posterior samples (**default 500** in the Methods):

```
ρ_{s,kj} = (1/C) Σ_c  φ_{s,ck} · ψ_{s,kj}
Δψ_{s,kj} = ρ_{s,kj} − (1/(K−1)) Σ_{k'≠k} ρ_{s,k'j}
β_kj      = (1/S) Σ_s Δψ_{s,kj}
P_kj      = (1/S) Σ_s  1( |Δψ_{s,kj}| > δ ),     δ default 0.1
```

**FDR.** Rank junctions by `P_kj`; the posterior expected false discovery proportion at threshold τ is

```
FDP_k(τ) = (1 / N_k(τ)) · Σ_{j : P_kj ≥ τ} (1 − P_kj)
```

and `τ*` is chosen as the **most permissive** τ with `FDP_k(τ*) ≤ α` (α default 0.05).

> 🟡 **RESOLVED — these are TWO DIFFERENT ANALYSES, and the Methods currently conflates them.**
>
> - **Factor-level DS** (`compute_psi_effect_size`) asks *"which junctions DEFINE program k?"* That is a
>   property of the learned dictionary Ψ: compare `ψ_kj` against `ψ_k'j`. **No cells are involved, and Φ
>   should NOT enter.** The Methods' `ρ_skj = ψ_skj · φ̄_sk` multiplies each program's ψ by how often that
>   program is *used*, so a rarely-used program has its own defining markers shrunk toward zero — which is
>   not what "what is this program" means.
> - **Group DS** (`differential_splicing_groups`, NEW) asks *"which junctions differ between cell type A and
>   B?"* Here **Φ must enter**, because a cell's imputed PSI *is* `φ_c @ ψ`. **This is the cell-type marker
>   analysis, and it is φ-weighted by construction.**
>
> **Action:** keep both. Use group DS for markers (the primary analysis); keep factor-level unweighted for
> program characterisation. **Fix the Methods to define the two separately** — right now it gives one
> equation and implies it does both jobs.
>
> Original finding (for the record): the code does not implement the ρ above — it ignores Φ entirely.
> `differential_splicing.compute_psi_effect_size` computes `ψ_{s,kj} − mean_{k'≠k} ψ_{s,k'j}`, with **no φ
> weighting**. Its own docstring says "without weighting by factor usage." Under the Methods,
> `ρ_{s,kj} = ψ_{s,kj} · φ̄_{s,k}`, so each factor's contribution is scaled by its **mean usage** and
> low-usage factors shrink toward zero. The code gives every factor equal weight regardless of π. This
> changes which junctions are called, especially for the low-utilization programs the paper highlights.
> **Decide which estimator is intended and make them agree.**
>
> ✅ **FIXED** (`fix: raise the posterior-sample default to 100`). **Posterior sample count.** Methods says 500. The atlas sweep used **100**
> (`01_generate_params.py: "num_samples": [100]`). The package default is **10** — at S=10, `P_kj` can only
> take values in {0, 0.1, …, 1.0} and the FDP estimator reports 0 for any junction where all 10 samples
> clear δ, which is an artefact, not evidence.
>
> ⚠️ **CORRECTION — an earlier revision of this document was WRONG here.** It claimed the two FDR functions
> "disagree with each other and neither implements max-detections". They do not disagree. `FDP(n)` is a
> running mean of `(1 − P)` over *descending* `P` — i.e. a running mean of an *ascending* sequence — so it is
> **monotone non-decreasing in n by construction**. Verified over 20,000 random draws:
> `len(where(fdp ≤ α))` and `argmax + 1` **always coincide**. Both existing functions were already correct.
>
> 🟡 What *is* wrong is the **Methods text**: it says "the **largest** τ satisfying FDP(τ) ≤ α". FDP is
> non-increasing in τ, so the largest such τ is trivially 1.0 and selects nothing. It must be the
> **smallest** τ — equivalently, **max detections**.
>
> ✅ **FIXED in the new group-DS module** (`fix: do not re-filter on effect size after computing the FDP`).
> Both old functions applied `|β| > δ` **after** the FDP, which changes the composition of the reported set
> after its FDR was computed — so the reported FDR did not describe the set being reported. δ is already
> inside `P_kj`. The legacy factor-level functions still do this. 🔴
>
> 🟡 **Text fixes already identified:** restore the missing `s` index in the β equation (Reviewer 4); change
> "largest τ" to "most permissive threshold under α / max detections"; soften "ensures at most 5% false
> positives" to "controls the posterior expected FDP".

---

## 7. Simulating Junction Usage Ratios Across Cell Types

Simulations are based on observed ATSE counts and predefined cell types from real TMS brain cells, so the
sparsity of read coverage is preserved. Every 3-junction ATSE is randomly labelled positive (differential
across cell states) or negative. The three junctions are sorted into two inclusion junctions (**J1**, **J2**)
and one exclusion junction (**J3**).

- **Negative ATSEs:** one shared `ψ^J3 ~ Beta(0.5, 0.5)` for all cell states (hyperparameters chosen to
  emulate the known bimodality of splicing).
- **Positive ATSEs:** a different `ψ^J3_k ~ Beta(0.5, 0.5)` per cell state `k`.
- In both cases `ψ^J1_k = ψ^J2_k = (1 − ψ^J3_k) / 2`.

Junction counts are sampled from binomial distributions with these `ψ` and the total counts observed in the
real TMS data. Observed differences are checked post-hoc to confirm they match the assigned label.

## 7b. Held-out evaluation — the 70/30 split

> **NEW SECTION. This is not in the current manuscript and must be added.**

### The split

Cells are split **70% train / 30% held-out**, at the level of **whole cells** (`train_h5ad` / `test_h5mu`),
not at the level of individual counts. The model is trained on the 70% and never sees the 30%.

**This is a different split from the imputation masking in Fig. 2B**, and the two answer different questions:

| | what is held out | what it tests |
|---|---|---|
| **entry masking** (Fig. 2B/2C) | random **(cell, junction) entries** within the test cells | *imputation* — can `φ@ψ` fill in a missing count for a cell and junction it has already seen? |
| **70/30 cell split** | **30% of cells entirely** | *generalisation* — does the learned dictionary work on a cell it has **never** seen? |

Only the second speaks to whether the released model is usable, and it is the within-species dress rehearsal
for cross-species transfer.

### How held-out φ is inferred: FREEZE the globals, INFER only φ

LeafletFA has a clean global/local split:

- **global** (shared by every cell): `Ψ`, `π`, `α`, `δ`, `a`/`b`, and the Beta-Binomial concentration;
- **local** (one per cell): `φ_c`.

Applying a trained model to held-out cells is therefore **posterior inference over the LOCAL variables only,
conditioned on FROZEN globals**:

```
p(φ_c | y_c, T_c, Ψ, π, δ, conc)  ∝  Dirichlet(φ_c ; δ·π) · Π_j BetaBinomial(y_cj | T_cj, (φ_c·ψ_:j)·conc, …)
```

Because Ψ is fixed, **each held-out cell is conditionally independent of every other** — so this is N_test
independent K-dimensional problems. It is near-convex, needs no restarts or waypoints, and batches trivially.
None of the initialization pathology in §11 applies to this step.

### 🔴 The leakage trap — and what was wrong

**If you merely *initialise* the globals and let SVI keep learning them, you are FITTING A NEW MODEL ON THE
TEST DATA.** Ψ and π adapt to the held-out cells, and the "held-out" score becomes partly a *training*
score. It flatters the result.

Two specific bugs, now fixed in all of `04*_evaluate_imputation*.py` and `07c_classify_heldout_phi.py`:

- 🔴→✅ **`pi_init` → `pi_fixed`.** `pi_init` only *seeds* π; SVI then **re-learned it on the test cells**.
  π is the **prior on φ**, so a re-learned π means held-out φ is inferred under a *different prior* than
  training φ — and any train-vs-test comparison is confounded. *(`fix: freeze pi in the held-out E-step; pi_init only seeded it`)*
- 🔴→✅ **Forced `input_conc_prior=inf`** (Binomial) even when the model was trained Beta-Binomial, so φ was
  fit under a different likelihood than the one that produced Ψ. Now uses the **trained `bb_conc`**.
  *(`fix: use the trained likelihood in the held-out E-step`)*

  **Why this was there:** a comment in `04e` reads *"instead of the tiny learned bb_conc that leaves the
  likelihood ~flat"*. That was a **workaround for a bad prior** — the old `bb_conc ~ Gamma(10,10)` has mean 1,
  and a concentration near 1 means `Beta(ρ, 1−ρ)`, maximally overdispersed, likelihood nearly flat. The new
  `Gamma(2, 0.1)` prior (mean 20) fixes the *cause*, so the workaround is no longer needed. **Models trained
  before that prior change will still have a tiny `bb_conc` — do not run the E-step against them.**

(`a`/`b` need no attention: they enter only through `p(ψ|a,b)`, and ψ is frozen, so they never touch φ.)

### What is measured on the 30%

1. **Imputation** (`04*`) — mask entries within the held-out cells, impute with `φ@ψ`, score RMSE / L1 /
   Spearman against NMF. **Report calibration, not just rank correlation** (Reviewer 1): predicted vs observed
   PSI binned across the range, MAE/RMSE in PSI units, and check for a systematic offset or a slope below one.
2. **Cell-type decodability** (`07c`) — decode broad cell type from the **held-out inferred φ**. The
   **train − heldout gap is the memorization estimate.** (`07c`'s docstring already identifies that decoding
   *training* φ was the confound in `07`.)
3. **Age regression** — within cell type, on held-out φ.

**Model selection across the K sweep must be done on these held-out metrics, not on the ELBO.**

## 8. Modeling Aging with Splicing and Expression

Predictive power of splicing vs expression for chronological age, via linear regression at cell-type-specific
and global levels. Per broad cell type with ≥50 cells spanning ≥2 age groups, three models predict numeric
age: splicing latents (`Φ`) alone, gene-expression factors alone, and both. `R²` by 10-fold CV; top features
by absolute coefficient. Globally, four feature combinations are compared: broad cell type alone (one-hot),
cell type + splicing, cell type + expression, and all features; 10-fold CV `R²` mean and SD.

## 9. RBP Aging Mediation Analysis

Mediation analysis over 269 curated RBPs, hypothesizing age (`X`) → RBP expression (`M`) → SP activity (`Y`),
run independently per tissue-cell-type pair. `pingouin.mediation_analysis`; the mediator is the
scVI-denoised expression of each RBP, the outcome is the LeafletFA `φ`. Indirect-effect significance by
bootstrapping (500 iterations) for 95% CIs and p-values. Benjamini-Hochberg FDR within each cell type;
significant at FDR < 0.05. The **Aging Mediation Coefficient** is the standardized indirect-effect
coefficient.

## 10. Cross-species Splicing Conservation

ATSEs filtered to junctions from genes with one-to-one Ensembl orthologs; only ATSEs with ≤5 junctions.
Annotations from GENCODE vM19 (mouse) and v45 (human), on GRCm38 (mm10) and GRCh38 (hg38).

Junction coordinates exported to BED and lifted in both directions with UCSC `liftOver`. Lifted coordinates
compared against all annotated junctions in the target species within a ±100 bp window. For sequence
validation, ±50 bp of exonic context flanking donor and acceptor sites was extracted from both the lifted and
matched coordinates; pairwise similarity by global alignment, plus splice-motif conservation (GT-AG, GC-AG).
Junctions classified as **conserved** (≥80% similarity *and* conserved motifs), **sequence_only** (≥80%
similarity, motifs not conserved), **motifs_only** (motifs conserved, <80% similarity), or **diverged**.
Duplicate mappings resolved by highest similarity and lowest coordinate distance.

---

# 11. Caveats, failure modes, and what to check BEFORE a full-scale run

> **This section is new.** Everything below was measured on the K=5 heart simulation
> (`sim_heart_K5_binom_eff1.0_seed0.h5ad`, 6,986 cells × 12,804 junctions, 7.3% density) where the ground
> truth is known, and reproduced on the RNA-atlas bulk data. It should go into the paper — a reviewer will
> ask about SVI and local optima, and it is far better to have measured it than to be asked.

## 11.1 The model starts *on* a symmetric saddle point

This is the single most important thing to understand about fitting LeafletFA.

The **generative model** is not symmetric: `φ_c ~ Dirichlet(δπ)` gives every cell an independent draw. But
SVI never samples φ from the prior. It fits a **variational distribution** `q(φ_c) = Normal(loc_c, scale_c)`
in unconstrained (stick-breaking) space and optimizes `loc_c` as a free parameter. Pyro's
`AutoDiagonalNormal` initializes `loc` with `init_to_median` — **the median of the prior**. Every cell shares
the same prior, so **every cell gets the same starting `loc`.**

Measured at epoch 0:

```
φ std ACROSS cells   = 0.00003     ← all cells identical
ψ contrast ACROSS K  = 0.00544     ← all K programs identical
```

Why that is a trap: when φ is uniform across cells, `ρ_cj = Σ_k φ_ck ψ_kj` is **the same for every cell**.
The likelihood then only constrains the *average* of ψ over programs, so **every ψ_k receives an identical
gradient** and the programs cannot separate. And because all programs look alike, φ has no reason to prefer
any of them. Each half holds the other in place.

The junction-specific ψ prior makes this worse: `a_j`, `b_j` are shared across all K factors, so
`ψ_kj ~ Beta(a_j, b_j)` actively **shrinks the K programs toward a common per-junction value** — i.e. toward
the saddle.

Escape is driven only by Monte-Carlo gradient noise, which grows roughly **exponentially** — slow at first,
then fast. The growth rate scales with the **learning rate**.

## 11.2 Learning rate is the escape velocity. This is the setting that matters most.

Measured on the simulation, **random init throughout**, K=5, chance dominant-agreement = 20%:

| config | ψ recovery | φ recovery | dominant agree | ψ contrast | ELBO |
|---|---|---|---|---|---|
| `lr=.001 γ=.01` 500ep `patience=5` **(the shipped default)** | **−0.010** | +0.013 | **21.7%** | 0.0027 | 1.184e9 |
| ↑ + patience disabled (500 ep) | −0.003 | +0.009 | 20.9% | 0.0028 | 1.173e9 |
| ↑ + 3× the epochs (1500 ep) | +0.017 | −0.016 | 15.0% | 0.0034 | 1.136e9 |
| `lr=.001 γ=.5` (1500 ep) | +0.073 | +0.031 | 16.5% | 0.0062 | 1.012e9 |
| **`lr=.01 γ=.5` (1500 ep)** | **+0.683** | **+0.728** | **74.7%** | ~0.10 | **3.19e8** |

Read this table carefully:

- **More epochs alone does NOT fix it.** At `lr=0.001` the symmetry never breaks *no matter how long you
  run*. It is not "under-trained," it is the wrong exponent.
- **Fixing `gamma` alone barely helps** (r = 0.07).
- **`lr` is the dominant knob.** `0.001 → 0.01` is the difference between recovering nothing and recovering
  the truth, and the ELBO improves **3.7×** — so this is not a tie between two optima, it is a bad one and a
  good one.

**`gamma` compounds the problem.** `gamma=0.01` sets `lrd = gamma^(1/num_epochs)`, decaying the LR **100×
across the run** — killing it before the escape inflection (visible between epochs 500–1000). Use
`gamma = 0.5` (a 2× decay), not `0.01`.

## 11.3 Early stopping silently killed EVERY batch of the atlas training

`min_delta` is **absolute**. Against an ELBO of 1e6–1e9, a value of `0.01` (the default) or even `100` (what
the pipeline used) is a relative tolerance of ~1e-7 — **numerically inert**. The rule therefore collapses to
*"stop after `patience` epochs with no new global best"*, and SVI noise trips that almost immediately.

**Measured on the real SLURM logs (9,239 batch fits across 200 runs of the mouse atlas):**

```
median epochs actually run :   10       (nominal: 500 first pass / 1000 later)
86.3% of batch fits ran <= 20 epochs
 0.0% ever reached 1000 epochs
min / max                  :   6 / 547

'Early stopping'   lines   : 9,239
'Training completed' lines : 9,239      <-- 100%. NOT ONE BATCH RAN TO COMPLETION.
```

The nominal `num_epochs_first=500` / `num_epochs_later=1000` are fiction: **every chunk got ~10 SVI steps.**
The `gamma=0.01` decay never even engages (only a ~6% LR drop in 10 epochs before the optimizer is discarded).

Put that next to §11.2: at `lr = 0.001` the programs do not separate **even with 1,500 full epochs**. The
atlas models received **~10 steps per chunk at lr ≤ 0.001**.

Use the new **`min_delta_rel`** parameter (scale-free; effective threshold is
`max(min_delta, min_delta_rel · |best_loss|)`), set `patience` generously, and treat *any* early stop in the
first ~25% of `num_epochs` as a red flag. The package now warns when this happens.

## 11.3b The waypoint arm does not rescue this either — half of ψ starts frozen

Quantified on the actual training h5ad (132,706 cells × 79,768 junctions):

- `random_non_zero` (§5.2) keeps only `arr > 0`, but **58.6% of stored `junc_ratio` values are exactly 0.0**.
  Observed zeros are discarded as if missing. (The h5ad ships a `psi_mask` layer encoding *true* missingness.
  `waypoints.py` never uses it.)
- Of the values it can actually draw from, **51.7% are exactly 1.0**.
- `biject_to(unit_interval).inv(1.0)` saturates at **loc = +15.94**, where `sigmoid'(loc) ≈ 1.2e-7`.

**So roughly half the ψ matrix is initialized in the flat tail of the sigmoid and cannot move.** The mean of
the sampled pool is **0.735**, where the Methods' prescribed neighbourhood average would be **0.305** — the
init is boundary-saturated *and* biased ~2.4× high. Waypoints exist to break the symmetry; as implemented,
half of ψ is frozen before training starts.

Two more, from the same trace:

- ✅ **FIXED** (`fix: TruncatedSVD.fit_transform already returns U*Sigma; stop squaring it`). **The PC space was double-scaled.** `TruncatedSVD.fit_transform` **already returns `U·Σ`**;
  `run_training.py` then multiplies by `singular_values_` again, giving **`U·Σ²`**. Max-min sampling and
  nearest-neighbour assignment therefore run in a space where each PC is weighted by `σ²` instead of `σ`,
  collapsing waypoint diversity toward PC1 — the opposite of "chosen to capture major subspaces of variation".
- ✅ **FIXED** (φ is now solved via the normal equations, `phi = (rho @ psi) @ pinv(psi.T @ psi)`, so the
  dense transpose never exists). `initialize_psi_phi_with_random_non_zero` used to call `rho_hat.toarray()` on the **full** matrix — a dense
  132,706 × 79,768 float64 array = **85 GB**. This is why the SLURM job requests 300 GB, and it is why
  waypoints do not scale.

## 11.3c Other pipeline settings that contradict the Methods

Verified against `01_generate_params.py`, the configs, and the parameter JSON actually used on disk:

| | Methods | Actual |
|---|---|---|
| passes | 10 | **20** |
| batch size | 5,000 | **8,192** |
| epochs first / later | 50 / 20 | **500 / 1000 nominal — ~10 real** (§11.3) |
| random init | "fit 3 times, keep best ELBO" | **`train(num_initializations=1)` — never done** |
| nearest neighbours | 50 | **30** |
| PCs | 10 | **`min(K, 50)`** → 5/10/20/50 |
| learning rates | — | yaml has `[1e-4, 1e-3]`; **the runs on disk used `[1e-4, 1e-3, 1e-2]`** — the repo does not reproduce its own runs (64 vs 96 param sets) |

✅ **FIXED** (`fix: select on the best loss, not the last`). **`best_elbo` was the LAST loss, not the best.**
Under early stopping the last epoch is by definition a *non-improving* one (it is what tripped `patience`),
and it is a single noisy MC estimate. That number drives model selection across the whole grid. Now
`min(losses)`, with the final-epoch loss kept as `final_elbo`.

✅ **FIXED** (`fix: prune every factor-indexed array together; assert the factor axis agrees`).
**Pruning left the saved arrays misaligned:** `psi`, `assign_post`, `pi` were pruned but **`psis_loc`,
`psis_scale` were not** — and all were pickled, so saved `psis_loc` had `original_K` rows against `psi`'s
`pruned_K`. **`psis_loc`/`psis_scale` are exactly what the DS analysis samples from.** (`a`/`b` are
per-junction and correctly left alone.)

## 11.4 Diagnostics: the obvious ones lie

On the simulation above, the model had learned **literally nothing** (ψ r = 0.008, dominant agreement *below*
chance) while reporting:

```
dead programs (psi std < 0.01):  0 / 5     ← "healthy"
programs never dominant:         0 / 5     ← "healthy"
```

Those checks ask *"is each program alive and used?"* — and all of them were. They were just all pointing at
the same thing. **The question you need answered is "are the programs different from each other?"**

**Report these three on every run. They need no ground truth.**

| diagnostic | what it means | failure signature |
|---|---|---|
| `psi.std(axis=0).mean()` | do the K programs differ **from each other**? | ~0.003 = symmetric collapse; ~0.10 = separated |
| `phi.std(axis=0).mean()` | do the cells differ **from each other**? | ~0 = φ collapsed to uniform |
| epoch at which early stopping fired | was the run killed prematurely? | < 25% of `num_epochs` = red flag |

ψ-contrast tracked the failure perfectly across every config (0.0027 → 0.0034 → 0.0062 → ~0.10). It is the
single most informative number you can print.

## 11.4b The schedule sweep — RUN THIS, and its answer

Swept on `sim_heart_K5_bb20_eff1.0` (K=5, TRUE Beta-Binomial concentration = 20, effect = 1.0), random
init, 1200 epochs, early stopping disabled. Scored against **ground truth**. Chance dominant-program
agreement = 20%.

```
     lr   gamma   psi_r   agree  psi_contrast  learned_conc  healthy
  0.001    0.01  +0.052  15.2%        0.0056          13.0    False   <- THE SHIPPED DEFAULT
  0.001     0.5  +0.108  15.6%        0.0091           7.1    False
  0.005    0.01  +0.140  16.1%        0.0124           4.6    False
  0.005     0.5  +0.530  64.0%        0.0443           1.8    True
   0.01    0.01  +0.240  16.2%        0.0270           1.3    True
   0.01     0.5  +0.848  53.2%        0.0868           8.2    True
   0.05    0.01  +0.920  87.4%        0.0967          13.9    True
   0.05     0.5  +0.987 100.0%        0.0738          21.2    True   <- USE THIS
```

**Recommended: `lr = 0.05`, `gamma = 0.5`.** ψ correlation **0.987**, dominant-program agreement **100%**.
The pipeline's `lr = 0.001, gamma = 0.01` sits at **chance**.

### Two things this sweep revealed

**`bb_conc` is a convergence diagnostic, not just a nuisance parameter.** Read the `learned_conc` column
against the true value of 20. When ψ is wrong, the Beta-Binomial absorbs the unexplained residual by
**collapsing the concentration** (1.3, 1.8, 4.6). When ψ is right, it snaps to the truth (**21.2**). So a
`bb_conc` that lands near a plausible value is *itself* evidence the fit converged — and the old
`Gamma(10,10)` prior (mean 1, sd 0.32) was masking this by pinning it near 1 regardless. The new
`Gamma(2, 0.1)` prior (mean 20) recovers 21.2. **Corollary: do not report `bb_conc` as an estimate of assay
overdispersion unless ψ recovery is good — it is confounded with model misspecification.**

**The `psi_contrast` gate needed recalibrating.** At `lr=0.01, gamma=0.01` the contrast is 0.027 while
dominant agreement is **16.2% — below chance**. A 0.02 threshold let that through. `health_check` now uses
**0.04**, calibrated against the table above.

## 11.4b-ii THE LIKELIHOOD MATTERS MORE THAN THE LEARNING RATE — always Beta-Binomial

Swept on `sim_heart_K5_binom_eff1.0` — i.e. data where the **Binomial is the CORRECT model**. Random init,
1,200 epochs, `gamma=0.5`. Chance: agreement 20%, AUROC 0.500.

```
 MODEL      lr   psi_r   agree   AUROC   psi_contrast
 binom   0.005  +0.315  67.4%   0.635         0.0413
 binom    0.01  +0.619  70.5%   0.840         0.0923
 binom    0.02  +0.731  67.8%   0.890         0.1205
 binom    0.05  +0.758  45.1%   0.914         0.1256    <- the CORRECT likelihood
    bb   0.005  +0.523  61.7%   0.661         0.0453
    bb    0.01  +0.850  51.7%   0.978         0.0862
    bb    0.02  +0.966  75.2%   0.996         0.0902
    bb    0.05  +0.984  99.9%   0.998         0.0742    <- the "WRONG" one. FAR better.
```

### **The Beta-Binomial model beats the Binomial model EVEN WHEN THE TRUTH IS BINOMIAL.**

ψ r **0.984 vs 0.758**; dominant-program agreement **99.9% vs 45.1%**; AUROC **0.998 vs 0.914**. The
concentration parameter absorbs residual misfit and reweights observations, and it evidently *helps the
optimizer* — a strictly-more-flexible likelihood is easier to descend.

🔴 **ACTION: `input_conc_prior=None` (Beta-Binomial) for ALL training. Never `"inf"`.** Keep the Binomial
arm only as an ablation for the calibration figure, not as a candidate model.

### Two corollaries

**`lr` interacts with the likelihood.** Binomial (`conc = inf`) has a much sharper likelihood — larger
gradients — so an `lr` swept on Beta-Binomial is **not valid** for it. They must be swept together. (An
earlier revision of this document recommended `lr=0.05` on the strength of a Beta-Binomial sweep and it was
then applied to a Binomial model, producing near-chance recovery. Retracted.)

**`bb_conc` is a convergence diagnostic.** Read the learned concentration against a plausible value: when ψ
is wrong the Beta-Binomial **collapses** the concentration to absorb the unexplained residual; when ψ is
right it **recovers the truth** (21.2 against a true 20). A sensible `bb_conc` is itself evidence the fit
converged. *Corollary:* do **not** report `bb_conc` as an estimate of assay overdispersion unless ψ recovery
is good — it is confounded with model misspecification.

## 11.4b-iii `psi_contrast` IS NECESSARY BUT NOT SUFFICIENT

🔴 **A correction to §11.4.** `psi_contrast` reliably detects **total** symmetric collapse (~0.003 collapsed
vs ~0.10 recovered). It does **not** detect a **partial** failure. Observed:

```
--- model health: OK ---            <- PASSED a 0.04 gate
  psi_contrast : 0.0450
  actual: psi_r = 0.22,  dominant agreement 35.3%,  AUROC 0.525 (chance = 0.500)
```

`health_check` now warns in a **marginal band** (contrast between 1× and 2× the threshold), but the real
lesson is: **on the simulation, always score against ground truth; on the atlas, always score on held-out
data.** No single unsupervised statistic is a substitute.

## 11.4c EXPERIMENT — does chunked training recover the same posterior as full-batch?

*Reproduce: `python experiments/qpsi_width_under_chunking.py`*

### Why this experiment was necessary

The mouse atlas **cannot** be trained full-batch. The training matrix is 132,706 × 79,768 at **13.8%
density = 1.46 × 10⁹ nonzeros**; the `[nnz, K]` gather alone needs **~218 GB** against a 95 GB GPU
(~305 GB total). So training is **chunked**: a fresh `LeafletFA` per 8,192-cell chunk, warm-started from
the previous chunk's ψ, with `obs_scale = N/batch` so each chunk's objective is an unbiased estimate of the
**full-data** ELBO.

That scheme is only trustworthy if it lands where full-batch lands — and **on the atlas there is no
full-batch reference to compare against.** So we test it on a simulation small enough that full-batch *is*
feasible and where the truth is known.

### Design

- **Data:** 1,200 cells × 120 junctions, K=3, simulated from LeafletFA's own generative model
  (`φ ~ Dirichlet`, `ψ ~ Beta`, `y ~ Binomial(T, φψ)`), so **ψ_true is known**.
- **Chunking:** 4 chunks × 300 cells, 3 passes (12 chunk-fits), `obs_scale = N/B = 4`.
- **Schedule:** the swept-optimal `lr=0.05, gamma=0.5` (§11.4b), so we are measuring **the chunking**, not a
  bad learning rate. Early stopping off.
- **Reference:** full-batch on all 1,200 cells, 900 epochs.
- **Two things measured separately, because they behave completely differently:**
  1. the ψ **point estimate** — Hungarian-matched correlation with ψ_true (factors are identifiable only up
     to permutation);
  2. the ψ **posterior width** — `mean(psis_scale)`. *This is the posterior the DS/FDR samples from.*

### Results

```
REFERENCE  full-batch, 1200 cells, 900 epochs
           mean q(psi) width = 0.1731     psi_r vs truth = 0.997

arm                                          q(psi) width   vs ref   psi_r
------------------------------------------------------------------------------
1. chunked, scale reset each chunk (60 ep)         0.2652    1.53x   0.986
2. chunked, scale CARRIED forward   (60 ep)        0.4106    2.37x   0.986
3. chunked, scale reset            (200 ep)        0.3372    1.95x   0.993
3. chunked, scale reset            (600 ep)        0.3721    2.15x   0.992
```

### Three hypotheses tested. All three wrong.

**(a) "Carry `q(ψ)`'s width forward, so it accumulates evidence across chunks."**
→ **Makes it worse** (1.53× → 2.37×). Each chunk's optimisation *widens* from its narrow start
(0.1 → 0.27), so carrying that width forward **compounds the widening** rather than accumulating evidence.
The capability exists (`train(psi_scale_init=...)`) but `run_training.py` deliberately does **not** use it.

**(b) "The 47.6 GB dense Triton mask is gone, so full-batch should fit now."**
→ **It does not.** See the density above. Chunking is genuinely required.

**(c) "`obs_scale` already targets the full-data width; each chunk just needs more epochs to converge."**
→ **Backwards.** More epochs makes the width *worse* (1.53× → 1.95× → 2.15×).

### Why (c) fails — this is the load-bearing insight

`obs_scale` gets the **magnitude** right: scaling the log-likelihood by N/B multiplies the Fisher information
by N/B, so **one** chunk's ψ posterior *should* have the full-data width — and briefly it does.

But **each chunk sees different cells, and therefore disagrees with the others.** Cycling through chunks pulls
`q(ψ)` back and forth, and its width **inflates to cover the between-chunk disagreement**. Training each chunk
*harder* makes it pull *harder*, so the inflation **grows**.

**This is structural. No hyperparameter fixes it, because the chunks are fighting each other.** Only a single
guide seeing all the data through one optimizer avoids it — i.e. `pyro.plate(subsample_size=B)` with a
**persistent** guide, which is what the deleted `LeafletFAminibatch.py` had
(`git show 6f9ce6f:src/BetaDirichletFactor/LeafletFAminibatch.py`).

### ⚠️ CORRECTION — re-run on the REAL simulation, the width error FLIPS DIRECTION

The table above is from a small toy (1,200 cells, 120 junctions). Re-running the same comparison on the
**actual Figure-2 simulation** (`sim_heart_K5_binom_eff1.0`, 6,986 cells × 12,804 junctions, 7.3% density,
B=2,000 × 2 passes):

```
                        psi_r vs TRUTH   q(psi) width
full batch                       0.738         1.7967   <- reference
chunked (B=2000, x2)             0.646         0.2793   (0.16x)

chunked vs full-batch agreement (matched psi r): 0.881
psi_contrast   full = 0.1262     chunked = 0.1193
```

**The chunked posterior is ~6× NARROWER, not 1.5× wider.** The direction of the width error is **not
consistent** across settings.

**Therefore: do NOT claim the DS/FDR is conservative.** An earlier revision of this document said so, derived
from the toy. That was an over-generalisation from a single small simulation, and a *narrower* posterior would
push DS the other way (anti-conservative / over-calling). Retracted.

(The full-batch width of 1.80 should also not be taken at face value: a scale of 1.8 in logit space means ψ is
essentially unconstrained — `sigmoid(±1.8)` spans 0.14–0.86 — which looks more like the guide's scale drifting
than a genuine posterior.)

### The chunked–full-batch gap is UNDER-TRAINING, not a cost of chunking

Re-run on the real Figure-2 simulation, sweeping the number of passes (random init, `waypoints_use=False`
— deliberately the *harder* test; waypoints start you outside the saddle):

```
full batch (1500 ep)          psi_r vs truth = 0.774   psi_contrast = 0.1248

chunked                psi_r   vs full   agree w/ full   psi_contrast
B=2000, 2 passes       0.645     0.83x         0.835        0.1193
B=2000, 4 passes       0.701     0.91x         0.855        0.1271
B=2000, 8 passes       0.727     0.94x         0.838        0.1283
```

**Chunked converges to full-batch as passes increase** (83% → 91% → **94%**), and `psi_contrast` converges to
the same value (0.128 vs 0.125). The production pipeline uses **`num_passes = 20`**, so it should land
essentially on top of full-batch.

**Conclusion: batching works.** The earlier "12% gap" was an artefact of running only 2 passes.

### Conclusion — and what we accept

| | verdict |
|---|---|
| **ψ, φ, π point estimates** | ✅ **Fine, and this replicates on the real simulation.** Chunked-vs-full-batch agreement **r = 0.881**; `psi_contrast` 0.126 vs 0.119. Everything Figs. 3/4/6 rest on is sound. |
| **`q(ψ)` posterior width** | 🔴 Does **not** match full-batch, and **the direction is inconsistent** (1.5× wider on a toy, 6× narrower on the real sim). Not currently characterised. |
| **Consequence for DS** | **Do not argue from the posterior width at all.** Use the **direct empirical FPR** instead (§ FDR CALIBRATION): on a pure null the pipeline calls **0 / 4,268 ATSEs, FPR = 0.0000**. That is measured end-to-end on data where the answer is known, needs no assumption about `q(ψ)`, and is exactly what Reviewer 1 asked for. **Lead with that.** |

**Free win, and it is counterintuitive: use FEWER epochs per chunk.** 60 is both **faster** *and* gives the
**least-inflated** posterior. When early stopping is fixed (§11.3), do **not** let `num_epochs_later` run to
1000 — cap it at **50–100**.

**Decision: accept.** We keep the chunked scheme, cap the epochs, and state the conservatism in the Methods.
The alternative (persistent guide + `AutoNormal(create_plates=...)` + a subsample-indexed `model()`) is
~150 lines of rewrite that risks new bugs before a resubmission, to buy a *less conservative* FDR.

✅ **Guarded permanently:** `test_chunked_training_recovers_the_same_psi_as_full_batch` fits the same
simulated data both ways and asserts they agree (r > 0.7) and both recover the truth. Plus
`test_obs_scale_is_what_makes_chunking_unbiased` pins the local-vs-global scaling. The minibatch scheme
cannot silently drift again. (`test: pin chunked training against a full-batch reference`)

## 11.4d 🔴 THE ψ PRIOR IS A SHRINKAGE PRIOR TOWARD THE SADDLE — and it was never tested

This is the largest single finding of the review, and it was found last, on 2026-07-13, by asking a
question nobody had asked: **what does `junc_specific_prior=True` actually do to the K programs?**

Look at the prior again:

```
ψ_kj ~ Beta(a_j, b_j)
```

**`a_j` and `b_j` are indexed by JUNCTION ONLY.** There is no `k`. So at every junction, all K programs are
drawn from **the same Beta distribution**, whose parameters are *learned from the data*. That is not a
neutral regularizer. It is, literally, a **shrinkage prior pulling the K programs toward each other at every
junction** — and "the K programs are all the same" is precisely the symmetric saddle of §11.1, the model's
main failure mode. The prior is pulling *toward* the thing training has to escape.

Worse, it is an *empirical-Bayes* layer (§4.2): `a_j`/`b_j` are learned. Early in training, when the
programs genuinely are all near each other, the model fits a **sharp** `Beta(a_j, b_j)` around their common
value — which then pulls them together *harder*. It is a positive feedback loop into the saddle.

A **global** `Beta(a, b)` (one `a`, one `b`, shared across all junctions *and* all factors) has no such
per-junction pull: it only says "ψ values are, overall, distributed like this", which is a statement about
the marginal, not a force between programs at a junction.

**Measured** (`experiments/junction_prior_shrinks_programs.py`, Beta-Binomial, lr=0.05, γ=0.5, random init):

```
 psi prior           seed   psi_r   psi_contrast          loss
 junction-specific     0   +0.395      0.1800      9.1559e+04
 junction-specific     1   +0.158      0.0690      4.2330e+05
 junction-specific     2   +0.386      0.1683      6.5399e+04
 global                0   +0.994      0.2648      5.0664e+04
 global                1   +0.994      0.2588      5.0656e+04
 global                2   +0.993      0.2563      5.0653e+04
```

Three things, each independently important:

1. **The global prior recovers the truth (r ≈ 0.99); the junction-specific one does not (r ≈ 0.31).**
2. **The global prior REMOVES THE SEED VARIANCE.** 0.993–0.994 across three seeds, versus 0.158–0.395. The
   run-to-run instability documented in §11.4b — the thing that made "fit three times and keep the best
   ELBO" necessary in the first place — **looks like a symptom of this prior, not an intrinsic property of
   the model.** If that holds up, the restart protocol is treating the fever, not the infection.
3. **The ELBO prefers the global prior** (5.07e4 vs 6.5e4–4.2e5), so best-ELBO model selection would have
   found this — if anyone had ever put both priors in the same comparison. Nobody did.

### Status: NOT settled — being tested on the atlas

One simulation is not enough to flip a modelling choice the paper is built on, so:

- `junc_specific_prior` was **promoted from an ablation to a CORE axis** of the atlas grid
  (`01_generate_params.py`). The 70% training array is **K(3) × prior(2) × init(2) × seed(3) = 36 core
  runs**, so the held-out 30% decides it. *(commit: `make the psi prior a core axis — it may be the cause
  of the seed variance`)*
- A confirmation on the full Figure-2 simulation (6,986 cells × 12,804 junctions) is running.
- **If it reproduces, the atlas models must be refit with the global prior**, and §4.2 / the parameter table
  must change. The junction-specific prior would then be reported as an ablation that *fails*, which is a
  perfectly good result — and a much better explanation of SVI local optima than "we used restarts".

**Do not report the junction-specific prior as the model without checking this.**

## 11.5 Recommended protocol before committing to a full-scale run

1. **Simulate first, with known truth.** Generate a small dataset from the generative model (or reuse
   `sim_heart_K5_*`), and confirm you recover ψ and φ. If the model cannot recover truth on data it
   *generated itself*, no atlas result is interpretable.
2. **Sweep `lr` × `gamma` on that simulation**, not on the atlas. It is cheap and it is the only place you
   can score against the truth. Start from `lr ∈ {0.005, 0.01, 0.05}`, `gamma ∈ {0.1, 0.5, 1.0}`.
3. **Check the three diagnostics in §11.4 on every candidate**, and reject any run with ψ-contrast near zero
   *even if its ELBO looks fine* — the collapsed solution has a perfectly respectable-looking loss curve.
4. **Then scale up**, carrying the chosen schedule.
5. **Prefer waypoint init**, which starts you *outside* the saddle so escape is immediate rather than taking
   1000+ epochs. But note (§5.2) that the waypoint initializer currently seeds ψ from **one random cell's
   PSI**, not the neighbourhood average — fix that first, or its symmetry-breaking is weaker than intended.
6. **Report that random init also works** once the schedule is right. "Programs are recovered from random
   initialization; waypoints simply converge faster" is a much stronger robustness claim than "we needed
   waypoints", and it directly answers the local-optima objection.

## 11.6 Other things that silently degrade a fit

- **Concentration prior is too tight** (§4.3). With a true concentration of 20, the model recovered 11.75.
  All three candidate priors have mean 1; the parameter lives on a scale of 10–100.
- **Beta-Binomial vs Binomial matters** (measured, K=5, effect=1.0):

  | truth | model | ψ r | φ r | dominant agree | AUROC |
  |---|---|---|---|---|---|
  | Binomial | Binomial | +0.684 | +0.729 | 74.6% | 0.870 |
  | Binomial | Beta-Binomial | +0.890 | +0.634 | 46.4% | 0.990 |
  | Beta-Binomial(20) | Binomial | +0.649 | +0.619 | 61.1% | 0.849 |
  | **Beta-Binomial(20)** | **Beta-Binomial** | **+0.908** | **+0.802** | **93.0%** | **0.985** |

  Beta-Binomial wins on ψ recovery and AUROC under *both* truths, which justifies it as the default. But
  under a Binomial truth it still learns a *finite* concentration (24), injecting spurious noise that blunts
  the per-cell likelihood and smears φ (agreement drops to 46%) even while global ψ stays excellent.
- **The dense/sparse switch.** `detect_data_density` flips to a dense code path above 30% density. Bulk data
  (~65–100% dense) takes the dense path; single-cell (~1–7%) takes the sparse one. Same maths, but remember
  that a result validated on bulk has **not** been validated in the sparse regime.
- **Environment.** LeafletFA needs a PyTorch build with kernels for your GPU's compute capability. On an
  RTX PRO 6000 Blackwell (sm_120), torch 2.4.1 fails every `lgamma` on CUDA with an opaque kernel-JIT error.
  Use torch ≥ 2.11 (`sm_120` in `torch.cuda.get_arch_list()`).
