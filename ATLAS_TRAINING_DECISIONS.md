# Atlas training: what was decided, on what evidence

Running record for the DM + simplex atlas fits. Every entry states the measurement behind the
choice, so the Methods section can cite evidence rather than inherited defaults. Dated 2026-08-04/05.

Companion documents: `METHODS_dirichlet_multinomial.md` (the model), and the memory notes
`project_leafletfa_ds_test_lrt` / `project_leafletfa_atlas_divergence`.

---

## 1. Settled by measurement

| choice | value | evidence |
|---|---|---|
| likelihood | Dirichlet-multinomial | log-prob verified term-for-term against LeafCutter's `dm_glm.stan` (−114.362812 vs −114.362793) and SpliceVI's independent implementation |
| psi parameterisation | within-ATSE simplex | psi sums to 1 on 100% of 24,475 atlas ATSEs, max deviation 1.1e-3 |
| concentration | one per ATSE | learned values track coverage (log-conc vs log-depth r = +0.56); m=2 events sit at 0.297 vs ~1.06 for m≥3; range 0.012–4,750, which a single scalar cannot represent |
| alpha floor | 1e-2 | the DM gradient goes as 1/a; the old 1e-6 floor sat 190× below where alpha lands and never fired. At 1e-2, max &#124;dL/da&#124; falls 5,249 → 96.9, and the log-prob is bit-identical in the ordinary regime |
| early stopping | OFF | `min_delta_rel=1e-5` against an ELBO of 4.7e9 is a 47,000 threshold while minibatch noise is ±5e7 — 1000× larger. It cut a fit at 130 epochs of a 1,000-epoch budget |
| batch size | 8192 | earlier convergence study: batch sets a recovery ceiling; largest feasible |
| DS test | per-ATSE likelihood-ratio test, phi as design matrix | realised FDP at or below nominal at all four effect sizes × 3 seeds; power 0.84–0.99. The previous fixed-δ TV rule was 2.9× over nominal at α=0.05 |

### Three training bugs found and fixed (all with regression tests, sections 13–15)

1. **The alpha floor never reached the model.** `minibatch` passed `eps=self.eps` (1e-6). `self.eps`
   also guards the Beta prior, where 1e-2 would be wrong, so `dm_alpha_floor` is now a separate knob.
2. **The NaN-guard rollback silently skipped every constrained parameter.** `param_store[name]`
   returns `transform_to(constraint)(unconstrained)` rebuilt on each access, so snapshotting
   `.items()` and `copy_()`-ing back writes into a temporary. AutoNormal holds every scale under
   `constraints.positive` — fits were SAVED with `psi_loc` finite and `psi_scale` all-NaN while the
   guard reported a successful restore. Now snapshots `get_state()["params"]` and restores in place
   (`set_state()` rebinds tensors and orphans the optimiser's references).
3. **Restoring parameters without clearing the optimiser state does nothing.** Adam's `exp_avg`
   buffers stay NaN and re-poison the restored values in one step. Measured before the fix: guard
   fired at steps 2016, 2018, 2020, 2022, 2024, 2026, 2028.

Also: **wandb was silently off** for every run (protobuf 7.35.1 vs wandb 0.19.8, exception
swallowed by a bare `except`). Upgraded to 0.28.1; the stack still imports and syncing is confirmed.

---

## 2. The divergence

Every atlas fit reached a non-finite loss at a **reproducible** step: K=20 seed 1 at 2016 in two
independent runs, K=20 seed 2 at 1942, K=30 seed 0 at 1513, K=30 seed 1 at 1491.

**Trigger: the learning rate.** Found because a repro with `num_steps=2200` did *not* diverge while
production at 16,200 did, with everything else identical. `lrd = gamma**(1/num_steps)`, so num_steps
changes the *schedule*, not just the budget — at step 2016 the surviving run was at lr 0.0265 and
production at 0.0459. Matching production's schedule reproduced the failure at step 2028.

**Mechanism: phi at the simplex corners.** phi is Dirichlet-distributed and AutoNormal
parameterises it through a transform whose gradient scales as 1/phi. On the atlas, **19 of
1,327,060 phi entries sit below 1e-6**, and **exactly 18 phi entries were non-finite at the
divergence** — the same handful of cells. They carry gradients of order 1e6 while every other cell
is ordinary. phi enters the shared loss, so one tipping over poisons every global parameter in the
same step. This is why lowering lr *delays* the divergence without removing it.

Caveat on ordering: the diagnostic fires after `svi.step()` has already applied the update, so all
gradients read 0 at that point. The 18-vs-19 correspondence is strong evidence, not proof; a
backward hook before the optimiser step would settle it.

**Ablation matrix** (K=20 seed 1, the deterministic reproducer, restarts disabled so the first hit
is the verdict, 3000 steps): baseline / phi-floor 1e-4 / clip-norm 1.0 / psi-clip 1e-3 /
global-conc / batch-4096. *Results pending — to be filled in below.*

---

## 3. Convergence: the ELBO cannot answer it

The atlas ELBO drops 11.25% and **essentially all of it lands in the first ~50 steps**. After that
the tail slope flips sign between fits (−418, +506, −500, −147, +1078, +6528), i.e. it is
noise-dominated. Two consequences:

- Do not read the atlas loss curve as a convergence diagnostic. It looks flat because the data term
  is ~4.7e9 and minibatch noise is ±1%, not because the model has stopped learning.
- A correction to an earlier reading in this project: an initial analysis reported "<1% total
  movement", which was an artefact of averaging the first 1,620 steps as the baseline — a window
  that sits entirely *after* the descent.

**What replaces it:** `fit(track_every=N)` now records `psi_contrast`, `phi_spread`, `pi`,
`conc_median`, `n_live` and the loss on a fixed interval, saved into `fit.npz` and streamed to
wandb. Whether the *structure* has stopped moving is the real criterion.

---

## 4. Atlas QC, on real data

- psi sums to 1 within every ATSE, and imputed usage sums to 1 per (cell, ATSE): 100% valid.
- **Imputation is denoising, not reconstruction.** r(model, observed) by read depth: 0.792 at 1–2
  reads → 0.916 at 50+. The pooled 0.849 is dragged down by the largest bin being the noisiest
  (8.35M entries at 1–2 reads). A model scoring 1.0 at 1–2 reads would be fitting shot noise. This
  is training-set reconstruction; held-out imputation is the separate A3 benchmark.
- **Effective K is 5–8 regardless of nominal K** (9/10 live at K=10, 5–8/20, 9/30). Extra programs
  carry no cells and converge to a common average composition.
- **Seed agreement**, on the two fits that both ran the full 16,200 steps under identical code:
  every live program in seed 0 recovered in seed 2 (7/7) at r ≥ 0.95; psi 0.973, phi 0.740,
  top-2000 junction overlap 0.703. Programs live in *both* seeds match near-perfectly; 62–88% of
  live programs are shared. A naive "100% reproduced" is inflated by dead programs matching each
  other at r ≈ 0.93, so it should be quoted live-only.
- phi remains the degenerate direction (0.740 vs psi's 0.973). The simplex fixed psi, not the
  per-cell encoding.

---

## 5. Decisions taken today, and why

- **K grid stays 10/20/30.** Effective K is 5–8, so the sweep is a robustness demonstration rather
  than model selection — there is no principled K-selection criterion in place. K=20 had the best
  ARI vs held-out cell type (0.39–0.41 vs 0.30 at K=10).
- **lr is chosen on performance, not on matching the previous recipe.** lr=0.05/gamma=0.5 is
  validated on simulations (6,986 cells) but diverges on the atlas (132,706 cells).
- **Decay matters more than it looks.** With gamma=0.5 over 16,200 steps, lr has moved only
  0.050 → 0.046 by the divergence at step 2016. gamma=0.01 puts it at 0.0282 there — below the
  0.0459 that dies, just above the 0.0265 that survived — while keeping the high early rate that
  escapes the initial saddle. (The project's earlier "gamma=0.01 kills training" finding was at
  lr0=0.001, a different regime.)
- **Production is NOT auto-launched on an a-priori lr.** An auto-launcher was written and then
  deliberately restricted to firing the lr/gamma sweep only, because launching 9 fits on a guess
  wastes a night if the guess is wrong. Sweep arms: (0.05, 0.5), (0.05, 0.01), (0.025, 0.5),
  (0.025, 0.1), 4,500 steps each, scored on psi_contrast / live programs, plus any stability knob
  the ablations vindicate.
- **Comparability.** Last night's 9 fits ran MIXED CODE: jobs in one sbatch batch start as GPUs
  free and Python imports at process start, so fits launched from one command straddled a commit
  and picked up different `max_nan_restarts`. Those seven partial fits are not comparable across K.
  All production fits must be launched from a single commit.

---

## 6. Open

- Whether chi-square holds at **m=2**, which is 44% of atlas ATSEs and cannot be tested on the
  cassette-exon simulator (it emits m=3 only). Being measured on real data via permuted phi, split
  by m — no truth labels needed. If it fails, the fix is local: chi-square for m ≥ 3, empirical
  calibration for m=2.
- The exact origin of the divergence (the 18-vs-19 phi correspondence is evidence, not proof).
- phi reproducibility at 0.74.
- No principled K-selection criterion.

---

## 7. Log

**Ablation v1 was invalid — my own error, same one it was built to diagnose.**
The arms were submitted with `--steps 3000`, and `lrd = gamma**(1/num_steps)`, so num_steps sets
the SCHEDULE. At step 2016 those arms sat at lr 0.0314 against production's 0.0459 — already below
the threshold that kills production. Baseline duly sailed past 2016 without diverging, which is the
tell. Relaunched with `--steps 16200` (production's schedule) and bounded by wall-clock instead:

| num_steps | lr at step 2016 | outcome |
|---|---|---|
| 2,200 | 0.0265 | survives |
| 3,000 | 0.0314 | survives (the invalid ablation) |
| 16,200 | 0.0459 | DIVERGES |

**clip_norm 1.0: rejected.** Diverged at step 1023 — *earlier* than baseline's 2016, and on the
safer schedule at that, so it is worse under any schedule. Informative beyond its own arm: if the
problem were simply gradient magnitude, tighter clipping should have helped. It made things worse,
which argues the issue is where the gradients POINT — a handful of corner cells dominating the
update direction — rather than a global explosion.

**Auto-launcher restricted to the sweep.** It was originally going to submit the 9 production fits
on an a-priori lr choice. With a live session that is the wrong trade: nine 4.7-hour fits on a guess
wastes a night if the guess is wrong. It now fires only the lr/gamma sweep, and production is
launched from the measured result (`A47_score_lr_sweep.py`, which ranks on psi_contrast and live
programs rather than on whether the run survived).

---

## 8. m=2 / chi-square on the real atlas, and a correction

**The permuted-phi diagnostic on the atlas (K=20 seed 0, 20,000 cells):**

| nominal | observed | ratio |
|---|---|---|
| 0.50 | 0.543 | 1.09 |
| 0.20 | 0.311 | 1.55 |
| 0.05 | 0.164 | 3.27 |
| 0.01 | 0.097 | 9.72 |

By event size the ratio at 0.05 is ~3 at EVERY m (2.95 at m=2, 3.93 at m=3, ~3.1 for m=4-8, ~2.5 at
m=9-11). **So this is not the m=2 problem it was expected to be** -- the earlier framing that "m=2
is where Wilks breaks" was wrong; m=2 is among the better-behaved sizes.

**First reading: chi-square is 3x anti-conservative on real data. That reading was wrong.**

The permuted-phi null is only a valid null on TRULY NULL events. Permuting phi on a genuinely
differential ATSE does not null it -- it converts real between-program variation into an
overdispersed mixture, which inflates the null. On simulation we can measure both sides:

| effect | permuted-null ratio @0.05 | REALISED FDP ratio @0.05 (vs truth) |
|---|---|---|
| 0.25 | 0.71 | 0.90 |
| 0.50 | 1.70 | 0.54 |
| 1.00 | 3.82 | 0.20 |

A permuted ratio of 3.82 coexists with a realised FDP of 0.20x nominal, i.e. strongly CONSERVATIVE.
The atlas's 3.27 sits inside that range, so it is consistent with chi-square being conservative
there too, with the diagnostic inflated by real signal. The permuted-null ratio is therefore an
upper bound on trouble, not a measurement of it, wherever real signal exists.

**Empirical calibration (A48) was built, validated, and REJECTED.** Replacing the chi-square
reference with the permuted null's own quantiles, stratified by event size (df = (m-1)*K, so strata
cannot be pooled), makes calibration WORSE because the null it calibrates against is the
contaminated one:

| effect | alpha | chi2 FDP ratio | empirical FDP ratio | chi2 power | emp power |
|---|---|---|---|---|---|
| 0.25 | 0.05 | 0.90 | 0.93 | 0.845 | 0.845 |
| 0.50 | 0.05 | 0.54 | 0.03 | 0.957 | 0.931 |
| 0.50 | 0.01 | 0.56 | 0.00 | 0.942 | 0.883 |

At effect 0.5 it is wildly over-conservative and gives up real power for no measured gain.

**DECISION: keep the chi-square reference.** Report the permuted-null table as a diagnostic with the
caveat above. A48 is kept in the repo as a documented negative result, not as the shipping path.

---

## 9. Ablation results: every structural knob REFUTED, lr is the only lever

Run on the deterministic reproducer (K=20 seed 1) with `--steps 16200` so the schedule matches
production, restarts disabled so the first hit is the verdict.

| arm | diverged at step | vs baseline |
|---|---|---|
| baseline | 2,028 | — |
| **phi-floor 1e-4** | **2,028** | identical |
| psi-clip 1e-3 | 2,016 | no better |
| global-conc | 2,016 | no better |
| batch 4096 | 1,703 | worse |
| clip-norm 1.0 | 1,023 | much worse |

**The phi-corner mechanism is refuted.** Flooring phi at 1e-4 produced a byte-identical step count,
so the 18-non-finite-vs-19-below-1e-6 correspondence was a consequence or a coincidence, not a
cause. The flag was verified to reach the model (final losses differ, `self.phi_floor` set), so this
is a real negative and not a plumbing failure.

Nothing structural moves it. Only the learning rate does:

| lr at the danger zone | outcome |
|---|---|
| 0.0459 (lr0=0.05, gamma=0.5, 16200 steps) | diverges at 2016-2028 |
| 0.0314 (same, but 3000-step schedule) | survives |
| 0.0265 (same, 2200-step schedule) | survives |
| lr0=0.025 | first hit at 3,432 |
| lr0=0.0125 | clean to at least 3,400 |

**The num_steps trap, caught three times.** `lrd = gamma**(1/num_steps)`, so num_steps sets the
SCHEDULE, not just the budget. It invalidated (1) the first repro, which failed to reproduce; (2)
ablation v1, submitted with `--steps 3000`; and (3) the first lr sweep, submitted with
`--steps 4500` -- where every arm would have sat below the divergence threshold and the sweep would
have taught nothing. **Any run being compared to production must use `--steps 16200` and be bounded
by wall-clock instead.**

Two operational notes for whoever reads the logs:
- Slurm did not flush the ablation `.out` files, so `result.json` in each run directory is the
  reliable source. An auto-launcher that parsed `.out` concluded "no cure found" from NO DATA; it
  should check for file existence before deciding.
- That auto-launcher also had an unbound-variable bug (`IDS` referenced after the production block
  it defined was removed), so the QC job it was supposed to chain never got submitted.

### The cells-scaling argument that closes it

The permuted-phi diagnostic was run at two cell counts. If chi-square were genuinely breaking down,
MORE cells should bring the ratio closer to 1 -- Wilks is asymptotic in sample size. It went the
other way:

| m | ratio@0.05, 20,000 cells | ratio@0.05, 60,000 cells |
|---|---|---|
| 2 | 2.95 | 3.50 |
| 3 | 3.93 | 5.88 |
| 5 | 3.06 | 6.14 |
| 8 | 3.17 | 6.21 |
| 11 | 3.26 | 6.51 |

Roughly doubling with 3x the cells is the signature of CONTAMINATION, not of a failing
approximation: with more cells the real differential signal is detected more strongly, so permuting
phi converts more real between-program variation into an overdispersed mixture and inflates the
"null" further.

Two independent lines now agree that chi-square is sound here:
1. on simulation, a permuted ratio of 3.82 coexists with a realised FDP of 0.20x nominal;
2. on the atlas, the ratio scales WITH sample size instead of against it.

**m=2 is not a special case and the DS test needs no per-m treatment.**

---

## 10. lr / gamma sweep, and the production launch

All arms on the production schedule (`--steps 16200`), K=20 seed 1, restarts enabled (20) so an arm
that takes a hit can still show what it learns. Bounded at 2h wall-clock.

| arm | step reached in 2h | restarts used | first NaN |
|---|---|---|---|
| lr 0.05, gamma 0.5 | 4,279 (exhausted) | 21 | 2,028 |
| lr 0.05, gamma 0.01 | 5,654 (exhausted) | 21 | 2,701 |
| lr 0.025, gamma 0.5 | 12,100 | 8 | 3,432 |
| **lr 0.025, gamma 0.1** | **12,100** | **6** | 4,711 |
| lr 0.0125, gamma 0.5 | 6,800 | **0** | none |

- **lr 0.05 is not viable at either gamma** -- both exhausted the full 20-restart budget before a
  third of the run. Faster decay helps (first hit 2,028 -> 2,701, and 32% further overall) but does
  not rescue it.
- **lr 0.025 with gamma 0.1 completes**: 12,100 of 16,200 steps inside 2h using 6 of 20 restarts.
- **lr 0.0125 took zero restarts**, but ran on a slower node so its step count is not comparable.
- **Lowering lr costs nothing in structure**: both lr=0.05 arms gave psi_contrast 0.1087 and 7 live
  programs, identical to each other.

**DECISION: production at lr0=0.025, gamma=0.1.** One extra arm at lr=0.0125 (K=20 seed 0) is run
alongside so the choice stays checkable against a cleaner-but-slower rate rather than asserted.

**Sweep design flaw, recorded.** `--time 2:00:00` KILLED the three lower-lr arms before they could
write `result.json`, so the arms that mattered produced no saved metrics; only the two lr=0.05 arms
appear in the results because they finished early by exhausting restarts. The step counts above were
recovered from the stdout logs. Any future sweep must either set a wall-clock generous enough to
finish or checkpoint metrics periodically rather than only at exit.

### Production launch (commit 1aead70)

9 fits: K in {10, 20, 30} x seeds {0, 1, 2}, 16,200 steps (1,000 epochs), batch 8192, lr0 0.025,
gamma 0.1, early stopping OFF, per-ATSE concentration, alpha floor 1e-2, max 20 NaN restarts,
trajectory tracked every 200 steps, wandb streaming. All launched from a SINGLE commit -- the
previous batch straddled a commit and picked up different restart budgets, which made those fits
non-comparable across K.

### Production results (lr0=0.025, gamma=0.1, commit 1aead70)

**7 of 9 fits completed the full 16,200 steps** -- the first time any atlas fit has finished.

| fit | steps | restarts | psi_contrast | live | settled/interval |
|---|---|---|---|---|---|
| K10 s0 | 16,200 | 1 | 0.0720 | 8/10 | 0.075% |
| K10 s2 | 16,200 | 1 | 0.0686 | 9/10 | 0.209% |
| K20 s0 | 16,200 | 4 | 0.1062 | 10/20 | 0.152% |
| K20 s1 | 16,200 | 8 | 0.0985 | 7/20 | 0.129% |
| K20 s2 | 16,200 | 3 | 0.1114 | 11/20 | 0.044% |
| K30 s0 | 16,200 | 10 | 0.1153 | 9/30 | 0.049% |
| K30 s1 | 14,974 | 21 | 0.1195 | 11/30 | 0.075% |
| K30 s2 | 16,200 | 6 | 0.1253 | 9/30 | 0.358% |
| K20 s0 @ lr 0.0125 | 16,200 | 2 | 0.1121 | 8/20 | 0.371% |

psi_contrast settled to 0.04-0.37% per interval in every fit -- the convergence evidence the ELBO
cannot supply. **Live programs rose to 7-11**, against 5-8 in the truncated fits, so the earlier
"effective K is 5-8" reading was partly an artefact of fits dying at 20-50% of budget.

Seed consistency (all K, completed fits): psi r 0.93-0.96 with the worst-matched program still at
0.80+, top-2000 junction overlap 0.75-0.80, phi 0.59-0.85. psi sums valid at 100% everywhere.

### Is the loss curve "not dropping much" a problem? No.

The ELBO falls 11.25% from step 0, nearly all in the first ~50 steps, which looks alarming without a
reference. Measured against proper bounds on 4,000 cells, with the same per-ATSE concentration:

| model | log-likelihood |
|---|---|
| null (one composition for all cells) | -143,937,600 |
| **cell-type pooled (38 known labels)** | **-142,066,240** |
| K=20 unsupervised programs | -142,096,176 |
| saturated (each cell's own composition) | -133,825,648 |

**The model captures 98% of what perfect cell-type labels would buy, using no labels at all.**

And the total explainable signal is genuinely small: null -> perfect labels is 1.87M out of 144M,
about 1.3%. So a small ELBO drop dominated by the first 50 steps is the expected shape -- most of it
is the model finding the overall composition, with program structure a thin layer on top.

Do NOT use "% of saturated" as the yardstick (it reads 18.2%): the saturated model fits each cell's
own shot noise, and 8.35M of the entries have 1-2 reads. The cell-type bar is the honest denominator.

### Known gap: a timeout loses the entire fit

K10 seed 1 was killed at the 8h wall at step 10,000 with ZERO restarts -- not diverging, just placed
on a node running 2.9 s/step against its siblings' 0.6 s/step. Because the fit is written only at
exit, 8 hours of clean training was lost. Relaunched with a 24h limit. The durable fix is periodic
checkpointing of psi/phi during training rather than a generous wall-clock.
