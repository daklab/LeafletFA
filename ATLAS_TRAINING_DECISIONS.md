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
