"""Proper minibatch SVI for LeafletFA using a real `pyro.plate` over cells.

WHY THIS EXISTS
---------------
The original model folds the per-cell factor loadings into a single site::

    assign = pyro.sample("assign", dist.Dirichlet(pi*dir_conc).expand([N]).to_event(1))

`.expand([N]).to_event(1)` makes all N cells ONE event. There is no plate, so Pyro cannot
subsample, and the analysis pipeline works around it by rebuilding a fresh LeafletFA (and a fresh
guide) on every 5-8k-cell chunk -- 96 rebuilds for the mouse atlas. Two consequences, both measured
on the 2026-07-18 atlas grid:

  * q(psi)'s scale is re-initialised to AutoDiagonalNormal's default width on every chunk, so the
    posterior never accumulates evidence across the data.
  * psi initialises correctly at the coverage-aware mean (~0.305), jumps to ~0.71 within the first
    pass, and stays at 0.63-0.72 for all 8 passes -- in every model, at every K, under both
    waypoint and random init. The likelihood measurably prefers psi ~ 0.31 (worth ~1.09 nats per
    observed entry, ~1e9 nats over the training set), and the psi prior can supply at most ~1e7
    nats of counterforce. So the objective wants low psi and the optimiser never gets there.

`AutoDiagonalNormal` cannot fix this: it concatenates every latent into a single flat vector, so it
has no plate structure and does NOT support local variables under subsampling. `AutoNormal` keeps
one distribution per site and accepts `create_plates`, which is what makes subsampling work.

WHAT THIS DOES
--------------
One model, one guide, alive for the whole run. `pyro.plate("cells", N, subsample_size=B)` declares
the cells conditionally independent and Pyro:
  * draws the subsample indices ONCE and uses the same ones in model and guide,
  * scales the local terms by N/B automatically -- no manual poutine.scale bookkeeping,
  * leaves the global terms (psi, pi, a, b, dir_conc, bb_conc) at weight 1, which is correct.

Local variational parameters stay full-size (N x K) and are indexed by the plate. No amortisation
is needed: at N=92,894 and K=50 that is ~37 MB for loc and scale together.

Generative model (matches METHODS.md):
    a_shape, a_rate, b_shape, b_rate ~ Gamma(1, 1)          [only if learn_ab_hyperprior]
    a ~ Gamma(a_shape, a_rate);  b ~ Gamma(b_shape, b_rate) [scalar, or one per junction]
    psi[k, j] ~ Beta(a, b)
    alpha_pi ~ Gamma(1, 1);  pi ~ Dirichlet(alpha_pi / K)
    dir_conc ~ Gamma(1, 1)
    bb_conc  ~ Gamma(2, 0.1)
    phi[c]   ~ Dirichlet(dir_conc * pi)                     [inside the cells plate]
    y[c, j]  ~ BetaBinomial(n=clus[c, j], mu=phi[c] @ psi, conc=bb_conc)   masked to clus > 0
"""

from __future__ import annotations

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_value
from pyro.optim import ClippedAdam

from .dirichlet_multinomial import (build_atse_index, dm_log_prob, segment_softmax,
                                    _segment_sum)


class LeafletFAMinibatch:
    """Beta-Dirichlet splicing factor model with true minibatch SVI over a cells plate.

    Parameters
    ----------
    junc, clus : (N, J) array or torch tensor
        Per-cell junction counts and their ATSE (cluster) totals. `clus > 0` is the coverage mask;
        an observed PSI of 0 is DATA, not missing, so coverage must come from clus, never from
        `junc > 0`.
    K : int
        Number of programs.
    junc_specific_prior : bool
        If True, a and b are per-junction vectors; otherwise scalars shared by all junctions.
    learn_ab_hyperprior : bool
        If True, a/b get the learned Gamma hyperprior level the original model uses. METHODS.md
        specifies a, b ~ Gamma(1, 1) with NO extra level, and the learned level is what drives the
        prior mean to ~0.70 on real data, so this defaults to False.
    phi_init : (N, K) array, optional
        Simplex-valued starting point for the per-cell loadings. Applied by writing the guide's
        location parameter directly rather than through `init_to_value`, which cannot accept a
        full-size value for a site under a subsampling plate.
    psi_init : (K, J) array, optional
        Coverage-aware mean initialisation. Clipped away from 0/1 -- Pyro maps a Beta site through
        a logit and psi_init = 1.0 becomes a loc of +15.9 where the sigmoid gradient is ~1e-7, i.e.
        frozen for the whole run.
    """

    def __init__(self, junc, clus, K, *, junc_specific_prior=False, learn_ab_hyperprior=False,
                 psi_init=None, phi_init=None, init_strategy="feasible",
                 device=None, eps=1e-6, clip=1e-4, data_on_cpu=None,
                 likelihood="betabinomial", fix_bb_conc=None, event_id=None,
                 psi_parameterization="beta"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        junc = np.asarray(junc); clus = np.asarray(clus)
        # MEMORY. Holding the full matrices dense on the GPU does not scale: the mouse atlas is
        # 92,894 x 79,768, i.e. 29.6 GB per matrix in float32 -- 59 GB for both, on a 44.5 GB card.
        # With data_on_cpu the matrices stay in host memory and only the (B x J) minibatch is moved
        # to the device each step (653 MB at B=2048, J=79,768). Auto-enables past ~4 GB per matrix.
        nbytes = junc.shape[0] * junc.shape[1] * 4
        if data_on_cpu is None:
            data_on_cpu = (self.device.type == "cuda") and (nbytes > 4e9)
        self.data_on_cpu = bool(data_on_cpu)
        store = torch.device("cpu") if self.data_on_cpu else self.device
        self.junc = torch.as_tensor(junc, dtype=torch.float32, device=store)
        self.clus = torch.as_tensor(clus, dtype=torch.float32, device=store)
        self.mask = self.clus > 0
        self.N, self.J = self.junc.shape
        if self.data_on_cpu:
            print(f"[minibatch] data kept on CPU ({2*nbytes/1e9:.1f} GB); "
                  f"moving only the minibatch to {self.device}", flush=True)
        self.K = int(K)
        self.junc_specific_prior = bool(junc_specific_prior)
        self.learn_ab_hyperprior = bool(learn_ab_hyperprior)
        assert likelihood in ("betabinomial", "binomial", "dirichletmultinomial")
        self.likelihood = likelihood   # "binomial" = the input_conc=inf ablation (no overdispersion)
        # "dirichletmultinomial" scores each ATSE ONCE as a count vector instead of scoring its m
        # junctions independently against a shared denominator. That repetition is what inflates the
        # effective sample size m-fold and makes the posterior about sqrt(m) too narrow; see
        # leafletfa/dirichlet_multinomial.py. It needs the junction -> ATSE map, so either pass
        # event_id here or call set_atse() before fit().
        # psi_parameterization:
        #   "beta"    (DEFAULT) psi[k,j] ~ Beta(a, b), independent per junction. Nothing ties the
        #             junctions of an ATSE together, so a program's usages within an event sum to
        #             1 only by accident -- measured at 86% of program-ATSE pairs on the atlas,
        #             and 47% for events with 6+ junctions.
        #   "simplex" psi[k,j] = softmax over the junctions of the ATSE, so the sum is 1 by
        #             construction. Required in practice by the Dirichlet-multinomial: the DM
        #             depends on alpha only through conc * psi, so without this the within-event
        #             scale of psi is unidentified and drifts (measured: sums land near 1.47).
        assert psi_parameterization in ("beta", "simplex")
        self.psi_parameterization = psi_parameterization
        self.atse_idx = self.rep_idx = self.atse_sizes = None
        self.n_atse = None
        if event_id is not None:
            self.set_atse(event_id)
        elif likelihood == "dirichletmultinomial" or psi_parameterization == "simplex":
            raise ValueError(f"likelihood={likelihood!r} / psi_parameterization="
                             f"{psi_parameterization!r} needs event_id "
                             "(or a set_atse() call before fit)")
        # fix_bb_conc: if set, hold the Beta-Binomial concentration at this constant instead of
        # learning it. Tests the high-K dispersion-collapse hypothesis (learned bb_conc runs away to
        # high overdispersion, flattening the likelihood so no factor commits) and matches the §7
        # transfer E-step, which should freeze bb_conc to the trained value rather than switch to Binomial.
        self.fix_bb_conc = None if fix_bb_conc is None else float(fix_bb_conc)
        # Frozen globals for the transfer / E-step path (see `infer_phi`). Set via freeze_globals().
        self.fixed_psi = self.fixed_pi = self.fixed_alpha_pi = self.fixed_dir_conc = None
        self.eps, self.clip = float(eps), float(clip)
        # phi_init: an (N, K) simplex-valued starting point for the per-cell loadings. Unlike
        # psi_init this cannot go through init_to_value -- phi lives under a subsampling plate, so at
        # prototype-trace time the site is batch-shaped (B, K) and a full (N, K) value raises a shape
        # error. It is applied instead by writing AutoNormal's full-size location parameter directly,
        # after the guide's prototype trace has created it (see _apply_phi_init).
        # Which of Pyro's initialisers seeds the guide when no explicit psi_init/phi_init is given.
        #   feasible (DEFAULT)  init_to_feasible -- pushes ZEROS through each constraint transform,
        #                       ignoring the data AND the distribution's parameters. Gives psi = 0.5
        #                       and phi = 1/K exactly, so every program and every cell starts
        #                       IDENTICAL and only gradient noise can separate them.
        #   sample              init_to_sample -- draws from the model's own prior, so psi differs
        #                       per program (spread ~0.12) and phi per cell (~0.14). Breaks both
        #                       symmetries for free, though it starts psi near 0.88 rather than the
        #                       data's ~0.30 (measured not to matter: location contributed -0.010).
        #   median              init_to_median -- breaks the program symmetry (psi spread ~0.13) but
        #                       leaves every cell at 1/K.
        assert init_strategy in ("feasible", "sample", "median")
        self.init_strategy = init_strategy
        self.phi_init = None
        if phi_init is not None:
            q = np.asarray(phi_init, dtype=np.float32)
            assert q.shape == (self.N, self.K), f"phi_init must be (N, K)={(self.N, self.K)}, got {q.shape}"
            q = np.clip(q, 1e-6, None)
            q = q / q.sum(1, keepdims=True)
            self.phi_init = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        self.psi_init = None
        if psi_init is not None:
            p = np.clip(np.asarray(psi_init, dtype=np.float32), self.clip, 1 - self.clip)
            self.psi_init = torch.as_tensor(p, dtype=torch.float32, device=self.device)
        self.guide = None
        self.losses = []
        self._B = None            # active minibatch size, set by fit() before the guide is traced

    # ── model ────────────────────────────────────────────────────────────────
    def model(self, subsample_size=None):
        # Subsample size is an INSTANCE attribute, not a call arg. AutoNormal traces the raw
        # self.model with no args, so a call-arg subsample_size would be None on the guide side
        # while B on the SVI side -- the two would disagree, the guide would materialise phi over
        # all N cells, and phi @ psi would be an (N, J) matrix (29.6 GB at atlas scale -> OOM).
        # Reading self._B guarantees model and guide use the SAME plate. (Job 19279941 OOM'd here.)
        subsample_size = self._B if subsample_size is None else subsample_size
        K, J, N = self.K, self.J, self.N
        one = torch.tensor(1.0, device=self.device)

        # TRANSFER / E-STEP. When a global is supplied it is plugged in as a constant instead of
        # being sampled, so it is not in the guide and SVI cannot move it. Freezing psi alone is not
        # enough: pi and dir_conc together ARE the prior on phi, so re-learning them on new cells
        # infers phi under a different prior than the training cells saw, which confounds any
        # train-vs-heldout comparison. With psi fixed the a/b hyperparameters only enter through
        # p(psi | a, b), so they are skipped entirely rather than sampled and ignored.
        if self.fixed_psi is None:
            if self.learn_ab_hyperprior:
                a_sh = pyro.sample("a_shape", dist.Gamma(one, one))
                a_rt = pyro.sample("a_rate", dist.Gamma(one, one))
                b_sh = pyro.sample("b_shape", dist.Gamma(one, one))
                b_rt = pyro.sample("b_rate", dist.Gamma(one, one))
            else:
                a_sh = a_rt = b_sh = b_rt = one

            if self.psi_parameterization == "simplex":
                # Free logits, normalised within each ATSE. The Beta prior and its a/b
                # hyperparameters do not apply here -- the constraint lives in the transform, so
                # the prior is a plain standard Normal on the logit scale.
                psi_logit = pyro.sample(
                    "psi_logit",
                    dist.Normal(torch.zeros(K, J, device=self.device),
                                torch.ones(K, J, device=self.device)).to_event(2))
                psi = segment_softmax(psi_logit, self.atse_idx, self.n_atse, clip=self.clip)
            else:
                if self.junc_specific_prior:
                    a = pyro.sample("a", dist.Gamma(a_sh, a_rt).expand([J]).to_event(1))
                    b = pyro.sample("b", dist.Gamma(b_sh, b_rt).expand([J]).to_event(1))
                else:
                    a = pyro.sample("a", dist.Gamma(a_sh, a_rt))
                    b = pyro.sample("b", dist.Gamma(b_sh, b_rt))
                psi = pyro.sample("psi", dist.Beta(a + self.eps, b + self.eps)
                                  .expand([K, J]).to_event(2))
        else:
            psi = self.fixed_psi

        if self.fixed_alpha_pi is None:
            alpha_pi = pyro.sample("alpha_pi", dist.Gamma(one, one))
        else:
            alpha_pi = self.fixed_alpha_pi
        if self.fixed_pi is None:
            pi = pyro.sample("pi", dist.Dirichlet((alpha_pi / K).expand([K])))
        else:
            pi = self.fixed_pi
        if self.fixed_dir_conc is None:
            dir_conc = pyro.sample("dir_conc", dist.Gamma(one, one))
        else:
            dir_conc = self.fixed_dir_conc
        dir_conc = torch.clamp(dir_conc, min=1e-6, max=1e6)
        # The Dirichlet-multinomial reuses bb_conc: it plays the same role (alpha = conc * mu),
        # so the site name, prior and freeze_globals path all carry over unchanged. Under the
        # simplex constraint it is exactly the per-event total concentration A_e.
        if self.likelihood in ("betabinomial", "dirichletmultinomial"):
            if self.fix_bb_conc is not None:
                bb_conc = torch.tensor(self.fix_bb_conc, device=self.device)   # held constant, not learned
            else:
                bb_conc = pyro.sample("bb_conc", dist.Gamma(torch.tensor(2.0, device=self.device),
                                                            torch.tensor(0.1, device=self.device)))
                bb_conc = torch.clamp(bb_conc, min=1e-3, max=1e6)

        # THE PLATE. Cells are conditionally independent given the globals. Pyro draws the
        # subsample indices once, shares them with the guide, and scales the LOCAL terms by N/B
        # while leaving the globals above at weight 1.
        with pyro.plate("cells", N, subsample_size=subsample_size) as idx:
            phi = pyro.sample("phi", dist.Dirichlet(pi * dir_conc))          # batch [B], event [K]
            mu = torch.clamp(phi @ psi, self.clip, 1 - self.clip)            # [B, J]
            # When the data lives on the host, index there and move ONLY this minibatch.
            i = idx.to(self.junc.device)
            b_junc = self.junc[i].to(self.device, non_blocking=True)
            b_clus = self.clus[i].to(self.device, non_blocking=True)
            b_mask = self.mask[i].to(self.device, non_blocking=True)
            if self.likelihood == "dirichletmultinomial":
                # ONE term per ATSE, not one per junction. pyro.factor rather than pyro.sample
                # because the log-probability is evaluated over ragged segments of the [B, J]
                # layout -- see leafletfa/dirichlet_multinomial.py for why the distribution
                # object cannot be used directly. The tensor is [B], so the enclosing plate
                # scales it by N/B exactly as it scales the .to_event(1) branch below.
                #
                # No mask: an ATSE with no reads in this cell has every y_j = 0, so its junction
                # terms vanish and lgamma(A_e) - lgamma(0 + A_e) = 0. It drops out on its own.
                pyro.factor("obs", dm_log_prob(b_junc, mu, bb_conc,
                                               self.atse_idx, self.n_atse, eps=self.eps))
            else:
                if self.likelihood == "binomial":
                    obs_dist = dist.Binomial(total_count=b_clus, probs=mu)   # no overdispersion (conc=inf)
                else:
                    obs_dist = dist.BetaBinomial(concentration1=mu * bb_conc,
                                                 concentration0=(1.0 - mu) * bb_conc,
                                                 total_count=b_clus)
                pyro.sample(
                    "obs",
                    obs_dist.mask(b_mask).to_event(1),
                    obs=b_junc,
                )

    # ── guide ────────────────────────────────────────────────────────────────
    def _make_guide(self, subsample_size, init_scale=0.1):
        """AutoNormal, not AutoDiagonalNormal.

        AutoDiagonalNormal flattens every latent into one vector and therefore cannot represent a
        plated local variable under subsampling. AutoNormal keeps one Normal per site and accepts
        `create_plates`, so q(phi) gets full-size (N, K) parameters indexed by the plate.
        """
        def create_plates(*args, **kwargs):
            return pyro.plate("cells", self.N, subsample_size=self._B)

        init_fn = None
        if self.psi_init is not None:
            init_fn = init_to_value(values={"psi": self.psi_init})
        elif self.init_strategy == "sample":
            from pyro.infer.autoguide.initialization import init_to_sample
            init_fn = init_to_sample
        elif self.init_strategy == "median":
            from pyro.infer.autoguide.initialization import init_to_median
            init_fn = init_to_median
        return AutoNormal(self.model, create_plates=create_plates,
                          init_scale=init_scale,
                          **({"init_loc_fn": init_fn} if init_fn is not None else {}))

    def _apply_phi_init(self):
        """Write phi_init into AutoNormal's location parameter, in unconstrained space.

        Dirichlet's support is the simplex, whose unconstrained parametrisation is stick-breaking with
        K-1 coordinates -- hence the (N, K-1) parameter. `biject_to(simplex).inv` maps our (N, K)
        simplex values onto exactly that. Must run AFTER the guide has been traced once (the parameter
        does not exist before) and BEFORE the first SVI step.
        """
        if self.phi_init is None:
            return
        from torch.distributions import biject_to, constraints as _c
        store = pyro.get_param_store()
        key = "AutoNormal.locs.phi"
        if key not in store:
            print("  [phi_init] guide has no phi location parameter -- skipped", flush=True)
            return
        unc = biject_to(_c.simplex).inv(self.phi_init)
        if tuple(store[key].shape) != tuple(unc.shape):
            print(f"  [phi_init] shape mismatch {tuple(store[key].shape)} vs {tuple(unc.shape)}"
                  " -- skipped", flush=True)
            return
        with torch.no_grad():
            store[key].copy_(unc)

    # ── fit ──────────────────────────────────────────────────────────────────
    def _param_snapshot(self):
        """Deep copy of the whole param store, on CPU, for checkpoint/rollback."""
        return {k: v.detach().cpu().clone()
                for k, v in pyro.get_param_store().items()}

    def _restore_snapshot(self, snap):
        store = pyro.get_param_store()
        for k, v in snap.items():
            if k in store:
                with torch.no_grad():
                    store[k].copy_(v.to(store[k].device))

    def fit(self, num_steps=2000, subsample_size=1024, lr=0.05, gamma=0.5,
            num_particles=1, seed=0, log_every=100, verbose=True, checkpoint_every=None,
            early_stop=True, patience=8, min_delta_rel=1e-4, es_smooth=0.98):
        """One model, one guide, one optimiser, for the whole run -- nothing is ever rebuilt.

        Both SVI and AutoNormal use the SAME `self.model` object and read the batch size from
        `self._B`, so model and guide trace the identical subsampling plate. Passing the model as a
        lambda with a call-arg subsample_size to SVI while AutoNormal traced the raw model was the
        cause of the atlas-scale OOM: the guide ran phi over all N cells.

        EARLY STOPPING (early_stop=True): num_steps becomes a CAP, not a fixed budget -- each model
        trains until its (smoothed) ELBO stops improving, so easy configs stop early and hard ones use
        more. The criterion is RELATIVE and SMOOTHED (min_delta_rel against an EMA of the loss), unlike
        the old pipeline's absolute min_delta=0.01 against a 1e6-1e8 ELBO, which was numerically inert.
        Stop after `patience` consecutive checkpoint intervals with < min_delta_rel relative improvement;
        the best (lowest-EMA) snapshot is restored, which doubles as the NaN-guard checkpoint.
        NOTE: the ELBO is not the same as held-out recovery -- for undersized batches the ELBO can keep
        improving while recovery degrades. Early stopping saves compute; it does not fix batch-size
        attenuation. See experiments/minibatch/CONVERGENCE_FINDINGS.md.
        """
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        self._B = min(int(subsample_size), self.N)
        self.guide = self._make_guide(self._B)
        if self.phi_init is not None:
            self.guide()                      # prototype trace, so the parameters exist
            self._apply_phi_init()
        lrd = gamma ** (1.0 / max(num_steps, 1))
        svi = SVI(self.model, self.guide,
                  ClippedAdam({"lr": lr, "lrd": lrd, "clip_norm": 10.0}),
                  Trace_ELBO(num_particles=num_particles))
        # NaN-GUARD + CHECKPOINT-BEST. Job 19286437 trained cleanly for 2745 steps, then a single
        # bad gradient blew the loss to NaN and corrupted every saved parameter -- 40 min of good
        # training lost. We now snapshot the param store on a stable interval and, if a step returns
        # a non-finite loss, restore the last good snapshot and stop. The returned model is always
        # the best finite state reached, never the blown-up one.
        checkpoint_every = checkpoint_every or log_every
        self.losses = []
        best_snapshot = self._param_snapshot()
        best_step = -1
        best_ema = np.inf
        ema = None
        no_improve = 0
        self.stopped_early = False
        self.stopped_step = num_steps
        for step in range(num_steps):
            loss = svi.step()
            self.losses.append(loss)
            if not np.isfinite(loss):
                print(f"  !! non-finite loss at step {step} -- restoring checkpoint from step "
                      f"{best_step} and stopping (NaN-guard).", flush=True)
                self._restore_snapshot(best_snapshot)
                self.stopped_step = step
                break
            ema = loss if ema is None else es_smooth * ema + (1.0 - es_smooth) * loss
            if step % checkpoint_every == 0:
                # snapshot the BEST (lowest-EMA) state; this is both the early-stop reference and the
                # NaN-guard restore point.
                if ema < best_ema * (1.0 - min_delta_rel):
                    best_ema = ema
                    best_snapshot = self._param_snapshot()
                    best_step = step
                    no_improve = 0
                else:
                    no_improve += 1
                    if early_stop and no_improve >= patience:
                        if verbose:
                            print(f"  early stop at step {step} (no >{min_delta_rel:.0e} relative ELBO "
                                  f"improvement for {patience} checks); restoring step {best_step}.",
                                  flush=True)
                        self._restore_snapshot(best_snapshot)
                        self.stopped_early = True
                        self.stopped_step = step
                        break
            if verbose and (step % log_every == 0 or step == num_steps - 1):
                with torch.no_grad():
                    psi = self.psi_mean()
                print(f"  step {step:>5}  loss={loss:.4e}  psi_mean={psi.mean():.4f}", flush=True)
        return self.losses

    # ── transfer / E-step ────────────────────────────────────────────────────
    def set_atse(self, event_id):
        """Attach the junction -> ATSE map that the Dirichlet-multinomial likelihood needs.

        `event_id` is a length-J array of ATSE labels, one per junction column, in the same
        order as the count matrices. Safe to call on any model; it is only read when
        likelihood='dirichletmultinomial'.
        """
        atse_idx, rep_idx, sizes = build_atse_index(event_id)
        if len(atse_idx) != self.J:
            raise ValueError(f"event_id has {len(atse_idx)} entries but the data has {self.J} "
                             "junction columns")
        self.atse_idx = atse_idx.to(self.device)
        self.rep_idx = rep_idx.to(self.device)
        self.atse_sizes = sizes.to(self.device)
        self.n_atse = int(len(sizes))
        return self

    def freeze_globals(self, psi=None, pi=None, alpha_pi=None, dir_conc=None, bb_conc=None):
        """Plug trained globals in as constants so only phi is inferred.

        This is the operation that applies a trained model to new cells: the global/local split
        means everything shared across cells (psi, pi, alpha, delta, the concentration) is frozen
        and each cell's loading is inferred against it. Freeze *all* of them, not just psi —
        pi and dir_conc are the Dirichlet prior on phi, so leaving them free means new cells get
        their phi under a prior fitted to themselves rather than the one the model was trained with.
        """
        def _t(x):
            if x is None:
                return None
            return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        if psi is not None:
            p = np.clip(np.asarray(psi, dtype=np.float32), self.clip, 1 - self.clip)
            self.fixed_psi = torch.as_tensor(p, device=self.device)
            assert self.fixed_psi.shape == (self.K, self.J), \
                f"psi is {tuple(self.fixed_psi.shape)}, expected {(self.K, self.J)}"
        self.fixed_pi = _t(pi)
        self.fixed_alpha_pi = _t(alpha_pi)
        self.fixed_dir_conc = _t(dir_conc)
        if bb_conc is not None and np.isfinite(float(bb_conc)):
            self.fix_bb_conc = float(bb_conc)
        return self

    def infer_phi(self, num_steps=3000, subsample_size=None, lr=0.1, gamma=0.1, seed=0,
                  verbose=False, **kw):
        """E-step: infer phi for these cells with every global frozen.

        Call `freeze_globals(...)` first. With psi fixed the per-cell problems are independent and
        K-dimensional, so this is a far easier optimisation than training and converges quickly;
        the same early stopping applies. Returns the posterior-median phi, (n_cells, K).
        """
        if self.fixed_psi is None:
            raise ValueError("call freeze_globals(psi=...) before infer_phi()")
        B = self.N if subsample_size is None else min(int(subsample_size), self.N)
        self.fit(num_steps=num_steps, subsample_size=B, lr=lr, gamma=gamma, seed=seed,
                 verbose=verbose, **kw)
        return self.phi_mean().cpu().numpy()

    # ── posterior summaries ──────────────────────────────────────────────────
    def _median(self):
        return self.guide.median()

    def _identified_psi(self, psi):
        """Return the psi that is actually identified by the likelihood in use.

        The Dirichlet-multinomial depends on alpha = conc * psi, and ONLY on alpha. Rescaling psi
        within an event by c while scaling conc by 1/c gives an identical likelihood, so the raw
        psi array carries an arbitrary per-event scale and only psi_j / sum_{j in e} psi_j is
        pinned. Returning the raw array is not just untidy -- every downstream consumer uses it,
        and TV = 0.5 * sum_j |dphi . psi_:,j| is then rescaled per event, which breaks its
        comparability against the fixed delta threshold. Measured: normalising raw DM psi moved
        recovery from r=0.926 to r=0.9956 and within-ATSE validity from 17.5% to 100%.

        Under `psi_parameterization="simplex"` psi already sums to 1 within each event, so this is
        a no-op there and safe to apply unconditionally for the DM.
        """
        if self.likelihood != "dirichletmultinomial":
            return psi
        tot = _segment_sum(psi, self.atse_idx, self.n_atse)      # [K, n_atse]
        return psi / tot.index_select(1, self.atse_idx).clamp_min(1e-12)

    def psi_mean(self):
        # When psi is frozen it is a constant, not a guide site, so the posterior median does not
        # contain it — return what was frozen so callers get the dictionary they supplied.
        if self.fixed_psi is not None:
            return self.fixed_psi.detach()
        with torch.no_grad():
            if self.psi_parameterization == "simplex":
                return segment_softmax(self._median()["psi_logit"].detach(),
                                       self.atse_idx, self.n_atse, clip=self.clip)
            return self._identified_psi(self._median()["psi"].detach())

    def phi_mean(self):
        """Posterior median of the per-cell loadings, renormalised to the simplex."""
        with torch.no_grad():
            phi = self._median()["phi"].detach()
            return phi / phi.sum(-1, keepdim=True).clamp_min(1e-12)


    def phi_uncertainty(self, n_samples=64, seed=0):
        """Per-cell posterior spread of phi -- the width we have been discarding.

        `phi_mean` alone presents a point estimate as though it were precise. It is not: at the
        overdispersion the atlas exhibits (c ~ 2), the Beta-Binomial caps the information any junction
        can carry at ~c+1 reads however deeply the cell is sequenced, so a single cell locates its own
        loadings only loosely. Reporting the width alongside the estimate is the honest form, and it
        costs one batch of guide samples.

        q(phi) is Normal in the unconstrained (stick-breaking) space, so the spread is not available in
        closed form on the simplex -- it is obtained by sampling and summarising on the constrained
        scale. The guide's full-size (N, K-1) location and scale parameters are sampled directly rather
        than by tracing the guide: tracing runs under the subsampling plate and would return only the
        current minibatch of cells.

        Returns
        -------
        sd : (N, K)  posterior standard deviation of each loading
        top_prob : (N,)  posterior probability that the argmax program is the one phi_mean reports.
            Low values flag cells whose assignment is not determined by their own reads.
        """
        import pyro
        from torch.distributions import biject_to, constraints as _c
        st = pyro.get_param_store()
        if "AutoNormal.locs.phi" not in st or "AutoNormal.scales.phi" not in st:
            raise RuntimeError("no fitted q(phi) in the param store -- call fit() first")
        with torch.no_grad():
            loc = st["AutoNormal.locs.phi"].detach()
            scale = st["AutoNormal.scales.phi"].detach()
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            to_simplex = biject_to(_c.simplex)
            draws = []
            for _ in range(int(n_samples)):
                z = torch.randn(loc.shape, generator=gen).to(loc.device)
                draws.append(to_simplex(loc + scale * z).cpu())
            D = torch.stack(draws)                      # (S, N, K)
            sd = D.std(0)
            top = to_simplex(loc).cpu().argmax(-1)
            agree = (D.argmax(-1) == top.unsqueeze(0)).float().mean(0)
        return sd.numpy(), agree.numpy()

    def globals_(self):
        with torch.no_grad():
            m = self._median()
            out = {}
            for k in ("pi", "alpha_pi", "dir_conc", "bb_conc", "a", "b"):
                if k in m:
                    out[k] = m[k].detach().cpu().numpy()
            # Frozen globals are constants rather than guide sites; report them too, so a caller
            # cannot tell whether a value was learned or supplied purely from a missing key.
            for k, v in (("pi", self.fixed_pi), ("alpha_pi", self.fixed_alpha_pi),
                         ("dir_conc", self.fixed_dir_conc)):
                if v is not None:
                    out[k] = v.detach().cpu().numpy()
            if self.fix_bb_conc is not None:
                out["bb_conc"] = np.float32(self.fix_bb_conc)
            return out

    def sample_psi(self, num_samples, seed=0):
        """Draw psi from q(psi) on the CONSTRAINED scale -> [S, K, J].

        Downstream scoring code used to do `sigmoid(loc + scale*eps)` by hand, which is only
        correct for the Beta parameterisation. Under `psi_parameterization="simplex"` the guide
        site is `psi_logit` and the transform is a within-ATSE softmax, not a sigmoid, so doing
        it by hand silently produces numbers that are not psi. This method owns the transform so
        callers cannot get it wrong.
        """
        site = "psi_logit" if self.psi_parameterization == "simplex" else "psi"
        st = pyro.get_param_store()
        lk, sk = f"AutoNormal.locs.{site}", f"AutoNormal.scales.{site}"
        if lk not in st or sk not in st:
            raise RuntimeError(f"guide has no {site} site; fit the model first")
        loc, scale = st[lk].detach(), st[sk].detach()
        g = torch.Generator(device=loc.device).manual_seed(int(seed))
        z = loc[None] + scale[None] * torch.randn((num_samples,) + tuple(loc.shape),
                                                  generator=g, device=loc.device)
        if self.psi_parameterization == "simplex":
            S, K, J = z.shape
            return segment_softmax(z.reshape(S * K, J), self.atse_idx, self.n_atse,
                                   clip=self.clip).reshape(S, K, J)
        psi = torch.sigmoid(z)
        if self.likelihood == "dirichletmultinomial":
            # Same identifiability issue as psi_mean: only the within-event composition is pinned.
            S, K, J = psi.shape
            psi = self._identified_psi(psi.reshape(S * K, J)).reshape(S, K, J)
        return psi

    def psi_loc_scale(self):
        """The guide's q(psi) loc and scale in LOGIT space (AutoNormal params).

        Returns (loc, scale) each (K, J), or (None, None) if unavailable. 04e samples
        psi ~ sigmoid(loc + scale*eps) to carry psi posterior uncertainty into imputation instead of
        plugging in the point estimate; verified sigmoid(loc) == psi median.

        NOTE the transform differs by parameterisation: sigmoid for "beta", a within-ATSE softmax
        for "simplex". Use `sample_psi()` rather than applying a transform by hand -- it owns the
        right one. These raw parameters are for inspecting posterior WIDTH."""
        # The guide site is "psi" under the Beta parameterisation and "psi_logit" under the
        # simplex. Hard-coding "psi" made this return (None, None) for every simplex fit, which
        # is silent: callers that guard on None simply store nothing and the posterior width
        # becomes unrecoverable after the run. Pick the site that actually exists.
        st = pyro.get_param_store()
        site = "psi_logit" if self.psi_parameterization == "simplex" else "psi"
        lk, sk = f"AutoNormal.locs.{site}", f"AutoNormal.scales.{site}"
        if lk not in st or sk not in st:
            return None, None
        return (st[lk].detach().cpu().numpy(), st[sk].detach().cpu().numpy())

    def observed_rate(self):
        """Read-weighted splicing rate the likelihood targets: sum(junc)/sum(clus)."""
        with torch.no_grad():
            return float(self.junc.sum() / self.clus.sum().clamp_min(1))
