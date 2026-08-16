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
                 psi_parameterization="beta", psi_logit_prior_sd=1.0,
                 conc_per_atse=False, dm_alpha_floor=1e-2, phi_floor=0.0, conc_min=1e-3,
                 likelihood_weight=None):
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
        # Prior sd on the softmax logits. LeafCutter reparameterises its beta through a simplex
        # with a FREE scale, i.e. effectively weak; a hard Normal(0,1) is not flat -- measured, it
        # implies a mean maximum within-ATSE usage of 0.60 at m=3 (uniform is 0.33) and 0.305 at
        # m=11 (uniform 0.091), so it pulls toward uneven compositions. This exposes the width so
        # that difference from LeafCutter is testable rather than baked in.
        self.psi_logit_prior_sd = float(psi_logit_prior_sd)
        self.atse_idx = self.rep_idx = self.atse_sizes = None
        # Must be set BEFORE set_atse(), which builds the weight from the ATSE sizes. None = the
        # correct likelihood; "m_over_m_minus_1" reproduces the BetaBinomial's accidental
        # over-count deliberately. See dm_log_prob's docstring.
        self.likelihood_weight = likelihood_weight
        self.event_weight = None
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
        # SEPARATE from self.eps, which guards the Beta prior's concentrations at line ~231 where
        # 1e-2 would be a real perturbation. This one floors alpha = conc * mu inside the DM, where
        # the gradient goes as 1/a. See dirichlet_multinomial.dm_log_prob for the full account.
        self.dm_alpha_floor = float(dm_alpha_floor)
        # PER-ATSE CONCENTRATION. A single global scalar has to absorb the overdispersion of every
        # event at once, and on the atlas it cannot: it collapsed to 0.81-1.99 against the ~48
        # recovered on simulations, which is what put alpha at the clip and drove the divergence.
        # LeafCutter fits a concentration per cluster for the same reason. Under the simplex it is
        # cleanly identified -- psi sums to 1 within an event, so A_e = conc_e exactly, and each
        # event's concentration is pinned by its own dispersion rather than shared. The cost is
        # n_atse parameters (24,475 on the atlas) against the K x J already in psi.
        self.conc_per_atse = bool(conc_per_atse)
        # PHI FLOOR. phi is Dirichlet-distributed and AutoNormal parameterises it through a
        # transform whose gradient scales as 1/phi, so a cell driven into a corner of the simplex
        # carries a gradient ~1e6 while every other cell is ordinary. Measured on the atlas: 19 of
        # 1,327,060 phi entries sit below 1e-6, and exactly 18 phi entries were non-finite at the
        # divergence -- the same handful of cells. phi enters the shared loss, so one of them going
        # non-finite poisons every global parameter in the same step. Flooring phi bounds that
        # gradient the way clip already bounds psi. 0.0 disables (the behaviour to date).
        self.phi_floor = float(phi_floor)
        # LOWER BOUND ON CONCENTRATION. Measured on the atlas: conc collapses from 0.975 to 0.518 by
        # step 1,600 -- before any program structure has formed -- and never moves again, while
        # psi_contrast peaks at step ~4,000 and then flatlines for 12,000 more steps. 67% of events
        # end below conc=1. A Dirichlet-multinomial with conc<1 says the composition is nearly
        # unconstrained, so the likelihood barely rewards a cell for loading on the RIGHT program,
        # which is the pressure that would otherwise spread phi across programs. The released
        # BetaBinomial atlas fits sat at bb_conc ~1.97 and separated cell types far better
        # (Fbln2 ||dphi||_1 1.28 vs 0.23).
        #
        # This is the documented "free concentration is an early-optimisation escape valve" failure
        # that freezing bb_conc fixed at K=50. Bounding rather than freezing keeps the per-event
        # flexibility that the DS test needs while removing the escape route.
        self.conc_min = float(conc_min)
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
                                torch.full((K, J), self.psi_logit_prior_sd,
                                           device=self.device)).to_event(2))
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
                # held constant, not learned. A scalar broadcasts; a per-ATSE vector is gathered to
                # [J] exactly as the learned per-ATSE site is below.
                if torch.is_tensor(self.fix_bb_conc) and self.fix_bb_conc.ndim == 1:
                    bb_conc = self.fix_bb_conc[self.atse_idx]
                else:
                    bb_conc = torch.tensor(float(self.fix_bb_conc), device=self.device)
            elif self.conc_per_atse:
                # One concentration per ATSE, same Gamma(2, 0.1) prior applied independently. The
                # site is [n_atse] and is gathered to [J] below, so alpha = conc_e * mu broadcasts
                # over the [B, J] batch without materialising anything ragged.
                if self.atse_idx is None:
                    raise ValueError("conc_per_atse=True needs event_id so ATSEs can be indexed")
                bb_conc = pyro.sample(
                    "bb_conc_atse",
                    dist.Gamma(torch.full((self.n_atse,), 2.0, device=self.device),
                               torch.full((self.n_atse,), 0.1, device=self.device)).to_event(1))
                bb_conc = torch.clamp(bb_conc, min=self.conc_min, max=1e6)[self.atse_idx]
            else:
                bb_conc = pyro.sample("bb_conc", dist.Gamma(torch.tensor(2.0, device=self.device),
                                                            torch.tensor(0.1, device=self.device)))
                bb_conc = torch.clamp(bb_conc, min=self.conc_min, max=1e6)

        # THE PLATE. Cells are conditionally independent given the globals. Pyro draws the
        # subsample indices once, shares them with the guide, and scales the LOCAL terms by N/B
        # while leaving the globals above at weight 1.
        with pyro.plate("cells", N, subsample_size=subsample_size) as idx:
            phi = pyro.sample("phi", dist.Dirichlet(pi * dir_conc))          # batch [B], event [K]
            if self.phi_floor > 0:
                phi = phi.clamp_min(self.phi_floor)
                phi = phi / phi.sum(-1, keepdim=True)                        # back onto the simplex
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
                pyro.factor("obs", dm_log_prob(b_junc, mu, bb_conc, self.atse_idx, self.n_atse,
                                               eps=self.dm_alpha_floor,
                                               event_weight=self.event_weight))
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
    def _report_nonfinite(self, step):
        """On a non-finite loss, say WHICH tensor went bad -- parameter or gradient, and where.

        The divergence is deterministic (K=20 seed 1 hits step 2016 in independent runs; K=30
        seed 0 at 1513), so it reproduces on demand and is worth instrumenting rather than
        guessing at. Every fix so far has been a workaround; this is what would identify the cause.
        """
        store = pyro.get_param_store()
        print(f"  [nan-debug] step {step}", flush=True)
        for k in sorted(store._params):
            p = store._params[k]
            g = p.grad
            bad_p = int((~torch.isfinite(p)).sum())
            bad_g = int((~torch.isfinite(g)).sum()) if g is not None else -1
            finite_p = p[torch.isfinite(p)]
            rng = (f"[{finite_p.min():.3e}, {finite_p.max():.3e}]" if finite_p.numel() else "all bad")
            note = ""
            if g is not None and torch.isfinite(g).any():
                fg = g[torch.isfinite(g)]
                note = f"  |grad|max {fg.abs().max():.3e}"
            if bad_p or bad_g > 0 or True:
                print(f"    {k:<34} shape {tuple(p.shape)}  nonfinite param {bad_p:,} "
                      f"grad {bad_g:,}  param range {rng}{note}", flush=True)

    def _param_snapshot(self):
        """Deep copy of the whole param store, on CPU, for checkpoint/rollback.

        Snapshots the UNCONSTRAINED tensors via get_state(), not the constrained views returned by
        items(). ParamStoreDict.__getitem__ builds transform_to(constraint)(unconstrained) fresh on
        every access, so a snapshot of items() paired with an in-place copy_ back writes into a
        temporary and is discarded -- silently, and only for constrained parameters. AutoNormal
        holds every scale under constraints.positive, so the previous version rolled back the guide's
        locs and left its scales at whatever diverged. That is why a fit whose NaN-guard reported
        "restoring checkpoint from step 900" was saved with psi_loc finite and psi_scale all-NaN.
        """
        state = pyro.get_param_store().get_state()
        return {"params": {k: v.detach().cpu().clone() for k, v in state["params"].items()},
                "constraints": dict(state["constraints"])}

    def _restore_snapshot(self, snap):
        store = pyro.get_param_store()
        if not isinstance(snap, dict) or "params" not in snap:      # legacy items()-style snapshot
            for k, v in (snap or {}).items():
                if k in store:
                    with torch.no_grad():
                        store[k].copy_(v.to(store[k].device))
                return
        # In-place into the UNCONSTRAINED tensors. set_state() would also work for the values, but it
        # REBINDS store._params to new tensor objects, orphaning the references already held by the
        # optimiser's param groups -- so training would carry on updating tensors nobody reads.
        # Copying in place keeps object identity intact for the guide and the optimiser both.
        with torch.no_grad():
            for k, v in snap["params"].items():
                p = store._params.get(k)
                if p is not None and p.shape == v.shape:
                    p.data.copy_(v.to(p.device))

    def fit(self, num_steps=2000, subsample_size=1024, lr=0.05, gamma=0.5, clip_norm=10.0,
            num_particles=1, seed=0, log_every=100, verbose=True, checkpoint_every=None,
            early_stop=True, patience=8, min_delta_rel=1e-4, es_smooth=0.98,
            max_nan_restarts=20, debug_nan=False, track_every=0, on_track=None):
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
                  ClippedAdam({"lr": lr, "lrd": lrd, "clip_norm": clip_norm}),
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
        self.n_nan_restarts = 0
        self.nan_steps = []
        # PARAMETER TRAJECTORY. The atlas ELBO cannot answer "has this converged": the whole run
        # moves it <1%, and the last-20% slope flips SIGN between fits, so it is noise-dominated.
        # What can answer it is whether the STRUCTURE has stopped moving -- psi_contrast (the
        # across-program spread that IS the differential-splicing effect size), phi_spread, the
        # program prior pi, and the concentration. Recorded on a fixed interval so convergence is a
        # measurement rather than an assumption. Costs one psi_mean()/phi_mean() per checkpoint.
        self.trace = []
        for step in range(num_steps):
            loss = svi.step()
            self.losses.append(loss)
            if not np.isfinite(loss):
                # RESTORE AND CONTINUE, rather than stop. Every atlas fit diverges somewhere around
                # step 1,500-2,200, so stopping meant no atlas fit ever got past ~10% of its budget.
                # Rolling back to the last good state and halving the learning rate lets the run
                # continue past the instability; give up only if it recurs `max_nan_restarts` times,
                # which distinguishes a transient bad step from a genuinely divergent configuration.
                self.n_nan_restarts += 1
                if debug_nan:
                    self._report_nonfinite(step)
                self._restore_snapshot(best_snapshot)
                if self.n_nan_restarts > max_nan_restarts:
                    print(f"  !! non-finite loss at step {step}, and {self.n_nan_restarts-1} "
                          f"restarts already used -- restoring step {best_step} and stopping.",
                          flush=True)
                    self.stopped_step = step
                    break
                # Clearing the OPTIMISER STATE is the part that makes the restart work at all.
                # Adam carries exp_avg / exp_avg_sq per parameter; once a step produces a
                # non-finite gradient those buffers are NaN, and restoring the parameters does not
                # touch them -- so the very next update writes NaN straight back. Measured before
                # this line existed: the guard fired at steps 2016, 2018, 2020, 2022, 2024, 2026,
                # 2028, i.e. every second step, burning all six restarts in twelve steps. Wiping
                # the state costs the momentum estimate, which is the right trade for a fit that
                # would otherwise stop at 12% of its budget.
                # Halve the lr ONLY when the divergence recurs quickly. A hit 400 steps after the
                # last one is a fresh transient and the lr was evidently fine in between; halving
                # unconditionally took it to lr/64 by the sixth event and the run crawled even
                # where it was recovering cleanly (measured: restarts at 2028, 2311, 2701, 3012,
                # 3419, 3432 -- mostly hundreds of steps apart).
                rapid = self.nan_steps and (step - self.nan_steps[-1]) < 100
                for g in svi.optim.optim_objs.values():
                    st_ = getattr(g, "state", None)
                    if st_ is not None:
                        st_.clear()
                    if rapid:
                        for pg in getattr(g, "param_groups", []):
                            pg["lr"] *= 0.5
                print(f"  !! non-finite loss at step {step} -- restored step {best_step}"
                      f"{', lr halved' if rapid else ''} "
                      f"(restart {self.n_nan_restarts}/{max_nan_restarts}), continuing.",
                      flush=True)
                self.nan_steps.append(int(step))
                ema = best_ema
                continue
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
            if track_every and (step % track_every == 0 or step == num_steps - 1):
                with torch.no_grad():
                    _psi = self.psi_mean()
                    _phi = self.phi_mean()
                    _psi = _psi.detach().cpu().numpy() if hasattr(_psi, "detach") else np.asarray(_psi)
                    _phi = _phi.detach().cpu().numpy() if hasattr(_phi, "detach") else np.asarray(_phi)
                    _med = self._median()
                    _pi = _med.get("pi")
                    _pi = (_pi.detach().cpu().numpy() if hasattr(_pi, "detach")
                           else (np.asarray(_pi) if _pi is not None else np.zeros(0)))
                    _c = self.concentration()          # clamped, i.e. what the model uses
                    _occ = np.bincount(_phi.argmax(1), minlength=self.K) / _phi.shape[0]
                    self.trace.append(dict(
                        step=int(step), loss=float(loss),
                        psi_contrast=float(_psi.std(axis=0).mean()),
                        phi_spread=float(_phi.std(axis=0).mean()),
                        psi_mean=float(_psi.mean()),
                        n_live=int((_occ >= 0.01).sum()),
                        occ_max=float(_occ.max()),
                        pi=_pi.astype(np.float32).copy(),
                        conc_median=float(np.median(_c)) if _c.size else float("nan")))
                    if on_track is not None:
                        # Lets the caller persist a usable fit mid-run. Without it a wall-clock
                        # timeout destroys everything: K=10 seed 1 was killed at 8h at step 10,000
                        # with ZERO restarts -- not diverging, just placed on a node running 2.9
                        # s/step against its siblings' 0.6 -- and because the fit was written only
                        # at exit, all of it was lost.
                        try:
                            on_track(int(step), _psi, _phi, self.trace[-1])
                        except Exception as _e:
                            print(f"  [checkpoint] skipped at step {step}: {_e}", flush=True)

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
        if self.likelihood_weight in (None, "none"):
            self.event_weight = None
        elif self.likelihood_weight == "m_over_m_minus_1":
            m = self.atse_sizes.to(torch.float32)
            # m = 1 cannot happen for a real ATSE (an event needs >= 2 junctions), but guard it
            # rather than emit an infinity that would be traced back here days later.
            self.event_weight = torch.where(m > 1, m / (m - 1), torch.ones_like(m))
            w = self.event_weight
            print(f"  [weight] tempered likelihood m/(m-1): mean {float(w.mean()):.3f}, "
                  f"max {float(w.max()):.3f} over {self.n_atse:,} events. This is NOT a "
                  f"probability model -- intervals will be too narrow by construction.", flush=True)
        else:
            raise ValueError(f"unknown likelihood_weight {self.likelihood_weight!r}")
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
        if bb_conc is not None:
            arr = np.asarray(bb_conc, dtype=np.float32)
            if arr.ndim == 0:
                if np.isfinite(float(arr)):
                    self.fix_bb_conc = float(arr)
            else:
                # Per-ATSE concentration, as the shipped atlas fits learn it. Applying a trained
                # model to new cells must freeze the SAME concentration each event was fitted with:
                # collapsing 24,475 per-event values to one scalar would give held-out cells a
                # different effective sample size per event than training cells had, and the
                # imputation comparison would then be against a model that was never trained.
                if not self.conc_per_atse:
                    raise ValueError("a vector bb_conc needs conc_per_atse=True")
                if arr.shape != (self.n_atse,):
                    raise ValueError(f"bb_conc vector is {arr.shape}, expected {(self.n_atse,)}")
                self.fix_bb_conc = torch.as_tensor(arr, device=self.device)
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
                out["bb_conc"] = (self.fix_bb_conc.detach().cpu().numpy()
                                  if torch.is_tensor(self.fix_bb_conc)
                                  else np.float32(self.fix_bb_conc))
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

    def concentration(self):
        """The concentration the model ACTUALLY uses, as a numpy array.

        `_median()` returns the guide's median for the site, which is NOT clamped -- the clamp is
        applied inside `model()` on the sampled value. Reading the raw median therefore reports
        something the model never used: with conc_min=2.0 the saved vector had a minimum of 1.536.
        That is cosmetic for training, which used the bound, but anything downstream reading the
        saved vector (the DS test takes it as a per-event initialisation) would get values below the
        floor the fit was trained under.

        Returns shape [n_atse] when conc_per_atse, else a scalar array. Empty if the likelihood has
        no concentration (the Binomial ablation).
        """
        if self.likelihood not in ("betabinomial", "dirichletmultinomial"):
            return np.zeros(0, np.float32)
        if self.fix_bb_conc is not None:
            return np.asarray(self.fix_bb_conc, np.float32)
        try:
            med = self._median()
        except Exception:
            return np.zeros(0, np.float32)
        c = med.get("bb_conc_atse", med.get("bb_conc"))
        if c is None:
            return np.zeros(0, np.float32)
        c = c.detach() if hasattr(c, "detach") else torch.as_tensor(np.asarray(c))
        c = torch.clamp(c, min=self.conc_min, max=1e6)          # the model's own clamp
        return np.atleast_1d(c.cpu().numpy()).astype(np.float32)

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
