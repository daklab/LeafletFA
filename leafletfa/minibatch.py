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
    psi_init : (K, J) array, optional
        Coverage-aware mean initialisation. Clipped away from 0/1 -- Pyro maps a Beta site through
        a logit and psi_init = 1.0 becomes a loc of +15.9 where the sigmoid gradient is ~1e-7, i.e.
        frozen for the whole run.
    """

    def __init__(self, junc, clus, K, *, junc_specific_prior=False, learn_ab_hyperprior=False,
                 psi_init=None, device=None, eps=1e-6, clip=1e-4, data_on_cpu=None):
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
        self.eps, self.clip = float(eps), float(clip)
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

        if self.learn_ab_hyperprior:
            a_sh = pyro.sample("a_shape", dist.Gamma(one, one))
            a_rt = pyro.sample("a_rate", dist.Gamma(one, one))
            b_sh = pyro.sample("b_shape", dist.Gamma(one, one))
            b_rt = pyro.sample("b_rate", dist.Gamma(one, one))
        else:
            a_sh = a_rt = b_sh = b_rt = one

        if self.junc_specific_prior:
            a = pyro.sample("a", dist.Gamma(a_sh, a_rt).expand([J]).to_event(1))
            b = pyro.sample("b", dist.Gamma(b_sh, b_rt).expand([J]).to_event(1))
        else:
            a = pyro.sample("a", dist.Gamma(a_sh, a_rt))
            b = pyro.sample("b", dist.Gamma(b_sh, b_rt))

        psi = pyro.sample("psi", dist.Beta(a + self.eps, b + self.eps).expand([K, J]).to_event(2))
        alpha_pi = pyro.sample("alpha_pi", dist.Gamma(one, one))
        pi = pyro.sample("pi", dist.Dirichlet((alpha_pi / K).expand([K])))
        dir_conc = pyro.sample("dir_conc", dist.Gamma(one, one))
        bb_conc = pyro.sample("bb_conc", dist.Gamma(torch.tensor(2.0, device=self.device),
                                                    torch.tensor(0.1, device=self.device)))
        dir_conc = torch.clamp(dir_conc, min=1e-6, max=1e6)
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
            pyro.sample(
                "obs",
                dist.BetaBinomial(concentration1=mu * bb_conc,
                                  concentration0=(1.0 - mu) * bb_conc,
                                  total_count=b_clus).mask(b_mask).to_event(1),
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
        return AutoNormal(self.model, create_plates=create_plates,
                          init_scale=init_scale,
                          **({"init_loc_fn": init_fn} if init_fn is not None else {}))

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
            num_particles=1, seed=0, log_every=100, verbose=True, checkpoint_every=None):
        """One model, one guide, one optimiser, for the whole run -- nothing is ever rebuilt.

        Both SVI and AutoNormal use the SAME `self.model` object and read the batch size from
        `self._B`, so model and guide trace the identical subsampling plate. Passing the model as a
        lambda with a call-arg subsample_size to SVI while AutoNormal traced the raw model was the
        cause of the atlas-scale OOM: the guide ran phi over all N cells.
        """
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        self._B = min(int(subsample_size), self.N)
        self.guide = self._make_guide(self._B)
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
        for step in range(num_steps):
            loss = svi.step()
            self.losses.append(loss)
            if not np.isfinite(loss):
                print(f"  !! non-finite loss at step {step} -- restoring checkpoint from step "
                      f"{best_step} and stopping (NaN-guard).", flush=True)
                self._restore_snapshot(best_snapshot)
                break
            if step % checkpoint_every == 0:
                best_snapshot = self._param_snapshot()   # last KNOWN-FINITE state
                best_step = step
            if verbose and (step % log_every == 0 or step == num_steps - 1):
                with torch.no_grad():
                    psi = self.psi_mean()
                print(f"  step {step:>5}  loss={loss:.4e}  psi_mean={psi.mean():.4f}", flush=True)
        return self.losses

    # ── posterior summaries ──────────────────────────────────────────────────
    def _median(self):
        return self.guide.median()

    def psi_mean(self):
        with torch.no_grad():
            return self._median()["psi"].detach()

    def phi_mean(self):
        """Posterior median of the per-cell loadings, renormalised to the simplex."""
        with torch.no_grad():
            phi = self._median()["phi"].detach()
            return phi / phi.sum(-1, keepdim=True).clamp_min(1e-12)

    def globals_(self):
        with torch.no_grad():
            m = self._median()
            out = {}
            for k in ("pi", "alpha_pi", "dir_conc", "bb_conc", "a", "b"):
                if k in m:
                    out[k] = m[k].detach().cpu().numpy()
            return out

    def observed_rate(self):
        """Read-weighted splicing rate the likelihood targets: sum(junc)/sum(clus)."""
        with torch.no_grad():
            return float(self.junc.sum() / self.clus.sum().clamp_min(1))
