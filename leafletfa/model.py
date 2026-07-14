import math
import os
import random
import inspect
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocessing

import torch
from torch.optim import Adam
import pyro
import pyro.distributions as dist
from torch.distributions import Distribution
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import (
    AutoGuideList, AutoDelta, AutoDiagonalNormal
)
from pyro.infer.autoguide.initialization import init_to_value
from pyro import poutine

from pyro.poutine import block
from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from leafletfa._ops import masked_matmul

# Configure logging
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Disable default argument validation for distributions
Distribution.set_default_validate_args(False)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_cpu_optimizations():
    """Configure PyTorch for optimal CPU performance"""
    
    # Get number of physical cores (avoid hyperthreading for numerical workloads)
    n_physical_cores = multiprocessing.cpu_count() // 2
    n_threads = min(n_physical_cores, 16)  # Cap at 16 for diminishing returns
    
    # Set PyTorch threading (interop threads can only be set once per process)
    torch.set_num_threads(n_threads)
    if torch.get_num_interop_threads() == 1:
        torch.set_num_interop_threads(2)
    
    # Set environment variables for underlying libraries
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    
    # Enable MKL optimizations if available
    if torch.backends.mkl.is_available():
        torch.backends.mkl.benchmark = True
        print(f"✓ MKL optimizations enabled")
    
    # Try to enable oneDNN fusion (may not be available on all systems)
    try:
        torch.jit.enable_onednn_fusion(True)
        print(f"✓ oneDNN fusion enabled")
    except:
        pass
    
    print(f"✓ CPU optimized with {n_threads} threads")
    return n_threads


class _ScaleSites(poutine.messenger.Messenger):
    """Multiply the ELBO weight of NAMED sites only, on whatever callable it wraps.

    `poutine.scale` inside model() cannot reach the guide -- the guide is a separate callable. Using
    it there scales log p(assign) and log p(y) but leaves log q(assign) at weight 1, which
    under-weights the entropy holding q(assign) open and drives the per-cell posteriors toward point
    masses by a factor of N/batch. Wrapping BOTH model and guide with this messenger keeps the two
    sides of each local term consistent, while leaving the global KL (psi, pi, a, b, ...) at weight 1.
    """

    def __init__(self, scale, site_names):
        super().__init__()
        self.scale = float(scale)
        self.site_names = set(site_names)

    def _process_message(self, msg):
        if msg["name"] in self.site_names:
            msg["scale"] = msg["scale"] * self.scale


class LeafletFA:
    def __init__(self, adata, K=10, junc_specific_prior=True, input_conc_prior=None, delta_fixed=None,
                 waypoints_use=True, fixed_psi=None, use_dense_mode=None,
                 pi_init=None, alpha_pi_init=None,
                 pi_fixed=None, alpha_pi_fixed=None, obs_scale=1.0,
                 log_wandb=False,
                 eps = 1e-6, lr=0.01, gamma = 0.05, num_epochs=500, print_epochs=10,
                 ELBO_num_particles=1, patience=5, min_delta=0.01, min_delta_rel=0.0,
                 num_samples=100, loss_plot=True,
                 output_dir=None, enable_cpu_optimization=True, clip_norm=10.0):
        """
        Initialize the LeafletFA model with CPU optimizations.

        Parameters:
        - adata (AnnData): AnnData object containing single-cell splicing data.
        - K (int): Number of factors.
        - junc_specific_prior (bool): Whether to learn a global prior over junctions.
        - input_conc_prior: Concentration prior (can be either 'None' or 'torch.tensor(np.inf)').
        - waypoints_use (bool): Whether to use waypoints for initialization.
        - pi_init (torch.Tensor): *Initialization* for the pi vector. pi is still learned.
        - alpha_pi_init (torch.Tensor): *Initialization* for alpha_pi. alpha_pi is still learned.
        - pi_fixed (torch.Tensor): *Freeze* pi to these values (it is not sampled or learned).
          Use this -- not pi_init -- when projecting new cells onto a trained model: pi is the prior
          over phi, so re-learning it on the target puts phi on a different footing than the source.
        - alpha_pi_fixed (torch.Tensor): Freeze alpha_pi. Only meaningful alongside pi_fixed.
        - obs_scale (float): Multiplier on the LOCAL terms (the `assign` prior/entropy and the
          likelihood), leaving the global KL at weight 1. Set to N_total/batch_size when training on
          a subsample so the objective is an unbiased estimate of the full-data ELBO. Leaving this at
          1.0 while chunking optimizes the ELBO of a batch_size-cell dataset instead, over-weighting
          the global prior by N/batch_size and shrinking the factors toward each other.
        - fixed_psi (torch.Tensor): Fix PSI during inference by using an existing PSI latent variable from previous model training.
        - use_dense_mode (bool): Whether to use dense mode for computation.
        - eps (float): Small value to prevent numerical issues.
        - lr (float): Learning rate.
        - gamma (float): Learning rate decay.
        - num_epochs (int): Number of training epochs.
        - print_epochs (int): Print loss every 'print_epochs' epochs.
        - ELBO_num_particles (int): Number of particles for ELBO calculation.
        - patience (int): Early stopping patience.
        - min_delta (float): ABSOLUTE minimum ELBO improvement for early stopping. Note the ELBO
          scales with the number of observations, so an absolute threshold is easy to set far below
          the Monte-Carlo noise floor, making it inert.
        - min_delta_rel (float): RELATIVE minimum improvement, as a fraction of |best_loss|
          (e.g. 1e-5). The effective threshold is max(min_delta, min_delta_rel * |best_loss|).
          Prefer this over min_delta -- it is scale-free.
        - num_samples (int): Posterior samples drawn from the guide. Default 100. This is S in the
          differential-splicing FDR: P_kj = (1/S) sum_s 1(|Delta psi| > delta). At the old default of
          S=10, P_kj could only take 11 distinct values and the FDP estimator returned exactly 0 for
          any junction where all 10 samples cleared delta -- an artefact, not evidence. Do not go
          below ~100.
        - loss_plot (bool): Whether to plot loss.
        - output_dir (str): Directory to save results.
        - enable_cpu_optimization (bool): CPU-only housekeeping (thread setup, sparse coalescing,
          index caching, progress bar). It does NOT select the device — that is taken from
          torch.cuda.is_available() — and since the CPU and GPU maths were unified it no longer
          changes any numerical path.
        - clip_norm (float or None): Gradient-norm clip for ClippedAdam, applied on every device.
          None disables it. Previously this was silently tied to enable_cpu_optimization, so CPU runs
          were clipped and GPU runs were not.
        """

        self.adata = adata
        self.K = K
        self.num_cells = adata.shape[0]
        self.num_junctions = adata.shape[1]
        self.junc_specific_prior = junc_specific_prior
        self.input_conc_prior = input_conc_prior
        self.dir_conc = delta_fixed
        self.waypoints_use = waypoints_use
        self.fixed_psi = fixed_psi
        self.pi_init = pi_init
        self.alpha_pi_init = alpha_pi_init
        self.pi_fixed = pi_fixed
        self.alpha_pi_fixed = alpha_pi_fixed
        self.obs_scale = float(obs_scale)
        self.eps = eps
        self.lr = lr
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.print_epochs = print_epochs
        self.ELBO_num_particles = ELBO_num_particles
        self.patience = patience
        self.use_dense_mode = use_dense_mode
        self.min_delta = min_delta
        self.min_delta_rel = min_delta_rel
        self.num_samples = num_samples
        self.loss_plot = loss_plot
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.enable_cpu_optimization = enable_cpu_optimization
        self.clip_norm = clip_norm
        self._log_binom_coeff = None   # cached sum of log(n choose k); constant in the parameters
        
        # Initialize other attributes
        self.guide = None
        self.losses = []
        self.variable_sizes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cache for indices (CPU optimization)
        self.cached_indices = None
        self.cached_cell_idx = None
        self.cached_junc_idx = None

        # Setup CPU optimizations if enabled and on CPU
        if self.enable_cpu_optimization and self.device.type == 'cpu':
            self.n_threads = setup_cpu_optimizations()
            print(f"🚀 CPU optimizations enabled for {self.num_cells} cells and {self.num_junctions} junctions")
        
        # If fixed PSI is provided, disable waypoint initialization
        if self.fixed_psi is not None:
            if self.waypoints_use:
                print("Fixed PSI provided! Disabling waypoint-based PSI initialization.")
            self.waypoints_use = False

            # A fixed psi is only meaningful if its junction axis is the SAME axis, in the same
            # order, as this adata's. A silent misalignment produces confident nonsense, so fail loud.
            if self.fixed_psi.shape != (self.K, self.num_junctions):
                raise ValueError(
                    f"fixed_psi has shape {tuple(self.fixed_psi.shape)} but this AnnData needs "
                    f"(K={self.K}, n_junctions={self.num_junctions}). The junction axis of fixed_psi "
                    f"must match adata.var exactly, in the same order — subset and reorder the target "
                    f"AnnData to the source model's junctions before projecting."
                )

        if self.pi_fixed is not None:
            self.pi_fixed = torch.as_tensor(self.pi_fixed, dtype=torch.float32)
            if self.pi_fixed.shape != (self.K,):
                raise ValueError(
                    f"pi_fixed has shape {tuple(self.pi_fixed.shape)}, expected (K={self.K},)."
                )
            if not torch.allclose(self.pi_fixed.sum(), torch.ones(()), atol=1e-4):
                raise ValueError(f"pi_fixed must sum to 1; got {self.pi_fixed.sum().item()}.")
        if self.alpha_pi_fixed is not None:
            self.alpha_pi_fixed = torch.as_tensor(self.alpha_pi_fixed, dtype=torch.float32)

    def _c(self, x):
        """A prior hyper-parameter as a tensor ON THE MODEL'S DEVICE.

        Passing python floats to dist.Gamma(...) builds CPU tensors, so Pyro's prototype trace runs
        on the CPU and every AutoGuide parameter is *created* on the CPU -- even when the model is on
        CUDA. Training then samples a 545k-dim Normal on the CPU (once per ELBO particle), copies it
        to the GPU, and runs Adam back on the CPU. That was ~10x slower than keeping everything on
        the GPU.
        """
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _conc_is_sampled(self):
        """True if bb_conc is a latent site (prior is None, or an explicit Distribution)."""
        return (self.input_conc_prior is None
                or isinstance(self.input_conc_prior, torch.distributions.Distribution))

    def _has_global_latents(self):
        """True if any latent site survives once 'psi' and 'assign' are blocked from the guide."""
        return (self.fixed_psi is None        # a, b, a_shape, a_rate, b_shape, b_rate
                or self.pi_fixed is None      # pi, alpha_pi
                or self.dir_conc is None      # dir_conc
                or self._conc_is_sampled())   # bb_conc

    def convertr(self, hyperparam, name):
        """ Convert hyperparameters to Pyro samples or fixed tensors.
        If hyperparam is a distribution, sample from it.
        If hyperparam is None, sample from a Gamma distribution with shape and rate 10.0.
        Otherwise, convert the hyperparameter provided to a tensor (on the model's device).
        """
        if isinstance(hyperparam, torch.distributions.Distribution):
            # pyro.sample needs a PYRO distribution. Handed a raw torch.distributions object it treats
            # it as a plain callable and dies with "'Gamma' object is not callable", which says nothing
            # useful. Fail with an actionable message instead.
            if not isinstance(hyperparam, dist.torch_distribution.TorchDistributionMixin):
                raise TypeError(
                    f"{name}: got a raw torch.distributions.{type(hyperparam).__name__}. Use the Pyro "
                    f"version instead, e.g. `import pyro.distributions as dist; "
                    f"dist.{type(hyperparam).__name__}(...)`."
                )
            # A user-supplied prior is usually built from CPU scalars. Sampling it on the CPU while the
            # model is on CUDA puts the site -- and hence its AutoGuide params -- on the wrong device.
            # Rebuild it from its own parameters, moved to the model's device.
            # arg_constraints is NOT a constructor signature: Binomial/Bernoulli/Categorical
            # materialise BOTH `probs` and `logits` as lazy properties, and passing both raises.
            # Only rebuild for the simple two-parameter priors anyone would use for a concentration.
            init_args = inspect.signature(type(hyperparam).__init__).parameters
            params = {k: getattr(hyperparam, k).to(self.device)
                      for k in type(hyperparam).arg_constraints
                      if k in init_args and isinstance(getattr(hyperparam, k, None), torch.Tensor)}
            if params and not ("probs" in params and "logits" in params):
                hyperparam = type(hyperparam)(**params)
            return pyro.sample(name, hyperparam)
        elif hyperparam is None:
            # Gamma(2, 0.1): mean 20, sd 14 -- weakly informative on the RIGHT SCALE.
            # It was Gamma(10,10): mean 1, sd 0.32. The Beta-Binomial concentration is an
            # INVERSE-overdispersion parameter whose true value is typically 10-100, so a prior with
            # mean 1 and sd 0.32 pins it near 1 (the maximally overdispersed regime) and makes it
            # hard for the data to push toward the binomial limit. Measured on a simulation with a
            # TRUE concentration of 20, the old prior recovered 11.75 -- biased ~40% low.
            # (The Methods text says Gamma(2,2) and the Methods TABLE says Gamma(1,1); all three
            # have mean 1, so all three were wrong for this parameter.)
            return pyro.sample(name, dist.Gamma(self._c(2.0), self._c(0.1)))
        else:
            return self._c(hyperparam)

    def from_anndata(self, float_type=None):
        """
        Process an AnnData object and return PyTorch sparse tensors for junction and cluster counts.
        Optimized for CPU with better memory patterns.
        """
        
        print(f"Taking in the AnnData object with {self.adata.shape[0]} cells and {self.adata.shape[1]} junctions.")

        # Set device automatically if not provided
        if float_type is None:
            float_type = {"dtype": torch.float32,
                          "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
        device = float_type["device"]

        print(f"Processing AnnData on {device}")

        # Ensure that adata.layers contains the necessary matrices 
        if "cell_by_cluster_matrix" not in self.adata.layers.keys():
            raise ValueError("cell_by_cluster_matrix not found in adata.layers.")
        if "cell_by_junction_matrix" not in self.adata.layers.keys():
            raise ValueError("cell_by_junction_matrix not found in adata.layers.")

        # Convert junction layer to COO to get non-zero positions
        junc_coo = coo_matrix(self.adata.layers["cell_by_junction_matrix"])

        # Extract cell and junction indices from junction matrix
        cell_index_array = np.array(junc_coo.row)
        junc_index_array = np.array(junc_coo.col)

        # Convert indices to PyTorch tensors
        cell_index_tensor = torch.tensor(cell_index_array, dtype=torch.int32, device=float_type["device"])
        junc_index_tensor = torch.tensor(junc_index_array, dtype=torch.int32, device=float_type["device"])

        # Convert sparse matrix data to PyTorch tensor
        ycount_array = np.array(junc_coo.data)
        ycount = torch.tensor(ycount_array, **float_type)
        del junc_coo

        # Create sparse tensor for junction counts
        full_y_tensor = torch.sparse_coo_tensor(
            indices=torch.stack([cell_index_tensor, junc_index_tensor]),
            values=ycount,
            size=(len(self.adata.obs), len(self.adata.var)),
            device=float_type["device"]
        )

        # clean up memory
        del ycount
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract cluster (ATSE total) counts at the junction non-zero positions.
        # cluster[c,j] can be non-zero even when junction[c,j] = 0 (other junctions
        # in the same ATSE have reads), so we must index by junction positions, not
        # by all cluster non-zeros.
        cluster_mat = self.adata.layers["cell_by_cluster_matrix"]
        cluster_csr = cluster_mat if isinstance(cluster_mat, csr_matrix) else csr_matrix(cluster_mat)
        total_counts_array = np.asarray(
            cluster_csr[cell_index_array, junc_index_array]
        ).flatten()
        del cluster_csr
        total_counts_tensor = torch.tensor(total_counts_array, **float_type)

        # Create sparse tensor for total counts
        full_total_counts_tensor = torch.sparse_coo_tensor(
            indices=torch.stack([cell_index_tensor, junc_index_tensor]), 
            values=total_counts_tensor,
            size=(len(self.adata.obs), len(self.adata.var)),
            device=float_type["device"]
        )

        # Clean up memory
        del cell_index_tensor, junc_index_tensor, total_counts_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Add the tensors to the self object
        self.y = full_y_tensor
        self.total_counts = full_total_counts_tensor

        # Coalesce on EVERY device. Both of these used to be gated on
        # `enable_cpu_optimization and device == 'cpu'`, so a GPU run (which passes that flag False)
        # left the indices uncached — and then re-extracted them, and re-ran torch.equal() over two
        # 2 x nnz index tensors, on every single SVI step.
        self.y = self.y.coalesce()
        self.total_counts = self.total_counts.coalesce()

        # Validate the two sparsity patterns agree ONCE here, rather than on every step.
        if not torch.equal(self.y._indices(), self.total_counts._indices()):
            raise ValueError("Mismatch between indices of y and total_counts.")

        # Auto-detect density if not explicitly set
        if self.use_dense_mode is None:
            self.detect_data_density()

        # log(n choose k) depends only on the data, so it is cached -- but it must be invalidated
        # whenever the data changes, or a reused object would keep the previous dataset's constant.
        self._log_binom_coeff = None

        self._cache_indices()
    
    def _cache_indices(self):
        """Cache indices for repeated use (CPU optimization)"""
        self.cached_indices = self.y._indices()
        self.cached_cell_idx = self.cached_indices[0]
        self.cached_junc_idx = self.cached_indices[1]
        print(f"✓ Cached {len(self.cached_cell_idx)} sparse indices")
        
    def detect_data_density(self):
        """Detect if data is dense enough to benefit from dense operations"""
        total_elements = self.num_cells * self.num_junctions
        nnz = self.y.nnz if hasattr(self.y, 'nnz') else self.y._nnz()
        density = nnz / total_elements

        print(f"Data density: {density:.1%} ({nnz:,} / {total_elements:,})")
        self.data_density = density
        self.nnz = nnz

        # Auto-decide: if >30% dense, use dense mode
        if self.use_dense_mode is None:
            # Adjust threshold based on matrix size for CPU
            if self.device.type == 'cpu':
                # For CPU, consider memory constraints
                memory_gb = (self.num_cells * self.num_junctions * 4) / (1024**3)  # float32
                if memory_gb > 4:  # If dense matrix would be >4GB
                    threshold = 0.5  # Be more conservative
                else:
                    threshold = 0.3
                self.use_dense_mode = density > threshold
            else:
                self.use_dense_mode = density > 0.3

        if self.use_dense_mode:
            print("🔄 Using DENSE computation mode (memory efficient for dense data)")
        else:
            print("✓ Using SPARSE computation mode (memory efficient for sparse data)")
    
    def initialize_triton_mask(self):
        """Deprecated no-op, kept so existing training scripts keep working.

        The GPU path used to build a dense C x J mask and hand it to a Triton masked-matmul
        kernel. That was wasteful and less accurate than it looked:
          * the mask cost C x J floats (~48 GB for 142k cells x 90k junctions) and was allocated
            twice, once for the mask and again for the kernel's dense output;
          * the kernel never actually read the mask — it only masked on array bounds, so it was
            a plain dense matmul;
          * `tl.dot` runs in TF32, giving ~8e-4 error against float64, versus ~9e-8 for a plain
            float32 gather.
        `predict_rho` now gathers only the observed (cell, junction) entries on both devices,
        which is exact float32, ~30x smaller, and needs no Triton.
        """
        print("initialize_triton_mask() is a no-op: the dense mask is no longer needed.")

    def compute_pred_triton(self, assign, psi):
        """Deprecated alias for `predict_rho` (kept for backwards compatibility)."""
        cell_idx, junc_idx = self.y._indices()
        return self.predict_rho(assign, psi, cell_idx, junc_idx)

    def predict_rho(self, assign, psi, cell_idx, junc_idx):
        """rho[c,j] = sum_k assign[c,k] * psi[k,j], evaluated only at the observed positions.

        Device-agnostic. In sparse mode this never materialises the C x J matrix: it gathers the
        nnz rows of `assign` and columns of `psi` and contracts over K, which is O(nnz*K) instead
        of O(C*J*K).
        """
        if hasattr(self, 'use_dense_mode') and self.use_dense_mode:
            pred_full = torch.matmul(assign, psi)
            return pred_full[cell_idx, junc_idx]

        # Sparse mode. index_select must stay inside the autograd graph: wrapping it in
        # torch.no_grad() detaches assign/psi, so the likelihood gradient never reaches them and
        # only the prior/entropy terms end up training psi and assign.
        assign_rows = assign.index_select(0, cell_idx.long())
        psi_selected = psi.index_select(1, junc_idx.long())
        return torch.einsum('nk,kn->n', assign_rows, psi_selected)

    # Backwards-compatible alias.
    sparse_dot_cpu = predict_rho

    def model(self, y=None, total_counts=None, K=None,
              junc_specific_prior=None, input_conc_prior=None):
        """
        Optimized probabilistic Bayesian model with CPU-specific improvements.
        """
        
        # If arguments are None, use self values
        if y is None:
            y = self.y
        if total_counts is None:
            total_counts = self.total_counts
        if K is None:
            K = self.K
        if junc_specific_prior is None:
            junc_specific_prior = self.junc_specific_prior
        if input_conc_prior is None:
            input_conc_prior = self.input_conc_prior

        N, P = self.y.shape # Cells, junctions

        # Sample all the PSI related parameters if fixed_psi is None
        if self.fixed_psi is None:

            # Sample input_conc from the prior provided during initialization 
            input_conc = self.convertr(input_conc_prior, "bb_conc")

            # Learnable priors for Gamma shape and rate
            shape_prior = dist.Gamma(self._c(1.0), self._c(1.0))
            a_shape, a_rate, b_shape, b_rate = [pyro.sample(name, shape_prior) for name in ["a_shape", "a_rate", "b_shape", "b_rate"]]

            # Sample psi from a Beta distribution
            if junc_specific_prior:
                # junction-specific shape parameters sampled from Gamma priors 
                a = pyro.sample("a", dist.Gamma(a_shape, a_rate).expand([P]).to_event(1))
                b = pyro.sample("b", dist.Gamma(b_shape, b_rate).expand([P]).to_event(1))
                psi = pyro.sample("psi", dist.Beta(a+self.eps, b+self.eps).expand([K, P]).to_event(2)) 
                psi = psi.to(dtype=torch.float32)
            else:
                # single shape parameters for all junctions sampled from Gamma priors
                a = pyro.sample("a", dist.Gamma(a_shape, a_rate))  
                b = pyro.sample("b", dist.Gamma(b_shape, b_rate))  
                psi = pyro.sample("psi", dist.Beta(a+self.eps, b+self.eps).expand([K, P]).to_event(2))
                psi = psi.to(dtype=torch.float32)

            # Assert that 'psi' has no NaN or negative values
            if not torch.isfinite(psi).all():
                raise ValueError("psi contains NaN or infinite values!")
            
        else: 
            # Use fixed PSI values and initialize pi from a previous model training
            psi = self.fixed_psi.to(self.device)
            # Sample input_conc as usual
            input_conc = self.convertr(input_conc_prior, "bb_conc")

        # PI
        # pi_fixed genuinely freezes pi: the site is never sampled, so the source model's prior over
        # phi is preserved and phi stays comparable across datasets. pi_init only *initialises* a pi
        # that is still learned (the warm-start is applied via init_to_value in the guide, see train()).
        if self.pi_fixed is not None:
            pi = self.pi_fixed.to(self.device, dtype=torch.float32)
            alpha_pi = (self.alpha_pi_fixed.to(self.device, dtype=torch.float32)
                        if self.alpha_pi_fixed is not None else None)
        else:
            alpha_pi = pyro.sample("alpha_pi", dist.Gamma(self._c(1.0), self._c(1.0)))
            # Build the concentration from alpha_pi itself so it inherits alpha_pi's device rather
            # than assuming CPU (torch.ones(K) was a latent device bug under a CUDA default dtype).
            pi = pyro.sample("pi", dist.Dirichlet((alpha_pi / K).expand(K)))

        # Assert that pi sums to 1 and contains no NaN/inf values
        assert torch.isfinite(pi).all(), "pi contains NaN or infinite values!"
        assert torch.allclose(pi.sum(), torch.ones((), dtype=pi.dtype, device=pi.device)), \
            f"pi does not sum to 1: {pi.sum()}"

        # Check if we are used a fixed delta value or if we are learning it 
        if self.dir_conc is None:
            # Sample concentration value that scales the pi vector (higher values lead to more uniform pi)
            # Gamma(1,1), matching the Methods. Was Gamma(2,2) -- same mean (1), but half the
            # variance, i.e. MORE informative than what the paper claims. Of the two, Gamma(1,1) is
            # the more diffuse and lets the data speak.
            #
            # CAVEAT (see METHODS.md): delta scales Dirichlet(delta * pi), so with pi ~ 1/K the
            # per-component concentration is delta/K. At delta ~ 1 and K = 20 that is 0.05 -- a prior
            # that on its own already implies a median phi perplexity of ~2.5. The paper reports a
            # median SP perplexity of 2.46. That does NOT make the finding wrong, but it does mean
            # perplexity alone cannot distinguish a biological result from a restatement of the
            # prior. `health_check` now reports the fitted-vs-prior perplexity so the gap is visible.
            dir_conc = pyro.sample("dir_conc", dist.Gamma(self._c(1.0), self._c(1.0)))
            dir_conc = torch.clamp(dir_conc, min=1e-6, max=1e6) # Prevent numerical issues
        else:
            # Scalar float broadcasts with pi regardless of device (avoids CPU/CUDA mismatch
            # during AutoDiagonalNormal prototype setup, where pi is sampled on CPU even on GPU)
            dir_conc = self.dir_conc.item() if isinstance(self.dir_conc, torch.Tensor) else float(self.dir_conc)

        # Sample the factor assignments for cells. `assign` is a LOCAL variable (one row per cell).
        # Its obs_scale rescaling is NOT applied here: a poutine.scale context inside model() cannot
        # reach the guide, so it would scale log p(assign) but not log q(assign) and silently
        # under-weight the entropy that stops q(assign) collapsing. `_ScaleSites` in fit() wraps the
        # model AND the guide, so both sides of the assign term get the same scale.
        assign = pyro.sample("assign", dist.Dirichlet(pi * dir_conc).expand([N]).to_event(1)).to(dtype=torch.float32)

        # Ensure no negative values in assign
        if torch.any(assign < self.eps):
            assign = assign + self.eps
            assign = assign / assign.sum(dim=1, keepdim=True)  # Re-normalize to sum to 1

        assign = assign.to(self.device, dtype=torch.float32)
        psi = psi.to(self.device, dtype=torch.float32)

        # Indices are cached once in from_anndata() (on every device), where the y/total_counts
        # sparsity patterns are also validated. Re-deriving them here would re-run an O(nnz)
        # extraction and comparison on every SVI step.
        if self.cached_cell_idx is not None and self.cached_junc_idx is not None:
            cell_idx = self.cached_cell_idx
            junc_idx = self.cached_junc_idx
        else:
            cell_idx, junc_idx = y._indices()

        # Compute log-probability. One path for both devices: gather rho at the observed positions,
        # then score. No dense C x J intermediate, and no Triton mask to initialise.
        assign = assign.contiguous()
        psi = psi.contiguous()
        pred = self.predict_rho(assign, psi, cell_idx, junc_idx)
        log_prob = self.log_prob_calc_sparse(y, total_counts, pred, input_conc)

        assert torch.isfinite(log_prob).all(), "log_prob contains NaN or infinite values!"

        # obs_scale is applied by _ScaleSites in fit(), not here -- see the note on `assign` above.
        pyro.factor("obs", log_prob)

    def log_prob_calc_sparse(self, y, total_counts, pred_values, input_conc):
        """Log-likelihood of the observed junction counts, summed over the observed positions.

        Written out with lgamma rather than via dist.Binomial / dist.BetaBinomial. Those build ~15-20
        intermediate tensors the size of nnz and autograd retains every one of them, once per ELBO
        particle -- at 38M observations x 5 particles that is enough to OOM a large GPU. This form is
        numerically identical (verified to 0 relative error) and ~5x faster.

        log(n choose k) depends only on the DATA, so it is precomputed once in from_anndata() as a
        scalar (`_log_binom_coeff`) instead of being rebuilt as an nnz-sized tensor on every step.
        """
        y_values = y._values()
        total_counts_values = total_counts._values()
        success_probs = pred_values.clamp(min=1e-6, max=1 - 1e-6)

        assert y_values.shape == total_counts_values.shape == success_probs.shape, (
            f"Shape mismatch: y={y_values.shape}, "
            f"total={total_counts_values.shape}, pred={success_probs.shape}"
        )

        log_binom = self._get_log_binom_coeff(y_values, total_counts_values)

        if torch.isinf(input_conc).any():
            # Binomial:  log(n C k) + k*log(p) + (n-k)*log(1-p)
            log_probs = (y_values * torch.log(success_probs)
                         + (total_counts_values - y_values) * torch.log1p(-success_probs))
        else:
            # Beta-Binomial:  log(n C k) + logB(k+a, n-k+b) - logB(a, b)
            alpha = success_probs * input_conc
            beta = input_conc - alpha            # == (1 - success_probs) * input_conc, one less temp
            log_probs = (torch.lgamma(y_values + alpha)
                         + torch.lgamma(total_counts_values - y_values + beta)
                         - torch.lgamma(total_counts_values + input_conc)
                         + torch.lgamma(input_conc)
                         - torch.lgamma(alpha)
                         - torch.lgamma(beta))

        return log_probs.sum() + log_binom

    def _get_log_binom_coeff(self, y_values, total_counts_values):
        """sum_i log(n_i choose k_i) -- constant in the parameters, so compute it once."""
        if getattr(self, "_log_binom_coeff", None) is None:
            with torch.no_grad():
                self._log_binom_coeff = (
                    torch.lgamma(total_counts_values + 1)
                    - torch.lgamma(y_values + 1)
                    - torch.lgamma(total_counts_values - y_values + 1)
                ).sum()
        return self._log_binom_coeff
    
    # There is now a single log-likelihood implementation; kept as an alias so older scripts that
    # called the "optimized" variant explicitly keep working.
    log_prob_calc_sparse_optimized = log_prob_calc_sparse

    def fit(self, guide):
        """
        Optimized training loop with better memory management for CPU.
        """

        # Calculate the learning rate decay factor per step
        lrd = self.gamma ** (1 / self.num_epochs)

        # Gradient clipping is an OPTIMIZER choice, not a device choice. It used to be switched on
        # only when enable_cpu_optimization was True, which meant CPU runs were clipped and GPU runs
        # (which pass the flag False) were not — an optimization difference hiding behind a flag
        # named after the device. Set clip_norm=None to disable.
        optimizer_args = {'lr': self.lr, 'lrd': lrd}
        if self.clip_norm is not None:
            optimizer_args['clip_norm'] = float(self.clip_norm)

        scheduler = ClippedAdam(optimizer_args)
        loss = Trace_ELBO(num_particles=self.ELBO_num_particles)

        # Apply obs_scale to the LOCAL sites on BOTH sides of the ELBO. In the model that is the
        # likelihood ("obs") and the phi prior ("assign"); in the guide it is q("assign"). Globals
        # (psi, pi, a, b, dir_conc, bb_conc) are untouched and stay at weight 1.
        if self.obs_scale != 1.0:
            scaled_model = _ScaleSites(self.obs_scale, {"assign", "obs"})(self.model)
            scaled_guide = _ScaleSites(self.obs_scale, {"assign"})(guide)
        else:
            scaled_model, scaled_guide = self.model, guide

        # SVI the step function minimizes the negative ELBO
        svi = SVI(scaled_model, scaled_guide, scheduler, loss)

        # NB: no pyro.clear_param_store() here. train() already clears it once per initialization,
        # and clearing again HERE would wipe any guide params train() has deliberately seeded --
        # which is exactly what the psi_scale warm-start needs to survive.

        # Initialize early stopping variables
        losses = []
        best_loss = float('inf')
        epochs_since_improvement = 0

        # Convert input_conc_prior to tensor if it's a float and move it to the device
        if isinstance(self.input_conc_prior, float):
            self.input_conc_prior = torch.tensor(self.input_conc_prior, device=self.device)
        elif isinstance(self.input_conc_prior, torch.Tensor):
            self.input_conc_prior = self.input_conc_prior.to(self.device)

        print(f"Training in progress for {self.num_epochs} epochs!")
        
        # Progress bar. Shown on every device — it used to be tied to enable_cpu_optimization, so a
        # GPU run (which passes that flag False) displayed nothing for the whole fit.
        pbar = tqdm(range(self.num_epochs), desc="Training")

        for epoch in pbar:

            # Call SVI step with dynamic passing of input_conc_prior
            with torch.enable_grad():  # Ensure gradients are enabled
                loss = svi.step(self.y, self.total_counts, self.K, self.junc_specific_prior, self.input_conc_prior)
            
            losses.append(loss)

            # Get current learning rate
            current_lr = list(scheduler.optim_objs.values())[0].param_groups[0]['lr']

            # Retrieve log likelihood from the model
            log_prob = self.current_log_prob if hasattr(self, "current_log_prob") else None

            if self.log_wandb and _WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    "ELBO_loss": loss,
                    "log_likelihood": log_prob})

            # Early stopping.
            # `min_delta` is ABSOLUTE. Against an ELBO of 1e6-1e9 a value like 0.01 (or even 100) is
            # a relative tolerance of ~1e-7, i.e. numerically inert, so the rule collapses to "stop
            # after `patience` epochs with no new GLOBAL best" -- which single-particle SVI noise
            # trips almost immediately. On the K=5 simulation this ended training at epoch 54 of 500,
            # long before the factors had separated. min_delta_rel makes the tolerance scale with the
            # loss; the effective threshold is the larger of the two.
            # NB best_loss starts at +inf, so the relative threshold must NOT be evaluated on the
            # first epoch: min_delta_rel * inf = inf, and `loss < inf - inf` is `loss < nan`, which is
            # ALWAYS False. best_loss would then never leave infinity, epochs_since_improvement would
            # increment on every epoch, and the fit would break at exactly `patience` no matter how
            # fast the ELBO was still descending -- turning early stopping into a hard epoch cap.
            # Seed best_loss from the first epoch instead.
            if not math.isfinite(best_loss):
                best_loss = loss
                epochs_since_improvement = 0
            else:
                threshold = max(self.min_delta, self.min_delta_rel * abs(best_loss))
                if loss < best_loss - threshold:
                    best_loss = loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

            if epochs_since_improvement >= self.patience:
                frac = (epoch + 1) / max(self.num_epochs, 1)
                print(f"\nEarly stopping at epoch {epoch} with loss: {loss:.4f}")
                if frac < 0.25:
                    print(
                        f"WARNING: stopped after only {frac:.0%} of num_epochs. With a non-convex "
                        f"factor model this usually means the run was killed before the K programs "
                        f"separated, not that it converged. Check psi.std(axis=0).mean() -- if it is "
                        f"~0 the programs are still identical. Raise `patience`, and set "
                        f"`min_delta_rel` (e.g. 1e-5) instead of relying on the absolute `min_delta`."
                    )
                break

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.2e}', 'lr': f'{current_lr:.4f}'})

            if epoch % self.print_epochs == 0 and not self.enable_cpu_optimization:
                print(f"Epoch {epoch}: Loss = {loss}")
                print(f"The current learning rate is {current_lr}")

        print(f"Training completed after {epoch + 1} epochs.")
        return losses

    def collect_samples(self, guide):
        """Collect samples from the trained guide"""
        samples = {}  # Dictionary to hold samples for each latent variable
        for _ in range(self.num_samples):

            # Generate a trace of the guide execution
            guide_trace = pyro.poutine.trace(guide).get_trace()
            # Collect samples from the trace
            for name, node in guide_trace.nodes.items():
                if node["type"] == "sample":
                    # Initialize the sample list if the variable is encountered for the first time
                    if name not in samples:
                        samples[name] = []
                    # Append the sample to the list, detached from the PyTorch computation graph
                    samples[name].append(node["value"].detach().cpu())

        # Convert lists of samples to torch tensors for easier downstream manipulation
        for name in samples:
            samples[name] = torch.stack(samples[name], dim=0) 
        return samples

    def calculate_summary_stats(self, samples):
        """Calculate summary statistics from samples"""
        stats = {}

        for name, values_tensor in samples.items():

            # Ensure the tensor is on CPU before converting to numpy
            if values_tensor.is_cuda:
                values_tensor = values_tensor.cpu()

            stats[name] = {
                'mean': torch.mean(values_tensor, dim=0).numpy(),
                'std': torch.std(values_tensor, dim=0).numpy()
            }
        return stats
    
    def extract_variable_sizes(self, *args, **kwargs):
        """
        Extract the sizes of the variables from the model.
        """
        
        trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        sizes = {}
        for name, node in trace.nodes.items():
            if node["type"] == "sample" and not node["is_observed"]:
                sizes[name] = node["value"].shape
        return sizes

    def train(self, num_initializations=3, seeds=None, psi_init=None, phi_init=None,
              psi_scale_init=None, save_to_file=False, file_prefix=None):
        """
        Train the model using SVI with CPU optimizations, collect samples, and return results.
        Main function to fit our Leaflet Bayesian beta-dirichlet factor model.

        Parameters:
        - num_initializations (int): Number of random initializations.
        - seeds (list, optional): List of seeds for random initializations.
        - psi_init (torch.Tensor, optional): Pre-initialized MEAN for psi (the guide's loc).
        - psi_scale_init (torch.Tensor, optional): Pre-initialized posterior WIDTH for psi (the
          guide's scale), shape (K, J). Carry this across chunks alongside psi_init, or q(psi) is
          reset to AutoDiagonalNormal's default width on every chunk and never accumulates evidence
          across the dataset -- which matters because that width is the posterior the
          differential-splicing FDR samples from.
        - phi_init (torch.Tensor, optional): Pre-initialized values for assign (phi).
        - save_to_file (bool): Whether to save the results to a file.
        - file_prefix (str, optional): Prefix for the saved file name.

        Returns:
        - all_results (list): List of results containing losses, latent variables, and summary statistics.
        - variable_sizes (dict): Dictionary containing sizes of model variables.
        """

        # Generate random seeds if not provided
        if seeds is None:
            seeds = [random.randint(1, 10000) for _ in range(num_initializations)]
            print(f"Random seeds: {seeds}")

        all_results = []
        pyro.clear_param_store()  # Clear previous optimizations
        all_params = []
        completed_inits = 0  # Track how many successful initializations

        # Track best initialization during training
        best_final_loss = float('inf')
        best_init_idx = 0

        # If num_initializations is 1, then save self.best_init to 0
        if num_initializations == 1:
            self.best_init = 0

        # Log model settings
        print(f"Training LeafletFA with {num_initializations} initializations.")
        print(f"Input concentration prior: {self.input_conc_prior}")
        print(f"Junction-specific prior: {self.junc_specific_prior}")
        print(f"Initial K to learn: {self.K}")
        
        if self.enable_cpu_optimization and self.device.type == 'cpu':
            print(f"✓ CPU optimizations active with {self.n_threads} threads")

        if self.waypoints_use:
            print(f"Initializing variational parameters with pre-defined PSI and PHI matrices!")
        
            # Dynamically select the correct PSI and PHI based on the value of K
            psi_key = f"psi_init_{self.K}_waypoints"
            phi_key = f"phi_init_{self.K}_waypoints"

            if psi_key in self.adata.varm and phi_key in self.adata.obsm:
                # Load the corresponding psi and phi initializations
                # .to(self.device): these come off AnnData as CPU arrays, and init_to_value uses
                # them to CREATE the guide's loc params -- so a CPU init pins the whole guide to
                # the CPU even on a GPU run.
                psi_init = torch.as_tensor(np.asarray(self.adata.varm[psi_key]), dtype=torch.float32,
                                           device=self.device).T  # (K, J)
                phi_init = torch.as_tensor(np.asarray(self.adata.obsm[phi_key]), dtype=torch.float32,
                                           device=self.device)
                print(f"Shape of PSI_init is {psi_init.shape}")
                print(f"Shape of PHI_init is {phi_init.shape}")
            else:
                raise ValueError(f"PSI and PHI initializations for {self.K} waypoints not found in adata.varm or adata.obsm.")
        else:
            # Preserve any psi_init/phi_init passed as train() arguments so they
            # reach the guide's init_to_value below. Previously this branch always
            # force-nulled them, silently discarding a caller-provided phi_init
            # (e.g. a fixed-psi E-step seeding phi). Only fall back to random init
            # when the caller supplied nothing.
            if phi_init is None and psi_init is None:
                print(f"Random initialization of variational parameters!")
            else:
                print(f"Using caller-provided psi_init/phi_init (non-waypoint init)")

        # init_to_value CREATES the guide's loc params from these tensors, so a CPU init would pin
        # the entire guide to the CPU even on a GPU run.
        if psi_init is not None:
            psi_init = torch.as_tensor(psi_init, dtype=torch.float32, device=self.device)
        if phi_init is not None:
            phi_init = torch.as_tensor(phi_init, dtype=torch.float32, device=self.device)
        # pi_init / alpha_pi_init are the CHUNK-TO-CHUNK warm start, and they arrive as CPU tensors
        # (run_training.py builds them with torch.tensor(global_pi)). They were the ONLY init tensors
        # not moved here, which pinned the guide's `pi` loc to the CPU on a GPU run: the model then
        # built Dirichlet(pi * dir_conc) on the CPU while the `assign` guide supplied a CUDA value,
        # and the `assign` site died with a device mismatch on the SECOND chunk -- i.e. only once the
        # warm start actually had something to carry, which is why chunk 1 always looked fine.
        if self.pi_init is not None:
            self.pi_init = torch.as_tensor(self.pi_init, dtype=torch.float32, device=self.device)
        if self.alpha_pi_init is not None:
            self.alpha_pi_init = torch.as_tensor(self.alpha_pi_init, dtype=torch.float32,
                                                 device=self.device)

        # Only empty CUDA cache if using a GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        for i, seed in enumerate(seeds):

            # Clear the parameter store for each initialization
            pyro.clear_param_store()  

            print("-------------------------------------------------")
            print(f"Initialization #{i+1} with seed {seed}", flush=True)
            print("-------------------------------------------------")

            # Set the seed for reproducibility
            pyro.set_rng_seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            # Also set numpy seed for consistency
            np.random.seed(seed)

            # Define the guide (AutoGuideList for variational inference)
            guide = AutoGuideList(self.model)

            # Only include 'psi' in the guide if self.fixed_psi is None
            if self.fixed_psi is None:
                if psi_init is not None:
                    guide.append(AutoDiagonalNormal(
                        block(self.model, expose=['psi']),
                        init_loc_fn=init_to_value(values={'psi': psi_init})
                    ))
                else:
                    guide.append(AutoDiagonalNormal(block(self.model, expose=['psi'])))

            # Guide for 'assign' with conditional initialization
            if phi_init is not None:
                guide.append(AutoDiagonalNormal(
                    block(self.model, expose=['assign']),
                    init_loc_fn=init_to_value(values={'assign': phi_init})
                ))
            else:
                guide.append(AutoDiagonalNormal(block(self.model, expose=['assign'])))

            # Guide for everything else (excluding 'psi' and 'assign'). With enough globals frozen
            # (fixed_psi + pi_fixed + fixed dir_conc + Binomial likelihood) nothing is left here, and
            # AutoDiagonalNormal over a model with no latent sites raises — so only append if needed.
            if self._has_global_latents():
                if self.pi_fixed is None and self.pi_init is not None and self.alpha_pi_init is not None:
                    guide.append(AutoDiagonalNormal(
                        block(self.model, hide=['psi', 'assign']),
                        init_loc_fn=init_to_value(values={'pi': self.pi_init, 'alpha_pi': self.alpha_pi_init})
                    ))
                else:
                    guide.append(AutoDiagonalNormal(block(self.model, hide=['psi', 'assign'])))

            # ── WARM-START THE GUIDE'S POSTERIOR WIDTH, not just its mean ────────────────────────
            # In chunked training a fresh LeafletFA (and so a fresh AutoGuide) is built per chunk, and
            # only psi's MEAN is carried across (via psi_init -> init_to_value). The guide's SCALE was
            # reset to AutoDiagonalNormal's default (0.1) on every one of the ~186 chunks, so q(psi)
            # never accumulated evidence: its width always reflected ONE batch, never the atlas.
            #
            # That width IS the posterior the differential-splicing analysis samples from, so it is
            # not a cosmetic issue -- the DS uncertainty was calibrated to 5,000 cells rather than
            # 92,894.
            #
            # Seeding it requires materialising the guide's params first (AutoGuide creates them
            # lazily on the first call), then writing the scale into the param store. fit() no longer
            # clears the store, so the seed survives.
            if psi_scale_init is not None and self.fixed_psi is None:
                guide(self.y, self.total_counts, self.K, self.junc_specific_prior,
                      self.input_conc_prior)                      # materialise the params
                ps = pyro.get_param_store()
                name = "AutoGuideList.0.scale"                    # sub-guide 0 is psi
                if name in ps:
                    seed_scale = torch.as_tensor(psi_scale_init, dtype=torch.float32,
                                                 device=self.device).reshape(-1)
                    if seed_scale.shape == ps[name].shape:
                        ps[name] = seed_scale.clamp(min=1e-6)
                        print(f"Warm-started q(psi) scale from the previous chunk "
                              f"(mean {float(seed_scale.mean()):.4f})")
                    else:
                        print(f"WARNING: psi_scale_init has shape {tuple(seed_scale.shape)} but the "
                              f"guide expects {tuple(ps[name].shape)} — ignoring the scale warm-start.")

            # Fit the model
            losses = self.fit(guide)
            
            # Track best initialization
            final_loss = losses[-1]
            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_init_idx = i

            # Optionally plot loss
            if self.loss_plot:
                plt.figure(figsize=(10, 6))
                plt.plot(losses)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")

                # Format y-axis in scientific notation
                plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation

                plt.title(f"Loss Plot for Initialization #{i+1} (Final: {final_loss:.2e})")
                plt.grid(True, alpha=0.3)

                if self.output_dir is not None:
                    plot_filename = f"loss_curve_seed_{seed}.png"
                    plot_filepath = os.path.join(self.output_dir, plot_filename)
                    plt.savefig(plot_filepath, dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"Loss plot saved to {plot_filepath}", flush=True)
                else:
                    plt.show()
                    print("No output directory specified, skipping loss plot save.")

            # Collect samples from the trained guide
            print(f"Collecting posterior samples for initialization #{i+1}", flush=True)
            all_samples = self.collect_samples(guide)

            # add all_samples to self object
            self.all_samples = all_samples

            # Compute summary statistics for latent variables
            print(f"Computing summary statistics for initialization #{i+1}", flush=True)
            summary_stats = self.calculate_summary_stats(all_samples)

            # Store results
            all_results.append({
                'seed': seed,
                'losses': losses,
                'latent_vars': all_samples,  # Store all sampled latent variables
                'summary_stats': summary_stats  # Store computed summary statistics
            })

            # Store the parameters from the ParamStoreDict
            params_copy = {name: pyro.get_param_store().get_param(name).detach().clone()
                           for name in pyro.get_param_store().get_all_param_names()}
            all_params.append(params_copy)

            # Move to the next initialization
            completed_inits += 1
            
            print(f"Initialization #{i+1} completed with final loss: {final_loss:.4e}")

        # Set best initialization if multiple were run
        if num_initializations > 1:
            self.best_init = best_init_idx
            print(f"\n✓ Best initialization: #{best_init_idx + 1} with loss {best_final_loss:.4e}")

        # Get model variable sizes
        variable_sizes = self.extract_variable_sizes()

        print("------------------------------------------------")
        print(f"Model variable sizes: {variable_sizes}", flush=True)
        print("------------------------------------------------")

        # Save results to file if required
        if save_to_file:
            print("Saving results to file.", flush=True)
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f"{file_prefix}_{date}_{self.K}_{num_initializations}_factors.pt" if file_prefix else f"results_{date}_{self.K}_{num_initializations}_factors.pt"
            
            # Save with CPU optimizations flag
            save_dict = {
                'results': all_results,
                'cpu_optimized': self.enable_cpu_optimization,
                'device': str(self.device),
                'data_density': self.data_density if hasattr(self, 'data_density') else None
            }
            torch.save(save_dict, file_name)
            print(f"Results saved to {file_name}", flush=True)

        # save these things to self
        self.latent_results = all_results
        self.all_params = all_params
        self.variable_sizes = variable_sizes
            
    def health_check(self, psi_contrast_min=0.04, phi_spread_min=1e-3, verbose=True):
        """Did this fit actually learn program structure? Call after `get_all_variables()`.

        The obvious diagnostics LIE. On a K=5 simulation where the model recovered nothing
        (psi-vs-truth r = 0.008, dominant-program agreement BELOW chance), it still reported
        "dead programs: 0/5" and "programs never dominant: 0/5" -- every program was alive and used.
        They were just all pointing at the same thing. Those checks ask "is each program alive?";
        the question you need answered is "are the programs DIFFERENT FROM EACH OTHER?".

        The failure mode is a symmetric saddle: AutoDiagonalNormal initialises every cell's phi at
        the SAME prior median and all K programs at nearly the same psi, and when phi is uniform the
        likelihood only constrains the AVERAGE of psi over programs -- so every psi_k gets an
        identical gradient and they cannot separate. See METHODS.md sec 11.

        Returns a dict; also stored on `self.health`. None of these need ground truth.

        psi_contrast : psi.std(axis=0).mean() -- spread of psi ACROSS the K programs. This IS the
                       differential-splicing effect size. Calibrated on the K=5 heart simulation
                       against ground truth (dominant-program agreement, chance = 20%):
                           0.0056 -> 15% agreement (chance)      lr=1e-3
                           0.0270 -> 16% agreement (chance)      lr=1e-2, gamma=1e-2
                           0.0443 -> 64% agreement               lr=5e-3, gamma=0.5
                           0.0738 -> 100% agreement, psi r=0.99  lr=5e-2, gamma=0.5
                       Hence the 0.04 default: 0.02 let a chance-level model through.
        phi_spread   : phi.std(axis=0).mean() -- do cells differ from each other, or is phi uniform?
        stopped_early: did early stopping fire in the first 25% of num_epochs? If so the run was
                       almost certainly killed before the programs separated.
        """
        def _np(x):
            # self.psi may be a CUDA tensor (e.g. when fixed_psi was supplied)
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        psi = _np(self.psi).astype(np.float64)
        phi = _np(self.assign_post).astype(np.float64)

        psi_contrast = float(psi.std(axis=0).mean())
        phi_spread = float(phi.std(axis=0).mean())

        n_ran = len(self.latent_results[self.best_init]["losses"]) if self.latent_results else 0
        frac_ran = n_ran / max(self.num_epochs, 1)
        stopped_early = bool(n_ran < self.num_epochs and frac_ran < 0.25)

        # PERPLEXITY vs its OWN PRIOR.
        # phi_c ~ Dirichlet(delta * pi), so with pi ~ 1/K the per-component concentration is delta/K.
        # At delta ~ 1 and K = 20 that is 0.05 -- a prior that already implies a median perplexity of
        # ~2.5 all by itself. If the FITTED phi has the same perplexity as draws from its own prior,
        # then phi carries no information beyond the prior and any "SP complexity" result is a
        # restatement of the prior, not a finding about the data.
        perp_fit = perp_prior = np.nan
        try:
            p = np.clip(phi, 1e-12, None)
            perp_fit = float(np.median(np.exp(-(p * np.log(p)).sum(axis=1))))
            delta_hat = float(np.asarray(getattr(self, "dir_conc_post", 1.0)))
            pi_hat = np.asarray(self.pi, dtype=np.float64)
            draws = np.random.default_rng(0).dirichlet(pi_hat * delta_hat, size=min(len(phi), 5000))
            d = np.clip(draws, 1e-12, None)
            perp_prior = float(np.median(np.exp(-(d * np.log(d)).sum(axis=1))))
        except Exception:
            pass

        # NB perplexity-vs-prior is REPORTED, not a pass/fail. A fitted phi whose perplexity matches
        # its prior is NOT necessarily uninformative: if the data really were generated with that
        # much mixing, the marginals coincide legitimately (verified -- a model that recovers the
        # true psi to r>0.4 still shows perp_fit 3.09 vs perp_prior 2.96). It is a CAVEAT for any
        # claim ABOUT perplexity (e.g. "SP complexity = 2.46"), not evidence of a broken fit: you
        # cannot tell a biological complexity finding from a restatement of Dirichlet(delta*pi)
        # using perplexity alone. To make that claim, show phi is informative about cell type.
        problems = []
        if psi_contrast < psi_contrast_min:
            problems.append(
                f"psi_contrast={psi_contrast:.4f} < {psi_contrast_min}: the {self.K} programs are "
                f"nearly IDENTICAL to each other (symmetric collapse). This model has not learned "
                f"program structure, however good its ELBO looks."
            )
        elif psi_contrast < 2 * psi_contrast_min:
            problems.append(
                f"psi_contrast={psi_contrast:.4f} is only marginally above the collapse threshold "
                f"({psi_contrast_min}). The programs have separated a little but may still be mostly "
                f"wrong -- psi_contrast is NECESSARY but NOT SUFFICIENT: it detects TOTAL collapse "
                f"(~0.003 vs ~0.10) but not a PARTIAL failure. Observed on the K=5 simulation: "
                f"psi_contrast=0.045 passed a 0.04 gate while true-psi recovery was only r=0.22 and "
                f"differential-vs-null AUROC was 0.525 (chance). Do not trust this fit without an "
                f"external check (held-out imputation, or ground truth if you have it)."
            )
        if phi_spread < phi_spread_min:
            problems.append(
                f"phi_spread={phi_spread:.5f} < {phi_spread_min}: every cell has essentially the "
                f"same program mixture (phi collapsed to uniform)."
            )
        if stopped_early:
            problems.append(
                f"early stopping fired at epoch {n_ran} of {self.num_epochs} ({frac_ran:.0%}): the "
                f"run was probably killed before the programs separated. Check `min_delta` -- an "
                f"ABSOLUTE threshold against an ELBO of 1e6-1e9 is inert; use `min_delta_rel`."
            )

        self.health = {
            "psi_contrast": psi_contrast,
            "phi_spread": phi_spread,
            "phi_perplexity": perp_fit,
            "phi_perplexity_prior": perp_prior,   # what the prior alone would give
            "epochs_ran": n_ran,
            "stopped_early": stopped_early,
            "healthy": not problems,
            "problems": problems,
        }

        if verbose:
            status = "OK" if not problems else "FAILED"
            print(f"\n--- model health: {status} ---")
            print(f"  psi_contrast (programs differ from each other) : {psi_contrast:.4f}")
            print(f"  phi_spread   (cells differ from each other)    : {phi_spread:.5f}")
            print(f"  phi perplexity  (fitted / prior-alone)         : {perp_fit:.2f} / {perp_prior:.2f}")
            print(f"  epochs ran                                    : {n_ran} / {self.num_epochs}")
            for p in problems:
                print(f"  !! {p}")
            if problems:
                print("  Do not trust this fit. See METHODS.md sec 11 (learning rate is the usual cause).")
        return self.health

    def get_best_initialization(self):
        """
        Select the best initialization based on the lowest loss.
        """
        # Score each init by its BEST loss, not its LAST. Under early stopping the last epoch is by
        # definition a non-improving one (it is the epoch that tripped `patience`), so `losses[-1]`
        # is systematically worse than the run actually achieved -- and it is noisier, since it is a
        # single Monte-Carlo ELBO estimate rather than the minimum over the run. This number drives
        # model selection across the whole hyperparameter grid, so the bias matters.
        finals = [min(result["losses"]) for result in self.latent_results]
        best_init = int(np.argmin(finals))
        best_elbo = finals[best_init]
        print(f"The best initialization was {best_init} with an ELBO of {best_elbo:.4e}")

        self.best_init = best_init
        self.best_elbo = best_elbo
        # Keep the final-epoch loss too: a large gap between the two means the run was still
        # descending (or is very noisy) when it stopped.
        self.final_elbo = self.latent_results[best_init]["losses"][-1]

    def get_psi_variational_params(self, initialization=None):
        """
        Extracts PSI variational parameters if PSI was learned.
        If PSI was fixed, it simply uses the fixed PSI.
        """
    
        if initialization is None:
            initialization = self.best_init
    
        if self.fixed_psi is not None:
            print("PSI was fixed during inference. Using the provided fixed PSI instead of learned variational parameters.")
            self.psi = self.fixed_psi.to(self.device)  # Use the fixed PSI directly
            self.psis_loc = None  # No variational parameters available
            self.psis_scale = None
            self.a = None
            self.b = None
            self.a_shape = None
            self.a_rate = None
            self.b_shape = None
            self.b_rate = None
        else:
            # Extract variational parameters if PSI was learned
            print("Extracting variational parameters for PSI...")
            self.psis_loc = self.all_params[initialization]["AutoGuideList.0.loc"].reshape(self.K, self.num_junctions)
            self.psis_scale = self.all_params[initialization]["AutoGuideList.0.scale"].reshape(self.K, self.num_junctions)
            
            # Latent variables sampled from the guide
            self.psi = self.latent_results[initialization]["summary_stats"]["psi"]["mean"]
            self.a = self.latent_results[initialization]["summary_stats"]["a"]["mean"]
            self.b = self.latent_results[initialization]["summary_stats"]["b"]["mean"]
            self.a_shape = self.latent_results[initialization]["summary_stats"]["a_shape"]["mean"]
            self.a_rate = self.latent_results[initialization]["summary_stats"]["a_rate"]["mean"]
            self.b_shape = self.latent_results[initialization]["summary_stats"]["b_shape"]["mean"]
            self.b_rate = self.latent_results[initialization]["summary_stats"]["b_rate"]["mean"]
    
        print("PSI parameters extraction completed.")

    def get_pi(self, initialization=None):
        """
        Extract the Pi vector from the trained model.
        """
        if initialization is None:
            initialization = self.best_init

        stats = self.latent_results[initialization]["summary_stats"]

        if self.pi_fixed is not None:
            # pi was frozen, so it has no posterior — carry the fixed values straight through.
            self.pi = self.pi_fixed.cpu().numpy()
            self.alpha_pi = (self.alpha_pi_fixed.cpu().numpy()
                             if self.alpha_pi_fixed is not None else None)
        else:
            self.alpha_pi = stats["alpha_pi"]["mean"]
            self.pi = stats["pi"]["mean"]

        # The learned dir_conc goes to dir_conc_post, NOT to self.dir_conc. Writing it back to
        # self.dir_conc would flip the model from sampling dir_conc to treating it as fixed on any
        # subsequent model() call, silently changing the model structure mid-object.
        if self.dir_conc is None:
            self.dir_conc_post = self.latent_results[initialization]["summary_stats"]["dir_conc"]["mean"]
        else:
            self.dir_conc_post = self.dir_conc

    def get_bb_conc(self, initialization=None):
        """Extract beta-binomial concentration parameter"""
        if initialization is None:
            initialization = self.best_init

        # bb_conc is a LATENT whenever the prior is None or a Distribution -- read the posterior.
        # Otherwise it was FIXED to the supplied value, which is only "infinity" when that value
        # actually is inf. Reporting "infinity" for a finite fixed concentration (e.g. 100) threw
        # away the number the model was actually run with.
        if self._conc_is_sampled():
            self.bb_conc = self.latent_results[initialization]["summary_stats"]["bb_conc"]["mean"]
        else:
            fixed = torch.as_tensor(self.input_conc_prior, dtype=torch.float32)
            self.bb_conc = "infinity" if torch.isinf(fixed).any() else float(fixed)

    def get_latent_representation(self, initialization=None):
        """
        Extract the latent representation from the trained model.
        """
        if initialization is None:
            initialization = self.best_init
        
        latent_vars = self.latent_results[initialization]['summary_stats']
        assign_post = latent_vars["assign"]["mean"]
        self.assign_post = assign_post  

    def get_all_variables(self, initialization=None):
        """
        Extract all variables from the best initialization:
        This includes PSI, PI, BB_CONC, and the latent representation PHI.
        """

        self.get_best_initialization()

        if initialization is None:
            initialization = self.best_init

        print(f"Extracting all variables from initialization #{initialization}")

        print(f"Extracting PSI variational parameters...")
        self.get_psi_variational_params(initialization)

        print(f"Extracting the learned beta-binomial concentration (if applicable)...")
        self.get_bb_conc(initialization)

        print(f"Extracting the latent representation, C x K matrix of factor activities...")
        self.get_latent_representation(initialization)

        print(f"Extracting the learned Pi vector...")
        self.get_pi(initialization)

        # Always run the health check. It is cheap, needs no ground truth, and it is the ONLY thing
        # that catches a symmetric collapse -- a fit whose loss curve and "dead programs: 0/K" both
        # look perfectly healthy while the K programs are identical and the model has learned
        # nothing. Callers can gate on `self.health["healthy"]`.
        self.health_check()

    def prune_K(self, threshold=0.05):
        """Prunes latent variables based on a threshold."""
        
        print(f"The K before pruning is {self.K}")
        
        pruned_indices = np.where(self.pi > threshold)[0]
        pruned_pi = self.pi[pruned_indices] / self.pi[pruned_indices].sum() # Re-normalize
        # update self 
        self.pi = pruned_pi
        
        pruned_assign_post = self.assign_post[:, pruned_indices]
        pruned_assign_post = pruned_assign_post / pruned_assign_post.sum(axis=1, keepdims=True) # Re-normalize factor activities in cells 
        self.assign_post = pruned_assign_post

        psis_loc_pruned = self.psis_loc[pruned_indices]
        psis_scale_pruned = self.psis_scale[pruned_indices]
        self.psis_loc = psis_loc_pruned
        self.psis_scale = psis_scale_pruned

        psi_learned_pruned = self.psi[pruned_indices, :]
        self.psi_learned = psi_learned_pruned
        self.psi = psi_learned_pruned

        # Need to also prune the sampled PSI values from the guide
        pruned_psi_samples = self.all_samples["psi"][:, pruned_indices, :] # samples x K x junctions
        self.psi_samples = pruned_psi_samples
        pruned_phi_samples = self.all_samples["assign"][:, :, pruned_indices] # samples x cells x K
        self.phi_samples = pruned_phi_samples

        # Get a mean and std of the pruned psi samples for every factor-junction
        psi_samples_mean = pruned_psi_samples.mean(dim=0)
        psi_samples_std = pruned_psi_samples.std(dim=0)
        self.psi_samples_mean = psi_samples_mean
        self.psi_samples_std = psi_samples_std

        # a and b are NOT pruned: with junc_specific_prior they are per-junction (shape [J]) and
        # broadcast across all K factors, so they carry no factor axis. Indexing them with factor
        # indices silently returned a[0..K-1] -- the priors of arbitrary junctions.

        print(f"The K after pruning is {len(pruned_indices)}")
        print(f"Updating K to {len(pruned_indices)} in the LeafletFA object.")
        self.K = len(pruned_indices)


# Usage example with CPU optimizations:
if __name__ == "__main__":
    # Example of how to use the optimized LeafletFA
    
    # Load your data
    # adata = load_your_anndata()
    
    # Create model with CPU optimizations enabled (default)
    # model = LeafletFA(
    #     adata, 
    #     K=20,
    #     num_epochs=500,
    #     enable_cpu_optimization=True  # This is the key flag
    # )
    
    # Process data
    # model.from_anndata()
    
    # For GPU: initialize Triton mask
    # if model.device.type == 'cuda':
    #     model.initialize_triton_mask()
    
    # Train the model
    # model.train(num_initializations=3)
    
    # Extract results
    # model.get_all_variables()
    
    print("LeafletFA with CPU optimizations loaded successfully!")
    print("Key optimizations included:")
    print("- Automatic thread configuration for CPU")
    print("- Optimized sparse/dense operations with einsum")
    print("- Vectorized log probability calculations")
    print("- Memory-efficient index caching")
    print("- Progress bars for training monitoring")
    print("- Improved numerical stability")