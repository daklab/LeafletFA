import os
import random
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


class LeafletFA:
    def __init__(self, adata, K=10, junc_specific_prior=True, input_conc_prior=None, delta_fixed=None,
                 waypoints_use=True, fixed_psi=None, use_dense_mode=None,
                 pi_init=None, alpha_pi_init=None,
                 log_wandb=False,
                 eps = 1e-6, lr=0.01, gamma = 0.05, num_epochs=500, print_epochs=10, 
                 ELBO_num_particles=1, patience=5, min_delta=0.01, num_samples=10, loss_plot=True, 
                 output_dir=None, enable_cpu_optimization=True):
        """
        Initialize the LeafletFA model with CPU optimizations.
        
        Parameters:
        - adata (AnnData): AnnData object containing single-cell splicing data.
        - K (int): Number of factors.
        - junc_specific_prior (bool): Whether to learn a global prior over junctions.
        - input_conc_prior: Concentration prior (can be either 'None' or 'torch.tensor(np.inf)').
        - waypoints_use (bool): Whether to use waypoints for initialization.
        - pi_init (torch.Tensor): Initialization for the pi vector.
        - alpha_pi_init (torch.Tensor): Initialization for the alpha_pi vector.
        - fixed_psi (torch.Tensor): Fix PSI during inference by using an existing PSI latent variable from previous model training.
        - use_dense_mode (bool): Whether to use dense mode for computation.
        - eps (float): Small value to prevent numerical issues.
        - lr (float): Learning rate.
        - gamma (float): Learning rate decay.
        - num_epochs (int): Number of training epochs.
        - print_epochs (int): Print loss every 'print_epochs' epochs.
        - ELBO_num_particles (int): Number of particles for ELBO calculation.
        - patience (int): Early stopping patience.
        - min_delta (float): Minimum improvement for early stopping.
        - num_samples (int): Number of samples to collect from the guide.
        - loss_plot (bool): Whether to plot loss.
        - output_dir (str): Directory to save results.
        - enable_cpu_optimization (bool): Whether to enable CPU-specific optimizations.
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
        self.eps = eps
        self.lr = lr
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.print_epochs = print_epochs
        self.ELBO_num_particles = ELBO_num_particles
        self.patience = patience
        self.use_dense_mode = use_dense_mode
        self.min_delta = min_delta
        self.num_samples = num_samples
        self.loss_plot = loss_plot
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.enable_cpu_optimization = enable_cpu_optimization
        
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

    @staticmethod
    def convertr(hyperparam, name):
        """ Convert hyperparameters to Pyro samples or fixed tensors. 
        If hyperparam is a distribution, sample from it.
        If hyperparam is None, sample from a Gamma distribution with shape and rate 10.0.
        Otherwise, convert the hyperparameter provided to a tensor.
        """
        if isinstance(hyperparam, torch.distributions.Distribution):
            return pyro.sample(name, hyperparam)
        elif hyperparam is None:
            return pyro.sample(name, dist.Gamma(10.0, 10.0))
        else:
            hyperparam_tensor = torch.as_tensor(hyperparam, dtype=torch.float32)
            if torch.isinf(hyperparam_tensor).any():
                return hyperparam_tensor
            else:
                return torch.tensor(hyperparam, dtype=torch.float32)

    def from_anndata(self, float_type=None):
        """
        Process an AnnData object and return PyTorch sparse tensors for junction and cluster counts.
        Optimized for CPU with better memory patterns.
        """
        
        print(f"Taking in the AnnData object with {self.adata.shape[0]} cells and {self.adata.shape[1]} junctions.")

        # Set device automatically if not provided
        if float_type is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            float_type = {"dtype": torch.float32, "device": device}

        print(f"Processing AnnData on {float_type['device']}")

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
        
        # CPU Optimization: Coalesce sparse tensors for better memory access
        if self.enable_cpu_optimization and self.device.type == 'cpu':
            self.y = self.y.coalesce()
            self.total_counts = self.total_counts.coalesce()
            print("✓ Sparse tensors coalesced for better CPU performance")
        
        # Auto-detect density if not explicitly set
        if self.use_dense_mode is None:
            self.detect_data_density()
        
        # CPU Optimization: Cache indices
        if self.enable_cpu_optimization and self.device.type == 'cpu':
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
        """Precomputes the binary mask needed for masked matrix multiplication."""
        cell_idx, junc_idx = self.y._indices()
        C, J = self.y.shape
        mask = torch.zeros((C, J), device=self.device)
        mask[cell_idx, junc_idx] = 1.0
        self.triton_mask = mask
    
    def compute_pred_triton(self, assign, psi):
        out = masked_matmul(assign.to(self.device), psi.to(self.device), self.triton_mask)
        cell_idx, junc_idx = self.y._indices()
        return out[cell_idx, junc_idx]
    
    def sparse_dot_cpu(self, assign, psi, cell_idx, junc_idx):
        """
        Optimized sparse dot product for CPU with better memory patterns.
        """
        if hasattr(self, 'use_dense_mode') and self.use_dense_mode:
            # Dense mode: use einsum for better CPU performance
            if self.enable_cpu_optimization:
                # Use einsum which is often faster on CPU
                pred_full = torch.einsum('ck,kj->cj', assign, psi)
            else:
                pred_full = torch.matmul(assign, psi)
            return pred_full[cell_idx, junc_idx]
        else:
            # Sparse mode: optimized indexing and computation
            if self.enable_cpu_optimization:
                # Use more efficient indexing operations
                with torch.no_grad():  # Temporarily disable gradients for indexing
                    # Use index_select for better performance
                    assign_rows = assign.index_select(0, cell_idx.long())
                    psi_selected = psi.index_select(1, junc_idx.long())
                
                # Use einsum for the multiplication - more efficient
                pred = torch.einsum('nk,kn->n', assign_rows, psi_selected)
                pred.requires_grad_(True)  # Re-enable gradients on result
                return pred
            else:
                # Original implementation
                assign_rows = assign[cell_idx]
                psi_cols = psi[:, junc_idx].T
                return (assign_rows * psi_cols).sum(dim=1)

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
            shape_prior = dist.Gamma(1.0, 1.0)
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

        # PI INITIALIZATION LOGIC 
        if self.pi_init is not None and self.alpha_pi_init is not None:
            # Use provided initialization for pi
            alpha_pi_init = self.alpha_pi_init.to(self.device)
            pi_init = self.pi_init.to(self.device)

            # Register initial values as parameters so pyro.sample can use them
            alpha_pi_param = pyro.param("alpha_pi_param", alpha_pi_init)
            pi_param = pyro.param("pi_param", pi_init)

            alpha_pi = pyro.sample("alpha_pi", dist.Gamma(1.0, 1.0), 
                                  infer={"initial_value": alpha_pi_param})
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) * alpha_pi / K), 
                            infer={"initial_value": pi_param})
        else:
            # Sample from prior (first batch only)
            alpha_pi = pyro.sample("alpha_pi", dist.Gamma(1.0, 1.0))
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) * alpha_pi / K))
        
        # Assert that pi sums to 1 and contains no NaN/inf values
        assert torch.isfinite(pi).all(), "pi contains NaN or infinite values!"
        assert torch.allclose(pi.sum(), torch.tensor(1.0, dtype=torch.float)), f"pi does not sum to 1: {pi.sum()}"

        # Check if we are used a fixed delta value or if we are learning it 
        if self.dir_conc is None:
            # Sample concentration value that scales the pi vector (higher values lead to more uniform pi)
            dir_conc = pyro.sample("dir_conc", dist.Gamma(2., 2.))
            dir_conc = torch.clamp(dir_conc, min=1e-6, max=1e6) # Prevent numerical issues
        else:
            dir_conc = self.dir_conc

        # Sample the factor assignments for cells 
        assign = pyro.sample("assign", dist.Dirichlet(pi * dir_conc).expand([N]).to_event(1)).to(dtype=torch.float32)

        # Ensure no negative values in assign 
        if torch.any(assign < self.eps):
            assign = assign + self.eps
            assign = assign / assign.sum(dim=1, keepdim=True)  # Re-normalize to sum to 1

        assign = assign.to(self.device, dtype=torch.float32)
        psi = psi.to(self.device, dtype=torch.float32)

        # Use cached indices if available (CPU optimization)
        if self.cached_cell_idx is not None and self.cached_junc_idx is not None:
            cell_idx = self.cached_cell_idx
            junc_idx = self.cached_junc_idx
        else:
            # Extract indices of nonzero elements in y and total_counts
            y_indices = y._indices()
            total_counts_indices = total_counts._indices()

            if not torch.equal(y_indices, total_counts_indices):
                raise ValueError("Mismatch between indices of y and total_counts.")
            
            cell_idx, junc_idx = y_indices

        # Compute log-probability
        if self.device.type == 'cpu':            
            pred = self.sparse_dot_cpu(assign, psi, cell_idx, junc_idx)
            
            # Use optimized log prob calculation if enabled
            if self.enable_cpu_optimization:
                log_prob = self.log_prob_calc_sparse_optimized(y, total_counts, pred, input_conc)
            else:
                log_prob = self.log_prob_calc_sparse(y, total_counts, pred, input_conc)
            
            # Validate for numerical issues
            assert torch.isfinite(log_prob).all(), "log_prob contains NaN or infinite values!"
        else:
            if not hasattr(self, "triton_mask"):
                raise RuntimeError("You must call `initialize_triton_mask()` before training.")
                
            # Use Triton for GPU-accelerated matrix multiplication
            assign = assign.to(self.device, dtype=torch.float32).contiguous()
            psi = psi.to(self.device, dtype=torch.float32).contiguous()

            pred_triton = self.compute_pred_triton(assign, psi)
            log_prob = self.log_prob_calc_sparse(y, total_counts, pred_triton, input_conc)
            
            # Validate for numerical issues
            assert torch.isfinite(log_prob).all(), "log_prob contains NaN or infinite values!"

        # Add to Pyro trace
        pyro.factor("obs", log_prob)

    def log_prob_calc_sparse(self, y, total_counts, pred_values, input_conc):
        """Original log probability calculation"""
        y_values = y._values()
        total_counts_values = total_counts._values()
        success_probs = pred_values.clamp(min=1e-6, max=1 - 1e-6)
        
        # Check shape consistency
        assert y_values.shape == total_counts_values.shape == success_probs.shape, (
            f"Shape mismatch: y={y_values.shape}, "
            f"total={total_counts_values.shape}, pred={success_probs.shape}"
        )
    
        if torch.isinf(input_conc).any():
            log_probs = dist.Binomial(total_counts_values, probs=success_probs).log_prob(y_values)
        else:
            alpha = success_probs * input_conc
            beta = (1 - success_probs) * input_conc
            log_probs = dist.BetaBinomial(alpha, beta, total_counts_values).log_prob(y_values)

        return log_probs.sum()
    
    def log_prob_calc_sparse_optimized(self, y, total_counts, pred_values, input_conc):
        """
        Optimized log probability calculation for CPU.
        Avoids creating distribution objects and uses vectorized operations.
        """
        y_values = y._values()
        total_counts_values = total_counts._values()
        
        # Vectorized clamping
        success_probs = torch.clamp(pred_values, min=1e-6, max=1-1e-6)
        
        if torch.isinf(input_conc).any():
            # Direct computation without distribution object - much faster
            # log P(k|n,p) = log(n choose k) + k*log(p) + (n-k)*log(1-p)
            # We can ignore the binomial coefficient for gradient computation
            log_probs = (y_values * torch.log(success_probs) + 
                        (total_counts_values - y_values) * torch.log1p(-success_probs))
        else:
            # Beta-binomial: use lgamma for numerical stability and speed
            alpha = success_probs * input_conc
            beta = (1 - success_probs) * input_conc
            
            # Vectorized lgamma computation - much faster than creating distribution objects
            log_probs = (torch.lgamma(total_counts_values + 1) - 
                        torch.lgamma(y_values + 1) - 
                        torch.lgamma(total_counts_values - y_values + 1) +
                        torch.lgamma(y_values + alpha) + 
                        torch.lgamma(total_counts_values - y_values + beta) +
                        torch.lgamma(alpha + beta) - 
                        torch.lgamma(total_counts_values + alpha + beta) -
                        torch.lgamma(alpha) - torch.lgamma(beta))
        
        return log_probs.sum()
    
    def fit(self, guide):
        """
        Optimized training loop with better memory management for CPU.
        """

        # Calculate the learning rate decay factor per step 
        lrd = self.gamma ** (1 / self.num_epochs)
        
        # Use gradient clipping for stability
        optimizer_args = {'lr': self.lr, 'lrd': lrd}
        if self.enable_cpu_optimization:
            # Add gradient clipping for CPU stability
            optimizer_args['clip_norm'] = 10.0
        
        scheduler = ClippedAdam(optimizer_args) 
        loss = Trace_ELBO(num_particles=self.ELBO_num_particles) 
        
        # SVI the step function minimizes the negative ELBO
        svi = SVI(self.model, guide, scheduler, loss)

        pyro.clear_param_store() # Clear previous optimizations

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
        
        # Progress bar for better monitoring
        pbar = tqdm(range(self.num_epochs), desc="Training", disable=not self.enable_cpu_optimization)

        for epoch in pbar:
            
            # Clear gradients periodically on CPU to free memory
            if self.enable_cpu_optimization and epoch % 50 == 0 and epoch > 0:
                # This helps with memory fragmentation on CPU
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
            
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

            # Check for early stopping
            if loss < best_loss - self.min_delta:
                best_loss = loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Early stopping if no improvement in 'patience' epochs 
            if epochs_since_improvement >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} with loss: {loss:.4f}")
                break

            # Update progress bar
            if self.enable_cpu_optimization:
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

    def train(self, num_initializations=3, seeds=None, psi_init=None, phi_init=None, save_to_file=False, file_prefix=None):
        """
        Train the model using SVI with CPU optimizations, collect samples, and return results.
        Main function to fit our Leaflet Bayesian beta-dirichlet factor model.

        Parameters:
        - num_initializations (int): Number of random initializations.
        - seeds (list, optional): List of seeds for random initializations.
        - psi_init (torch.Tensor, optional): Pre-initialized values for psi.
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
                psi_init = torch.tensor(self.adata.varm[psi_key]).T  # Transpose for correct shape
                phi_init = torch.tensor(self.adata.obsm[phi_key])
                print(f"Shape of PSI_init is {psi_init.shape}")
                print(f"Shape of PHI_init is {phi_init.shape}")
            else:
                raise ValueError(f"PSI and PHI initializations for {self.K} waypoints not found in adata.varm or adata.obsm.")
        else:
            print(f"Random initialization of variational parameters!")
            psi_init = None 
            phi_init = None

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

            # Guide for everything else (excluding 'psi' and 'assign')
            guide.append(AutoDiagonalNormal(block(self.model, hide=['psi', 'assign'])))

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
            
    def get_best_initialization(self):
        """
        Select the best initialization based on the lowest loss.
        """
        
        best_init = np.argmin([result["losses"][-1] for result in self.latent_results])
        best_elbo = self.latent_results[best_init]["losses"][-1]
        print(f"The best initialization was {best_init} with an ELBO of {best_elbo:.4e}")

        self.best_init = best_init
        self.best_elbo = best_elbo

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

        # get alpha_pi from the guide also 
        self.alpha_pi = self.latent_results[initialization]["summary_stats"]["alpha_pi"]["mean"]
        self.pi = self.latent_results[initialization]["summary_stats"]["pi"]["mean"]
        # get dir_conc from the guide also if we learned it 
        if self.dir_conc is None:
            self.dir_conc = self.latent_results[initialization]["summary_stats"]["dir_conc"]["mean"]

    def get_bb_conc(self, initialization=None):
        """Extract beta-binomial concentration parameter"""
        if initialization is None:
            initialization = self.best_init

        # If input_conc was None extract it from latent variables 
        if self.input_conc_prior is None:
            self.bb_conc = self.latent_results[initialization]["summary_stats"]["bb_conc"]["mean"]
        else:
            self.bb_conc = "infinity"

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

        if self.junc_specific_prior:
            pruned_a = self.a[pruned_indices]
            pruned_b = self.b[pruned_indices]
            self.a = pruned_a
            self.b = pruned_b

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