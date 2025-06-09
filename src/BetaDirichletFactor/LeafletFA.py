import os
import random
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import ticker for scientific notation

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
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader
import wandb 
from tqdm import tqdm

# Configure logging
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Disable default argument validation for distributions
Distribution.set_default_validate_args(False)

# Print Torch and CUDA version info
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")

# Load leaflet_fa_torch_ops
# from leaflet_fa_torch_ops import masked_matmul
# from BetaDirichletFactor.leaflet_fa_torch_ops import masked_matmul
import sys
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src')
from BetaDirichletFactor.leaflet_fa_torch_ops import masked_matmul

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
class LeafletFA:
    def __init__(self, adata, K=50, junc_specific_prior=True, input_conc_prior=None, delta_fixed=None,
                 waypoints_use=True, fixed_psi=None, 
                 pi_init=None, alpha_pi_init=None,
                 log_wandb=False,
                 eps = 1e-6, lr=0.01, gamma = 0.05, num_epochs=500, print_epochs=10, 
                 ELBO_num_particles=1, patience=5, min_delta=0.01, num_samples=10, loss_plot=True, output_dir=None):
        """
        Initialize the LeafletFA model.
        
        Parameters:
        - adata (AnnData): AnnData object containing single-cell splicing data.
        - K (int): Number of factors.
        - junc_specific_prior (bool): Whether to learn a global prior over junctions.
        - input_conc_prior: Concentration prior (can be either 'None' or 'torch.tensor(np.inf)').
        - waypoints_use (bool): Whether to use waypoints for initialization.
        - pi_init (torch.Tensor): Initialization for the pi vector.
        - alpha_pi_init (torch.Tensor): Initialization for the alpha_pi vector.
        - fixed_psi (torch.Tensor): Fix PSI during inference by using an existing PSI latent variable from previous model training.
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
        self.min_delta = min_delta
        self.num_samples = num_samples
        self.loss_plot = loss_plot
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        
        # Initialize other attributes
        self.guide = None
        self.losses = []
        self.variable_sizes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        Parameters:
        - adata (AnnData): AnnData object containing single-cell data.
        - float_type (dict): Dictionary specifying dtype and device (default: float32 on CPU).

        Returns:
        - full_y_tensor (torch.sparse_coo_tensor): Sparse tensor of junction counts.
        - full_total_counts_tensor (torch.sparse_coo_tensor): Sparse tensor of total counts.
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

        # Convert layers to COO sparse matrices only if they are not already sparse
        if not isinstance(self.adata.layers["cell_by_cluster_matrix"], coo_matrix):
            self.adata.layers["Cluster_Counts"] = coo_matrix(self.adata.layers["cell_by_cluster_matrix"])
        else:
            self.adata.layers["Cluster_Counts"] = self.adata.layers["cell_by_cluster_matrix"]

        if not isinstance(self.adata.layers["cell_by_junction_matrix"], coo_matrix):
            self.adata.layers["Junction_Counts"] = coo_matrix(self.adata.layers["cell_by_junction_matrix"])
        else:
            self.adata.layers["Junction_Counts"] = self.adata.layers["cell_by_junction_matrix"]

        # Extract cell and junction indices
        cell_index_array = np.array(self.adata.layers["Junction_Counts"].row)
        junc_index_array = np.array(self.adata.layers["Junction_Counts"].col)

        # Convert indices to PyTorch tensors
        cell_index_tensor = torch.tensor(cell_index_array, dtype=torch.int32, device=float_type["device"])
        junc_index_tensor = torch.tensor(junc_index_array, dtype=torch.int32, device=float_type["device"])

        # Convert sparse matrix data to PyTorch tensor
        ycount_array = np.array(self.adata.layers["Junction_Counts"].data)
        ycount = torch.tensor(ycount_array, **float_type)

        # Create sparse tensor for junction counts
        full_y_tensor = torch.sparse_coo_tensor(
            indices=torch.stack([cell_index_tensor, junc_index_tensor]), 
            values=ycount,
            size=(len(self.adata.obs), len(self.adata.var)),
            device=float_type["device"]
        )

        # clean up memory
        del ycount
        torch.cuda.empty_cache()

        # Extract cluster counts (ATSE counts for all junctions in it)
        total_counts_array = np.array(self.adata.layers["Cluster_Counts"].data)
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
        torch.cuda.empty_cache()
        
        # Add the tensors to the self object
        self.y = full_y_tensor
        self.total_counts = full_total_counts_tensor

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
        Compute output[k] = assign[cell_idx[k]] @ psi[:, junc_idx[k]]
        Returns a 1D tensor of shape (nnz,)
        """
        assign_rows = assign[cell_idx]             # shape: (nnz, K)
        psi_cols = psi[:, junc_idx].T              # shape: (nnz, K)
        return (assign_rows * psi_cols).sum(dim=1) # shape: (nnz,)

    def model(self, y=None, total_counts=None, K=None, 
              junc_specific_prior=None, input_conc_prior=None):

        """
        Define a probabilistic Bayesian model using a Beta-Dirichlet factorization.

        Parameters:
        - y (torch.Tensor): Sparse tensor of observed junction counts.
        - total_counts (torch.Tensor): Sparse tensor of total intron cluster counts.
        - K (int): Number of factors representing cell states.
        - junc_specific_prior (bool): Whether to use junction-specific priors.
        - input_conc_prior (float, torch.Tensor, or None): Prior for the concentration parameter.

        Returns:
        - None (Pyro automatically tracks distributions)
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
            
            # Sample Pi prior from a Dirichlet distribution
            # First sample alpha from a gamma distribution
            alpha_pi = pyro.sample("alpha_pi", dist.Gamma(1.0, 1.0))
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) * alpha_pi / K))

        else: 
            # Use fixed PSI values and initialize pi from a previous model training
            psi = self.fixed_psi.to(self.device)

            # initialize alpha_pi and pi as learnable variables, using provided initial values
            alpha_pi_init = self.alpha_pi_init.to(self.device)
            pi_init = self.pi_init.to(self.device)

            # Register initial values as parameters so pyro.sample can use them
            alpha_pi_param = pyro.param("alpha_pi_param", alpha_pi_init)
            pi_param = pyro.param("pi_param", pi_init)

            alpha_pi = pyro.sample("alpha_pi", dist.Gamma(1.0, 1.0), infer={"initial_value": alpha_pi_param})
            pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) * alpha_pi / K), infer={"initial_value": pi_param})

            # Sample input_conc as usual
            input_conc = self.convertr(input_conc_prior, "bb_conc")
        
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

        # Extract indices of nonzero elements in y and total_counts
        y_indices = y._indices()
        total_counts_indices = total_counts._indices()

        if not torch.equal(y_indices, total_counts_indices):
            raise ValueError("Mismatch between indices of y and total_counts.")

        # Compute predicted probabilities only at these indices
        # pred = torch.matmul(assign, psi).to(self.device)

        # Compute log-probability
        if self.device.type == 'cpu':            
            cell_idx, junc_idx = self.y._indices()
            pred = self.sparse_dot_cpu(assign, psi, cell_idx, junc_idx)
            # print("Requires grad:", pred_values.requires_grad)  # Should now be True
            #pred_vanilla = torch.matmul(assign, psi).to(self.device)
            #pred_vanilla_sparse = pred_vanilla[cell_idx, junc_idx]
            log_prob = self.log_prob_calc_sparse(y, total_counts, pred, input_conc)
            # print(f"The log probability via dense matmul {log_prob}")
            # Validate for numerical issues
            assert torch.isfinite(log_prob).all(), "log_prob contains NaN or infinite values!"
        else:
            if not hasattr(self, "triton_mask"):
                raise RuntimeError("You must call `initialize_triton_mask()` before training.")
                
            # Use Triton for GPU-accelerated matrix multiplication
            cell_idx, junc_idx = y._indices()
            assign = assign.to(self.device, dtype=torch.float32).contiguous()
            psi = psi.to(self.device, dtype=torch.float32).contiguous()

            pred_triton = self.compute_pred_triton(assign, psi)
            # print("Requires grad:", pred.requires_grad)  # Should now be True
            log_prob = self.log_prob_calc_sparse(y, total_counts, pred_triton, input_conc)
            # print(f"The log probability via sparse matmul is {log_prob}")
            # Validate for numerical issues
            assert torch.isfinite(log_prob).all(), "log_prob contains NaN or infinite values!"

        # Add to Pyro trace
        pyro.factor("obs", log_prob)

    def log_prob_calc_sparse(self, y, total_counts, pred_triton, input_conc):
        y_values = y._values()
        total_counts_values = total_counts._values()
        success_probs = pred_triton.clamp(min=1e-6, max=1 - 1e-6)
        
        # Check shape consistency
        assert y_values.shape == total_counts_values.shape == success_probs.shape, (
            f"Shape mismatch: y={y_values.shape}, "
            f"total={total_counts_values.shape}, pred={success_probs.shape}"
        )
    
        if torch.isinf(input_conc).any():
            log_probs = dist.Binomial(total_counts_values, probs=success_probs).log_prob(y_values)
        else:
            alpha = success_probs 
            beta = (1 - success_probs) * input_conc
            log_probs = dist.BetaBinomial(alpha, beta, total_counts_values).log_prob(y_values)

        return log_probs.sum()
    
    def fit(self, guide):
        """
        Fit a probabilistic model using Stochastic Variational Inference (SVI) with gradient clipping
        to ensure numerical stability and early stopping to prevent overfitting.
        """

        # Calculate the learning rate decary factor per step 
        lrd = self.gamma ** (1 / self.num_epochs)
        scheduler = ClippedAdam({'lr': self.lr, 'lrd': lrd}) 
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

        for epoch in range(self.num_epochs):
            
            # Call SVI step with dynamic passing of input_conc_prior
            loss = svi.step(self.y, self.total_counts, self.K, self.junc_specific_prior, self.input_conc_prior)
            losses.append(loss)

            # Assuming a single parameter group:
            current_lr = list(scheduler.optim_objs.values())[0].param_groups[0]['lr']

            # Retrieve log likelihood from the model
            log_prob = self.current_log_prob if hasattr(self, "current_log_prob") else None

            # Log loss, log likelihood, learning rate, and latent variables in wandb
            if self.log_wandb:
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
                print(f"Early stopping at epoch {epoch} with loss: {loss}")
                break

            if epoch % self.print_epochs == 0:
                print(f"Epoch {epoch}: Loss = {loss}")
                print(f"The current learning rate is {current_lr}")

        print(f"Training completed after {epoch + 1} epochs.")
        return losses

    def collect_samples(self, guide):

        samples = {}  # Dictionary to hold samples for each latent variable
        for _ in range(self.num_samples):

            # Generate a trace of the guide execution. Include `input_conc_prior` according to its presence.
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
        Train the model using SVI, collect samples, and return results.
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

        # If num_initializations is 1, then save self.self.best_init to 0
        if num_initializations == 1:
            self.best_init = 0

        # Log model settings
        print(f"Training LeafletFA with {num_initializations} initializations.")
        print(f"Input concentration prior: {self.input_conc_prior}")
        print(f"Junction-specific prior: {self.junc_specific_prior}")
        print(f"Initial K to learn: {self.K}")

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
                raise ValueError(f"PSI and PHI initializations for {K} waypoints not found in adata.varm or adata.obsm.")
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

            # Optionally plot loss
            if self.loss_plot:
                plt.plot(losses)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")

                # Format y-axis in scientific notation
                plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation

                plt.title(f"Loss Plot for Initialization #{i+1}")

                if self.output_dir is not None:
                    plot_filename = f"loss_curve_seed_{seed}.png"
                    plot_filepath = os.path.join(self.output_dir, plot_filename)
                    plt.savefig(plot_filepath)
                    print(f"Loss plot saved to {plot_filepath}", flush=True)
                else:
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
            torch.save(all_results, file_name)
            print(f"Results saved to {file_name}", flush=True)

        # save these things to self. all_results, all_params, variable_sizes        
        self.latent_results = all_results
        self.all_params = all_params
        self.variable_sizes = variable_sizes
            
    def get_best_initialization(self):
        
        """
        Select the best initialization based on the lowest loss.
        """
        
        best_init = np.argmin([result["losses"][-1] for result in self.latent_results])
        best_elbo = self.latent_results[best_init]["losses"][-1]
        print(f"The best initialization was {best_init} with an ELBO of {best_elbo}")

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

        # pruned_pi, pruned_assign_post, psis_mus[pruned_indices], psis_loc[pruned_indices], psi_learned[pruned_indices]
        print(f"The K after pruning is {len(pruned_indices)}")
        print(f"Upating K to {len(pruned_indices)} in the LeafletFA object.")
        self.K = len(pruned_indices)

        