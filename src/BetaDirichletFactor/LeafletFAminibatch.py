import os
import random
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import ticker for scientific notation

import torch
import pyro
import pyro.distributions as dist
from torch.distributions import Distribution
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import (
    AutoGuideList, AutoDelta, AutoDiagonalNormal
)
from pyro.infer.autoguide.initialization import init_to_value, init_to_uniform
from pyro import poutine

from pyro.poutine import block
from pyro.infer.autoguide.initialization import init_to_value
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Disable default argument validation for distributions
Distribution.set_default_validate_args(False)

# Print Torch and CUDA version info
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")

def sparse_collate_fn(batch, J):
    """
    Custom collate function that aligns y and total_counts 
    using nonzero indices from total_counts.
    
    Parameters:
    - batch: List of (cell_idx, y, total_counts) tuples.
    - J: Total number of junctions (global max_junctions).
    
    Returns:
    - A dictionary containing batched dense tensors for y, total_counts, and a mask.
    """

    cell_indices_batch = []
    y_tensors = []
    total_counts_tensors = []
    mask_tensors = []  # Mask for valid values

    for (cell_idx, y, total_counts) in batch:
        cell_indices_batch.append(cell_idx)

        # Ensure tensors are coalesced (important for proper indexing)
        y = y.coalesce()
        total_counts = total_counts.coalesce()

        # Initialize dense tensors
        y_dense = torch.zeros(J, dtype=torch.float32)
        total_counts_dense = torch.zeros(J, dtype=torch.float32)
        mask = torch.zeros(J, dtype=torch.bool)  # Mask for valid entries

        # Retrieve nonzero indices
        y_indices = y.indices()[0]  # Extracts row indices (assuming y is (1D) sparse)
        total_counts_indices = total_counts.indices()[0]

        # Place values in the correct positions
        y_dense[y_indices] = y.values()
        total_counts_dense[total_counts_indices] = total_counts.values()
        mask[y_indices] = True  # Mark positions where values exist

        y_tensors.append(y_dense)
        total_counts_tensors.append(total_counts_dense)
        mask_tensors.append(mask)

    return {
        "cell_indices": torch.tensor(cell_indices_batch, dtype=torch.long),
        "y_values": torch.stack(y_tensors),  # (batch_size, J)
        "total_counts_values": torch.stack(total_counts_tensors),  # (batch_size, J)
        "mask": torch.stack(mask_tensors)  # (batch_size, J)
    }

class SparseTensorDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_y, sparse_total_counts):
        self.y = sparse_y
        self.total_counts = sparse_total_counts

    def __len__(self):
        return self.y.size(0)  # number of cells

    def __getitem__(self, idx):
        # row index = cell index
        y_row = self.y[idx]
        total_counts_row = self.total_counts[idx]
        # Return the index alongside the data
        return idx, y_row, total_counts_row
    
class LeafletFA:
    def __init__(self, adata, K=50, junc_specific_prior=True, input_conc_prior=None, 
                 waypoints_use=True, fixed_psi=None, batch_size=128, 
                 eps = 1e-6, alpha_hyper = 5.0, lr=0.01, gamma = 0.05, num_epochs=500, print_epochs=10, 
                 ELBO_num_particles=1, patience=5, min_delta=0.01, num_samples=10, loss_plot=True, output_dir=None):
        """
        Initialize the LeafletFA model.
        
        Parameters:
        - adata (AnnData): AnnData object containing single-cell splicing data.
        - K (int): Number of factors.
        - junc_specific_prior (bool): Whether to learn a global prior over junctions.
        - input_conc_prior: Concentration prior (can be either 'None' or 'torch.tensor(np.inf)').
        - waypoints_use (bool): Whether to use waypoints for initialization.
        - fixed_psi (torch.Tensor): Fix PSI during inference by using an existing PSI latent variable from previous model training.
        - batch_size (int): Batch size for training.
        - eps (float): Small value to prevent numerical issues.
        - alpha_hyper (float): Hyperparameter for the Dirichlet prior on Pi.
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
        self.waypoints_use = waypoints_use
        self.fixed_psi = fixed_psi
        self.batch_size = batch_size
        self.eps = eps
        self.alpha_hyper = alpha_hyper
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
        
        # Initialize other attributes
        self.guide = None
        self.losses = []
        self.results = None
        self.variable_sizes = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If fixed PSI is provided, disable waypoint initialization
        if self.fixed_psi is not None:
            if self.waypoints_use:
                print("Fixed PSI provided! Disabling waypoint-based PSI initialization.")
            self.waypoints_use = False

    @staticmethod
    def convertr(hyperparam, name):
        """ Convert hyperparameters to Pyro samples or fixed tensors. """
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
        del cell_index_tensor, junc_index_tensor, ycount, total_counts_tensor
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Add the tensors to the self object
        self.y = full_y_tensor
        self.total_counts = full_total_counts_tensor

    # set up data loader for minibatch training
    def setup_data_loader(self, batch_size):
        """
        Set up a data loader for minibatch training.
        """

        # Create dataset
        dataset = SparseTensorDataset(self.y, self.total_counts)
        J = dataset[0][1].size(0)  # Number of junctions

        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: sparse_collate_fn(batch, J),
            pin_memory=False,  # Turn off since data is already on GPU
            num_workers=0      # Use main process to avoid extra transfers
        )
       
        return dataloader

    def model(self, y_batch, total_counts_batch, mask_batch):
        """
        Probabilistic model for LeafletFA with mini-batching.

        Parameters:
        - y_batch (Tensor): Observed junction counts for the batch.
        - total_counts_batch (Tensor): Total counts for intron clusters in the batch.
        - mask_batch (Tensor): Mask indicating valid junctions in the batch.
        """

        # Number of junctions
        P = y_batch.shape[1]

        # Number of cells in the batch
        actual_batch_size = y_batch.shape[0]

        # Sample concentration prior
        input_conc = self.convertr(self.input_conc_prior, "bb_conc")

        # Sample all the PSI related parameters if fixed_psi is None
        # Learnable priors for Gamma shape and rate
        a_shape = pyro.sample("a_shape", dist.Gamma(1.0, 1.0))  # Learn shape
        a_rate = pyro.sample("a_rate", dist.Gamma(1.0, 1.0))  # Learn rate
        b_shape = pyro.sample("b_shape", dist.Gamma(1.0, 1.0))
        b_rate = pyro.sample("b_rate", dist.Gamma(1.0, 1.0))
       
        # Sample psi from a Beta distribution
        if self.junc_specific_prior:
            # junction-specific shape parameters sampled from Gamma priors 
            a = pyro.sample("a", dist.Gamma(a_shape, a_rate).expand([P]).to_event(1))
            b = pyro.sample("b", dist.Gamma(b_shape, b_rate).expand([P]).to_event(1))
            psi = pyro.sample("psi", dist.Beta(a+self.eps, b+self.eps).expand([self.K, P]).to_event(2)) 
            psi = psi.to(dtype=torch.float32)
        else:
            # single shape parameters for all junctions sampled from Gamma priors
            a = pyro.sample("a", dist.Gamma(a_shape, a_rate))  
            b = pyro.sample("b", dist.Gamma(b_shape, b_rate))  
            psi = pyro.sample("psi", dist.Beta(a+self.eps, b+self.eps).expand([self.K, P]).to_event(2))
            psi = psi.to(dtype=torch.float32)
            # Assert that 'psi' has no NaN or negative values
            if not torch.isfinite(psi).all():
                raise ValueError("psi contains NaN or infinite values!")

        # Sample concentration value that scales the pi vector (higher values lead to more uniform pi)
        conc = pyro.sample("dir_conc", dist.Gamma(2., 2.))
        conc = torch.clamp(conc, min=1e-6, max=1e6) # Prevent numerical issues

        # Sample Pi prior from a Dirichlet distribution
        pi = pyro.sample("pi", dist.Dirichlet(torch.ones(self.K) * self.alpha_hyper / self.K))
        
        # Assert that pi sums to 1 and contains no NaN/inf values
        assert torch.isfinite(pi).all(), "pi contains NaN or infinite values!"
        assert torch.allclose(pi.sum(), torch.tensor(1.0, dtype=torch.float)), f"pi does not sum to 1: {pi.sum()}"

        # Define `assign` inside the batch plate
        with pyro.plate("batch", actual_batch_size, dim=-2):
        
            assign_raw = pyro.sample("assign", dist.Dirichlet(pi * conc * torch.ones(self.K, device=self.device)))
        
            # Remove the extra dimension
            assign = assign_raw.squeeze(1)  # (batch_size, K)
        
            # Normalize the assignment probabilities
            assign = torch.clamp(assign, min=self.eps)
            assign = assign / assign.sum(dim=1, keepdim=True)

            # Compute predicted success probabilities
            pred = torch.matmul(assign, psi)

        # Compute log probability
        log_prob = self.log_prob_calc(y_batch, total_counts_batch, pred, input_conc, mask_batch)
        scaled_log_prob = log_prob * (self.num_cells / actual_batch_size)

        pyro.factor("obs", scaled_log_prob)

    def log_prob_calc(self, y, total_counts, pred, input_conc, mask):
        """
        Compute the log probability of observed data under either a binomial or beta-binomial distribution,
        depending on whether fixed_psi is used.

        Parameters:
        - y (torch.Tensor): Dense tensor of observed junction counts (batch_size, max_junctions).
        - total_counts (torch.Tensor): Dense tensor of total intron cluster counts (batch_size, max_junctions).
        - pred (torch.Tensor): Dense tensor of predicted success probabilities (batch_size, max_junctions).
        - input_conc (torch.Tensor or float): The concentration parameter.
        - mask (torch.Tensor): Boolean tensor indicating valid (non-padded) values.

        Returns:
        - torch.Tensor: The summed log probabilities for all non-zero data points.
        """

        batch_size = y.shape[0]

        # Ensure mask has the correct batch size
        # Remove this should not be needed 
        # if mask.shape[0] != batch_size:
        #    print(f"Fixing mask batch size from {mask.shape[0]} to {batch_size}")
        #    mask = mask[:batch_size]

        # Ensure success_probs are in the valid range to prevent log issues
        success_probs = pred.clamp(min=1e-6, max=1-1e-6)  # Avoid log issues

        # Apply mask: Only keep values where mask == True (where junctions are present in given batch of cells)
        y_masked = torch.masked_select(y, mask)
        total_counts_masked = torch.masked_select(total_counts, mask)
        success_probs_masked = torch.masked_select(success_probs, mask)

        if self.fixed_psi is not None:
            # Binomial likelihood
            log_probs = dist.Binomial(total_counts_masked, probs=success_probs_masked).log_prob(y_masked)
        else:
            if torch.isinf(input_conc).any():
                # Binomial likelihood
                log_probs = dist.Binomial(total_counts_masked, probs=success_probs_masked).log_prob(y_masked)
            else:
                # Beta-Binomial likelihood
                alpha = success_probs_masked * input_conc
                beta = (1 - success_probs_masked) * input_conc
                # import pdb ; pdb.set_trace()
                log_probs = dist.BetaBinomial(alpha, beta, total_counts_masked).log_prob(y_masked)

        return log_probs.sum()
    
    def train(self, svi, train_loader):
        """
        Train the model using mini-batching.

        Parameters:
        - svi (SVI): Stochastic Variational Inference optimizer.
        - train_loader (DataLoader): PyTorch DataLoader providing mini-batches.

        Returns:
        - total_epoch_loss_train (float): Average ELBO loss per data point in the epoch.
        """
        epoch_loss = 0.0  # Accumulate ELBO loss across mini-batches

        # if self.device.type == "cuda" then use_cuda = True else False
        use_cuda = self.device.type == "cuda"

        # Convert input_conc_prior to tensor if it's a float and move it to the device
        if isinstance(self.input_conc_prior, float):
            self.input_conc_prior = torch.tensor(self.input_conc_prior, device=self.device)
        elif isinstance(self.input_conc_prior, torch.Tensor):
            self.input_conc_prior = self.input_conc_prior.to(self.device)
        
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            # Extract batch components from DataLoader
            cell_indices = batch["cell_indices"]
            y_batch = batch["y_values"]
            total_counts_batch = batch["total_counts_values"]
            mask_batch = batch["mask"]

            # Compute ELBO loss using the batch
            batch_loss = svi.step(y_batch, total_counts_batch, mask_batch)
            epoch_loss += batch_loss

        # Sum loss across all batches
        sum_epoch_loss = epoch_loss
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Sum Loss: {sum_epoch_loss:.2f}, Avg Loss: {avg_epoch_loss:.4f}")
        return sum_epoch_loss

    def fit(self, guide=None):
        """
        Fit the model using Stochastic Variational Inference (SVI) with mini-batching.
        Implements gradient clipping, learning rate decay, and early stopping.
        """

        # Define the guide
        if guide is None:
            guide = AutoGuideList(self.model)
        
            # Add a guide for continuous global parameters
            guide.add(AutoDiagonalNormal(poutine.block(
                self.model, 
                expose=["a_shape", "a_rate", "b_shape", "b_rate", "dir_conc", "pi", "bb_conc"]
             )))
        
            # Add a guide for the continuous array parameters
            guide.add(AutoDiagonalNormal(poutine.block(
                self.model, 
                expose=["a", "b", "psi"]
            )))

            # Add a guide for the per-batch assignment parameters
            guide.add(AutoDiagonalNormal(poutine.block(
                self.model, 
                expose=["assign"]
            )))
        
        self.guide = guide
        pyro.clear_param_store()

        # Initialize SVI
        optimizer = ClippedAdam({'lr': self.lr, 
                                 'lrd': self.gamma ** (1 / self.num_epochs)})  
        loss_fn = Trace_ELBO(num_particles=self.ELBO_num_particles)  
        svi = SVI(self.model, guide, optimizer, loss_fn)

        # Prepare DataLoader
        dataloader = self.setup_data_loader(batch_size=self.batch_size)
        print(f"Training in progress for {self.num_epochs} epochs!")

        train_elbo = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_epochs):
            total_epoch_loss = self.train(svi, dataloader)
            # total_epoch_loss is the SUM of the negative log-likelihood (or negative ELBO).
            # We *minimize* that quantity, so "better" => smaller loss.
            # if epoch divides by 2 then print it 
            if epoch % self.print_epochs == 0:
                print(f"Epoch {epoch}: Total Loss: {total_epoch_loss:.2f}")

            if total_epoch_loss < (best_loss - self.min_delta):
                best_loss = total_epoch_loss
                patience_counter = 0

                # Save the best parameter state so we can restore it after stopping
                best_params = {
                    name: p.clone().detach()
                    for name, p in pyro.get_param_store().items()
                    }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                break

            # Keep track of negative total_epoch_loss for plotting an ELBO-like curve
            train_elbo.append(-total_epoch_loss)

        return train_elbo
    
    def get_cell_assignments(self):
        """
        Extract assignment values (cell factor activities) for all cells after training.

        Returns:
        - assignments (np.ndarray): Array of shape [num_cells, K] with cell-to-factor assignments
        """

        # Make sure model is fitted
        if self.guide is None:
            raise ValueError("Model must be fit before extracting assignments")

        batch_size = self.batch_size
        dataloader = self.setup_data_loader(batch_size=batch_size)

        # Initialize array to hold results
        all_assignments = np.zeros((self.num_cells, self.K))

        # Set model to evaluation mode
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                print(f"Processing batch {batch_idx+1}/{len(dataloader)} for assignments")

                # Extract batch data
                cell_indices = batch["cell_indices"].cpu().numpy()
                y_batch = batch["y_values"]
                total_counts_batch = batch["total_counts_values"]
                mask_batch = batch["mask"]

                # Move to GPU if needed
                if self.device.type == "cuda":
                    y_batch = y_batch.cuda()
                    total_counts_batch = total_counts_batch.cuda()
                    mask_batch = mask_batch.cuda()

                # Run the guide to get posterior samples
                guide_trace = poutine.trace(self.guide).get_trace(
                    y_batch, total_counts_batch, mask_batch
                )

                # Extract assignment values from guide trace
                if "assign" in guide_trace.nodes:
                    # Get the raw assignment tensor
                    assign_values = guide_trace.nodes["assign"]["value"]

                    # Remove any extra dimensions (should be [batch_size, K] after squeezing)
                    if len(assign_values.shape) > 2:
                        assign_values = assign_values.squeeze(1)

                    # Convert to probabilities if needed
                    # For Dirichlet - values are already probabilities 
                    # If using Normal guide, may need to do assign_probs = torch.softmax(assign_values, dim=-1)?
                    assign_probs = assign_values

                    # Convert to numpy
                    batch_assignments = assign_probs.cpu().numpy()

                    # Store in result array
                    for i, cell_idx in enumerate(cell_indices):
                        if i < len(batch_assignments):
                            all_assignments[cell_idx] = batch_assignments[i]

        return all_assignments



