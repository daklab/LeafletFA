# factor_model.py 
# maintainer: Karin Isaev 
# date: 2024-01-11

# purpose:  
#   - define a probabilistic Bayesian model using a Beta-Dirichlet factorization to infer cell states driven by splicing differences across cells
#   - fit the model using Stochastic Variational Inference (SVI)
#   - sample from the guide (posterior)
#   - extract the latent variables

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import torch
print (torch.__version__)
print (torch.version.cuda)
from torch.optim import Adam
from pyro.optim import MultiStepLR
from pyro.optim import ClippedAdam

import matplotlib.pyplot as plt
import random
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
from pyro.infer import Predictive

import datetime
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

import os

import logging
import matplotlib.pyplot as plt
import torch
import pyro
import random
import datetime
import numpy as np
from pyro import poutine
from pyro.poutine import trace as p_trace
import pyro.distributions as dist
from torch.distributions import constraints
# import the AutoGuideList class from the pyro.infer.autoguide module
from pyro.infer.autoguide import AutoGuideList, AutoDelta, AutoDiagonalNormal
from pyro.infer.autoguide.initialization import init_to_value
from pyro.infer.autoguide.initialization import init_to_uniform

def convertr(hyperparam, name):
    """
    Convert a hyperparameter input into a Pyro sample or a fixed PyTorch tensor.

    Parameters:
    - hyperparam (torch.distributions.Distribution or float or None): The hyperparameter to convert.
      This can be a PyTorch distribution, a float, or None.
    - name (str): The name associated with the Pyro sample. This is used as the key in Pyro's
      internal trace when `hyperparam` is a distribution. For fixed values, this parameter
      is not directly used but is required for consistency with the sampling case.

    Returns:
    - torch.Tensor or pyro.Sample: If `hyperparam` is a distribution, returns a sample from
      this distribution as part of a Pyro model's execution trace. If it is a fixed value or None,
      returns a PyTorch tensor representing this value. The tensor's dtype is set to `torch.float32`,
      ensuring compatibility with most PyTorch and Pyro operations.
    """

    if isinstance(hyperparam, torch.distributions.Distribution):
        return pyro.sample(name, hyperparam)
    elif hyperparam is None:
        return pyro.sample(name, dist.Gamma(10.0, 10.0))
    else:
        # Ensure hyperparam is a tensor before checking if it's infinite
        hyperparam_tensor = torch.as_tensor(hyperparam, dtype=torch.float32)
        if torch.isinf(hyperparam_tensor).any():
            return hyperparam_tensor
        else:
            return torch.tensor(hyperparam, dtype=torch.float32)

def my_log_prob(y_sparse, total_counts_sparse, pred, input_conc):
   
    """
    Compute the log probability of observed data under either a binomial or beta-binomial distribution,
    depending on the concentration parameter.

    This function calculates the log probabilities for non-zero elements in a sparse
    tensor representation of observed data. It dynamically chooses between a binomial 
    distribution and a beta-binomial distribution based on the value of the input concentration parameter.
    The binomial distribution is parameterized by predicted probabilities (`pred`), while the
    beta-binomial distribution parameters (alpha and beta) are derived from the input concentration and the predicted probabilities.

    Parameters:
    y_sparse (torch.Tensor): A sparse tensor representing observed junction count data, where each non-zero
                             element corresponds to a junction count for a specific cell-junction pair.
    total_counts_sparse (torch.Tensor): A sparse tensor representing the total intron cluster counts
                                        (i.e., the number of trials in a binomial distribution) for each junction.
    pred (torch.Tensor): A dense tensor of predicted probabilities for success in each trial (i.e., the 'p' in a binomial distribution).
                         This parameter is used directly in the binomial distribution and influences the beta-binomial
                         distribution through the derivation of alpha and beta parameters based on the concentration parameter.
    input_conc (float or torch.Tensor): A concentration parameter affecting the distribution of success probabilities (junction PSIs),
                                        introducing variability in the success probability across trials for the
                                        beta-binomial distribution. If set to "infinite" (or a very large value),
                                        a binomial distribution is used, implying a fixed probability of success
                                        across all trials. If set to None, a default Gamma(2.0, 1.0) distribution is used and the beta binomial conc gets learned by the model.
                                        If set to a fixed value, the beta-binomial distribution is used with the specified concentration and it will not get learned by the model.

    Returns:
    torch.Tensor: The sum of log probabilities for all observed non-zero data points across all cell-junction pairs.

    Raises:
    ValueError: If the indices of `y_sparse` and `total_counts_sparse` do not match, indicating a misalignment
                between observed junction counts and total intron cluster counts.
    """

    # Extract non-zero elements and their indices from y (junction counts) and total_counts (intron cluster counts)
    y_indices = y_sparse._indices() #y_indices[0] is the row index, y_indices[1] is the column index
    y_values = y_sparse._values()

    total_counts_indices = total_counts_sparse._indices()
    total_counts_values = total_counts_sparse._values()

    # Ensure that y and total_counts align on the same indices
    if not torch.equal(y_indices, total_counts_indices):
        raise ValueError("The indices of y_indices and total_counts_indices do not match.")

    if torch.isinf(input_conc).any():
        # Use binomial distribution
        log_probs = dist.Binomial(total_counts_values, probs=pred[y_indices[0], y_indices[1]]).log_prob(y_values)
    else:
        # Extract the success probabilities for the relevant indices
        success_probs = pred[y_indices[0], y_indices[1]]
        success_probs = torch.clamp(pred[y_indices[0], y_indices[1]], min=1e-6, max=1-1e-6) # Newly added! ensure this makes sense to keep... 

        # Derive alpha and beta from success_probs and input_conc
        alpha = success_probs * input_conc
        beta = (1 - success_probs) * input_conc

        # Calculate the log probabilities for the beta-binomial distribution
        log_probs = dist.BetaBinomial(alpha, beta, total_counts_values).log_prob(y_values)

    # Sum the log probabilities
    return log_probs.sum()

def model(y, total_counts, K, use_global_prior=True, input_conc_prior=None):

    """
    Define a probabilistic Bayesian model using a Beta-Dirichlet factorization.

    This model assumes observed data (junction and intron cluster counts) are generated from a mixture of 
    beta distributions, with latent variables and priors modeling various aspects of the data. The mixture 
    model accounts for the possibility that observed data might be influenced by several different factors 
    or cell states, each with its own characteristic distribution of values.

    The likelihood of observations is conditioned on latent variables and can be modeled using either a 
    Binomial distribution (for fixed success probabilities) or a Beta-Binomial distribution (to account for 
    variability in success probabilities across trials), depending on the input concentration parameter.

    Parameters:
    - y (torch.Tensor): Observed data (junction counts).
    - total_counts (torch.Tensor): Total counts for each observation (total intron cluster counts).
    - K (int): Number of factors representing cell states.
    - use_global_prior (bool, optional): Whether to use a global prior for psi. Defaults to True.
    - input_conc_prior (float, torch.Tensor, torch.distributions.Distribution, optional): Prior or fixed value for the concentration 
      parameter of the Beta-Binomial distribution. If not provided, defaults to a Gamma(2.0, 1.0) distribution.

    Returns:
    None: This function contributes to the Pyro model's trace and does not return any value.
    """

    # print("++++++ MODEL CALLED ++++++")
    N, P = y.shape

    # Sample input_conc from its prior
    input_conc = convertr(input_conc_prior, "bb_conc")
    eps = 1e-4

    if use_global_prior:
        # Only sample 'a' and 'b' for use in the guide if not passed externally
        a = pyro.sample("a", dist.Gamma(2., 2.).expand([P]).to_event(1))
        b = pyro.sample("b", dist.Gamma(2., 2.).expand([P]).to_event(1))
        psi = pyro.sample("psi", dist.Beta(a+eps, b+eps).expand([K, P]).to_event(2)) 
        psi = psi.to(dtype=torch.float32)

    else:
        a = pyro.sample("a", dist.Gamma(2., 2.)) # single value for all junctions
        b = pyro.sample("b", dist.Gamma(2., 2.))
        psi = pyro.sample("psi", dist.Beta(a+eps, b+eps).expand([K, P]).to_event(2))
        psi = psi.to(dtype=torch.float32)

    # Assert that 'psi' has no NaN or negative values
    assert torch.isfinite(psi).all(), "psi contains NaN or infinite values!"
    # assert torch.all(psi >= 0), "psi contains negative values!"

    # Sample priors for Dirichlet distribution
    alpha = 5.0  # A reasonable value to avoid extreme imbalance
    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) * alpha / K))

    # Assert that pi sums to 1 and contains no NaN/inf values
    assert torch.isfinite(pi).all(), "pi contains NaN or infinite values!"
    assert torch.allclose(pi.sum(), torch.tensor(1.0, dtype=torch.float)), f"pi does not sum to 1: {pi.sum()}"

    # conc value scales the pi vector (higher conc makes the sampled probs more uniform, 
    #a lower conc allows more variability, leading to probability vectors that might be skewed towards certain factors).
    conc = pyro.sample("dir_conc", dist.Gamma(2, 2))
    conc = torch.clamp(conc, min=1e-6, max=1e6)

    assign = pyro.sample("assign", dist.Dirichlet(pi * conc).expand([N]).to_event(1))
    assign = assign.to(dtype=torch.float32)
    
    # Ensure no negative values in assign
    if torch.any(assign < 1e-8):
        assign = torch.clamp(assign, min=1e-8)
        assign = assign / assign.sum(dim=1, keepdim=True)  # Re-normalize to sum to 1

    # Assert that assign has no NaN, Inf, or negative values
    # assert torch.isfinite(assign).all(), "assign contains NaN or infinite values!"
    # assert torch.all(assign >= 0), "Assign has negative values!"
    # assert torch.allclose(assign.sum(dim=1), torch.tensor(1.0, dtype=torch.float32)), f"assign rows do not sum to 1: {assign.sum(dim=1)}"

    pred = torch.mm(assign, psi)

    # Assert pred contains no NaN/Inf values
    # assert torch.isfinite(pred).all(), "pred contains NaN or infinite values!"

    # Move pred to CUDA if available
    if torch.cuda.is_available():
        pred = pred.cuda()

    # calculate the log probability of observed data under either a binomial or beta-binomial distribution
    log_prob = my_log_prob(y, total_counts, pred, input_conc) 
    assert torch.isfinite(log_prob).all(), "log_prob contains NaN or infinite values!"
    pyro.factor("obs", log_prob) 


def fit(y, total_counts, K, use_global_prior, input_conc_prior, guide, patience=5, min_delta=0.01, lr=0.01, num_epochs=500):
    
    """
    Fit a probabilistic model using Stochastic Variational Inference (SVI) with gradient clipping
    to ensure numerical stability and early stopping to prevent overfitting.
    """

    # scheduler = MultiStepLR({'optimizer': Adam,
    #                         'optim_args': {'lr': lr},
    #                         'milestones': [20, 50, 80, 100, 120, 150, 200, 250],
    #                         'gamma': 0.1})

    # Initialize learning rate and decay factor
    initial_lr = lr
    num_steps = num_epochs
    gamma = 0.05 

    # Calculate the learning rate decay factor per step
    lrd = gamma ** (1 / num_steps)

    # Use ClippedAdam optimizer with continuous learning rate decay
    scheduler = ClippedAdam({'lr': initial_lr, 'lrd': lrd})

    # Define the number of samples used to estimate the Evidence Lower Bound (ELBO).
    # More samples can provide a more accurate estimate but will increase computation time.
    ELBO_SAMPLES = 1
    
    # Initialize the loss function as Trace_ELBO, which estimates the ELBO.
    loss = Trace_ELBO(num_particles=ELBO_SAMPLES) 

    # SVI the step function minimizes the negative ELBO
    svi = SVI(model, guide, scheduler, loss)

    # Clear Pyro's parameter store before starting the optimization. This ensures that previous
    # runs do not interfere with the current optimization, providing a clean slate - do we ened this?
    pyro.clear_param_store()

    losses = []
    best_loss = float('inf')
    epochs_since_improvement = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Move tensors to the appropriate device
    # y, total_counts = y.to(device), total_counts.to(device)
    
    # Convert input_conc_prior to tensor if it's a float and move it to the device
    if isinstance(input_conc_prior, float):
        input_conc_prior = torch.tensor(input_conc_prior, device=device)
    elif isinstance(input_conc_prior, torch.Tensor):
        input_conc_prior = input_conc_prior.to(device)

    print(f"Training in progress for {num_epochs} epochs!")

    for epoch in range(num_epochs):

        # Call SVI step passing the input_conc_prior dynamically
        loss = svi.step(y, total_counts, K, use_global_prior, input_conc_prior)

        losses.append(loss)

        # Check for improvement based on min_delta and update best loss and epochs_since_improvement.
        if best_loss - loss > min_delta:
            best_loss = loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Early stopping if no improvement for a number of epochs specified by patience.
        if epochs_since_improvement >= patience:
            print(f"Stopping early at epoch {epoch}. Best Elbo Loss: {best_loss}")
            logging.info("Elbo loss: {}".format(loss)) 
            break

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Elbo loss: {loss}")

        if epoch == num_epochs - 1:
            logging.info("Elbo loss: {}".format(loss))

    return losses

def print_global_prior(use_global_prior):
    if use_global_prior:
        logging.info("Using prior for a and b per junction to model average behaviour!")
    else:
        logging.info("Not using priors on a and b, running simpler non-hierarchical version!")

def print_inits(psi_init=None, phi_init=None):
    if psi_init is None and phi_init is None:
        logging.info("No initialization matrices provided for PSI and PHI, doing random initialization for variational parameters!")
    else:
        logging.info("Using waypoint-based PSI and PHI matrices for initializing these variational parameters!")

def print_concentration(input_conc_prior):
    """
    Log the type of concentration parameter used in the model based on the prior provided.

    Parameters:
    - input_conc_prior: The prior or fixed value for the concentration parameter of the Beta-Binomial distribution.
                        If None, the parameter will be learned and initialized with a default distribution.
    """
    if input_conc_prior == float('inf'):
        logging.info("Using a fixed probability of success across all trials (infinite concentration parameter) with a binomial distribution.")
    elif input_conc_prior is None:
        logging.info("No input concentration parameter provided. Using default Gamma(2.0, 2.0) to initialize and learn bb concentration.")
    else:
        logging.info(f"Using a Beta-binomial distribution with initial concentration parameter set to {input_conc_prior}.")

def collect_samples(guide, y, total_counts, K, use_global_prior, input_conc_prior, num_samples=10):
    
    samples = {}  # Dictionary to hold samples for each latent variable
    for _ in range(num_samples):
        
        # Generate a trace of the guide execution. Include `input_conc_prior` according to its presence.
        if input_conc_prior is None:
            guide_trace = pyro.poutine.trace(guide).get_trace(y, total_counts, K, use_global_prior, input_conc_prior=None)
        else:
            guide_trace = pyro.poutine.trace(guide).get_trace(y, total_counts, K, use_global_prior, input_conc_prior)
        
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

def calculate_summary_stats(samples):
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

def extract_variable_sizes(model, *args, **kwargs):
    trace = poutine.trace(model).get_trace(*args, **kwargs)
    sizes = {}
    for name, node in trace.nodes.items():
        if node["type"] == "sample" and not node["is_observed"]:
            sizes[name] = node["value"].shape
    return sizes

def main(y, total_counts, num_initializations=5, seeds=None, psi_init=None, phi_init=None, file_prefix=None, use_global_prior=True, input_conc_prior=None, save_to_file=True, K=50, loss_plot=True, lr=0.05, num_epochs=100, output_dir=None):

    """
    Main function to fit our Leaflet Bayesian beta-dirichlet factor model.

    Parameters:
    y (torch.Tensor): A tensor representing observed junction counts.
    total_counts (torch.Tensor): A tensor representing the observed intron cluster counts.
    K (int, optional): The number of components in the mixture model. Default is 50.
    num_initializations (int, optional): Number of random initializations. Default is 5.
    seeds (list, optional): List of seeds for random initializations. Default is None, which will generate random seeds.
    input_conc_prior: The prior or fixed value for the concentration parameter. If None, it will be learned.

    - psi_init (torch.Tensor, optional): Pre-initialized values for psi.
    - phi_init (torch.Tensor, optional): Pre-initialized values for assign (phi).
    """

    # If seeds are not provided, create a list of random seeds and print them
    if seeds is None:
        seeds = [random.randint(1, 10000) for _ in range(num_initializations)]
        print (f"Random seeds: {seeds}")

    all_results = []

    print_concentration(input_conc_prior)  # Log the concentration setting
    print_global_prior(use_global_prior)  # Log the global prior setting
    print_inits(psi_init, phi_init) # Log whether initializaiton is based on waypoints or not  

    # Run the model for each seed
    for i, seed in enumerate(seeds):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Initialization {i+1} with seed {seed}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        # Set the seed
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Define the guide
        print("Define the guide using AutoDiagonalNormal based on the model structure.")

        # Diagonal Gaussian: This guide assumes that the posterior distribution can be 
        # approximated by a multivariate normal distribution with a diagonal covariance matrix. 
        # This means that it assumes all latent variables are uncorrelated and normally distributed.
        # Variational Parameters: It automatically introduces variational parameters (means and variances) 
        # for each latent variable in the model. These parameters are optimized during inference to 
        # make the guide's distribution as close as possible to the true posterior.

        guide = AutoGuideList(model)

        # Guide for 'psi' with conditional initialization
        if psi_init is not None:
            guide.append(AutoDiagonalNormal(
                poutine.block(model, expose=['psi']),
                init_loc_fn=init_to_value(values={'psi': psi_init})
            ))
        else:
            guide.append(AutoDiagonalNormal(
                poutine.block(model, expose=['psi'])
            ))

        # Guide for 'assign' with conditional initialization
        if phi_init is not None:
            guide.append(AutoDiagonalNormal(
                poutine.block(model, expose=['assign']),
                init_loc_fn=init_to_value(values={'assign': phi_init})
            ))
        else:
            guide.append(AutoDiagonalNormal(
                poutine.block(model, expose=['assign'])
            ))
            
        # Guide for everything else (excluding 'psi' and 'assign')
        guide.append(AutoDiagonalNormal(poutine.block(model, hide=['psi', 'assign']))) 

        # Fit the model
        print("Fit the model")

        # Fit the model
        losses = fit(y, total_counts, K, use_global_prior, input_conc_prior, guide, patience=10, min_delta=0.01, lr=lr, num_epochs=num_epochs)
        
        if loss_plot:
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss Plot for Initialization {i+1}")
            
            # Automatically generate a filename with a random number for uniqueness
            random_num = random.randint(1, 100000)
            plot_filename = f"random_init_{random_num}_loss_curve.png"
            
            # Join the output directory with the filename
            if output_dir is not None:
                plot_filepath = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_filepath)
                print(f"Loss plot automatically saved to {plot_filepath}")
            else:
                raise ValueError("output_dir is required to save the loss plot.")

        # Sample from the guide (posterior) multiple times
        print("Sample from the guide (posterior)")
        all_samples = collect_samples(guide, y, total_counts, K, use_global_prior, input_conc_prior)

        # Calculate summary statistics for each latent variable
        print("Calculate summary statistics")
        summary_stats = calculate_summary_stats(all_samples)

        # Append results
        all_results.append({
            'seed': seed,
            'losses': losses,
            'latent_vars': all_samples,  # store all sampled latent variables
            'summary_stats': summary_stats  # store computed summary statistics
        })

    print("All initializations complete. Returning results.")
    print("------------------------------------------------")

    # Get model variable sizes 
    variable_sizes = extract_variable_sizes(model, y, total_counts, K, use_global_prior, input_conc_prior)

    print("------------------------------------------------")
    print("Model variable sizes:", variable_sizes)
    print("------------------------------------------------")

    if save_to_file:
        print("Saving results to file")
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"{file_prefix}_{date}_{K}_{num_initializations}_factors.pt" if file_prefix else f"results_{date}_{K}_{num_initializations}_factors.pt"
        torch.save(all_results, file_name)
        print(f"Results saved to {file_name}")
        print("------------------------------------------------")

    return all_results, variable_sizes


if __name__ == "__main__":
    main()