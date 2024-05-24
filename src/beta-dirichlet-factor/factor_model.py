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
from pyro.infer.autoguide import AutoDiagonalNormal
import torch
print (torch.__version__)
print (torch.version.cuda)
#print (torch.cuda.get_device_name())

import matplotlib.pyplot as plt
import random
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
from pyro.infer import Predictive

import datetime
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

import logging
import matplotlib.pyplot as plt
import torch
import pyro
import random
import datetime
import numpy as np

# Define functions for factor model 

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


    # Convert input_conc to a Pyro sample or a fixed value
    # conc = convertr(input_conc, "bb_conc")

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
        # print("Using Binomial distribution")
        log_probs = dist.Binomial(total_counts_values, probs=pred[y_indices[0], y_indices[1]]).log_prob(y_values)
    else:
        # print("Using Beta-Binomial distribution")
        # Extract the success probabilities for the relevant indices
        success_probs = pred[y_indices[0], y_indices[1]]
        # Derive alpha and beta from success_probs and input_conc
        alpha = success_probs * input_conc
        beta = (1 - success_probs) * input_conc
        # Calculate the log probabilities for the beta-binomial distribution
        log_probs = dist.BetaBinomial(alpha, beta, total_counts_values).log_prob(y_values)

    # Sum the log probabilities
    return log_probs.sum()

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
        return pyro.sample(name, dist.Gamma(2.0, 2.0))
    else:
        # Ensure hyperparam is a tensor before checking if it's infinite
        hyperparam_tensor = torch.as_tensor(hyperparam, dtype=torch.float32)
        if torch.isinf(hyperparam_tensor).any():
            return hyperparam_tensor
        else:
            return torch.tensor(hyperparam, dtype=torch.float32)

def model(y, total_counts, K, use_global_prior=True, input_conc_prior = 10):

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

    N, P = y.shape
    
    # Sample input_conc from its prior
    input_conc = convertr(input_conc_prior, "bb_conc")
    
    # Print the input_conc_prior that model got for beta-binomial setting 
    # print("The input_conc_prior is: ", input_conc)
    # print("What went into the model is: ", input_conc_prior)

    # Sample psi from a Beta distribution with concentration parameters a and b (with or without global priors on a and b)
    if use_global_prior:
        a = pyro.sample("a", dist.Gamma(2., 2.).expand([P]).to_event(1)) # every junction has its own a and b
        b = pyro.sample("b", dist.Gamma(2., 2.).expand([P]).to_event(1))
        psi = pyro.sample("psi", dist.Beta(a, b).expand([K, P]).to_event(2))
        psi = psi.to(dtype=torch.float64)
    else:
        psi = pyro.sample("psi", dist.Beta(pyro.sample("a", dist.Gamma(2., 2.)), 
                                           pyro.sample("b", dist.Gamma(2, 2.))).expand([K,P]).to_event(2))
        psi = psi.to(dtype=torch.float64)

    # sample priors for dirichlet distribution
    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(K) / K))
    conc = pyro.sample("dir_conc", dist.Gamma(2, 2)) # value scales the pi vector (higher conc makes the sampled probs more uniform, a lower conc allows more variability, leading to probability vectors that might be skewed towards certain factors).

    with pyro.plate('data', N):
        # sample the factor assignments from the categorical distribution
        assign = pyro.sample("assign", dist.Dirichlet(pi * conc))
        assign = assign.to(dtype=torch.float64)

    # print("assign shape:", assign.shape)  # Expected to be [N, K]
    # print("psi shape:", psi.shape)        # Expected to be [K, P]
    pred = torch.mm(assign, psi)

    # calculate the log probability of observed data under either a binomial or beta-binomial distribution
    log_prob = my_log_prob(y, total_counts, pred, input_conc) 
    pyro.factor("obs", log_prob) 

def fit(y, total_counts, K, use_global_prior, input_conc, guide, patience=5, min_delta=0.01, lr=0.05, num_epochs=500):
    
    """
    Fit a probabilistic model using Stochastic Variational Inference (SVI) with gradient clipping
    to ensure numerical stability and early stopping to prevent overfitting.

    Parameters:
    - y (torch.Tensor): A tensor representing observed junction counts.
    - total_counts (torch.Tensor): A tensor representing the observed intron cluster counts.
    - K (int): The number of components in the mixture model.
    - use_global_prior (bool): Flag to use a global prior in the model.
    - guide (function): A guide function for the SVI (variational distribution).
    - patience (int): The number of epochs to wait without improvement before stopping. Default is 5.
    - min_delta (float): Minimum change in the loss to qualify as an improvement. Default is 0.01.
    - lr (float): Learning rate for the Adam optimizer. Default is 0.05.
    - num_epochs (int): Maximum number of epochs for training. Default is 500.

    Returns:
    - list: A list of loss values for each epoch during the optimization process.
    """
    
    adam = pyro.optim.Adam({"lr": lr})
    ELBO_SAMPLES = 1
    loss = Trace_ELBO(num_particles=ELBO_SAMPLES)
    svi = SVI(model, guide, adam, loss)
    pyro.clear_param_store()
    losses = []
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Sample input_conc from its prior
    # print("The input_conc in fit function input: ", input_conc)     
   
    for epoch in range(num_epochs):
        ## Perform a single step of SVI optimization.
        loss = svi.step(y, total_counts, K, use_global_prior, input_conc)
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

        if epoch % 40 == 0:
            print(f"Epoch {epoch}, Elbo loss: {loss}")

        if epoch == num_epochs - 1:
            logging.info("Elbo loss: {}".format(loss))
    return losses

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_global_prior(use_global_prior):
    if use_global_prior:
        logging.info("Using prior for a and b per junction to model average behaviour!")
    else:
        logging.info("Not using priors on a and b, running simpler non-hierarchical version!")

def print_concentration(input_conc):
    if input_conc == float('inf'):
        logging.info("Using a fixed probability of success across all trials (infinite concentration parameter) with a binomial distribution.")
    elif input_conc is None:
        logging.info("No input concentration parameter provided. Using default Gamma(2.0, 2.0) to initialize and learn bb concentration.")
    else:
        logging.info(f"Using a Beta-binomial distribution with concentration parameter {input_conc}.")

def initialize_seeds(num_initializations, seeds):
    if seeds is None:
        seeds = [random.randint(1, 10000) for _ in range(num_initializations)]
    return seeds

def fit_model(y, total_counts, K, use_global_prior, input_conc, seed, lr, num_epochs):
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    guide = AutoDiagonalNormal(model)
    # Under the hood, this defines a guide that uses a Normal distribution with 
    # learnable parameters corresponding to each sample statement in the model. e.g. in our case, 
    # this distribution should have a size of (5,) correspoding to the 3 regression coefficients for 
    # each of the terms, and 1 component contributed each by the intercept term and sigma in the model.
    losses = fit(y, total_counts, K, use_global_prior, input_conc, guide, patience=10, min_delta=0.01, lr=lr, num_epochs=num_epochs)
    return guide, losses

def plot_losses(losses, i):
    if plt.isinteractive():
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Plot for Initialization {i+1}")
        plt.show()

def save_results(all_results, save_to_file, file_prefix, K, num_initializations):
    if save_to_file:
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"{file_prefix}_{date}_{K}_{num_initializations}_factors.pt" if file_prefix else f"results_{date}_{K}_{num_initializations}_factors.pt"
        torch.save(all_results, file_name)
        logging.info(f"Results saved to {file_name}")


def collect_samples(guide, y, total_counts, K, use_global_prior, num_samples=100):
    """
    Collect samples from the posterior distribution of latent variables using a guide function.

    This function runs a probabilistic guide a specified number of times to collect samples from
    the posterior distribution of latent variables given the data and model configuration. It
    is useful for posterior predictive checks, inference, and uncertainty estimation.

    Parameters:
    - guide (callable): The guide function, usually the approximate posterior in Pyro.
    - y (torch.Tensor): Observed junction counts.
    - total_counts (torch.Tensor): Observed intron cluster counts.
    - K (int): The number of cell states used to train the model.
    - use_global_prior (bool): Flag to indicate whether junction a and b parameters are learned per junction or not.
    - num_samples (int, optional): The number of samples to collect. Defaults to 100.

    Returns:
    - dict: A dictionary where keys are the names of the latent variables and values are numpy arrays
            of collected samples.
    """
    samples = {}  # Dictionary to hold samples for each latent variable

    for _ in range(num_samples):
        
        # Generate a trace of the guide execution
        guide_trace = pyro.poutine.trace(guide).get_trace(y, total_counts, K, use_global_prior)
        
        # Collect samples from the trace
        for name, node in guide_trace.nodes.items():
            if node["type"] == "sample":
                # Initialize the sample list if the variable is encountered for the first time
                if name not in samples:
                    samples[name] = []
                # Append the sample to the list, detached from the PyTorch computation graph
                samples[name].append(node["value"].detach().cpu().numpy())

    # Convert lists of samples to numpy arrays for easier downstream manipulation
    for name in samples:
        samples[name] = np.array(samples[name])
    
    return samples

def calculate_summary_stats(samples):
    stats = {}
    for name, values in samples.items():
        # Convert lists of numpy arrays back to tensors for statistical computation
        values_tensor = torch.tensor(values, dtype=torch.float)
        
        # Ensure the tensor is on CPU before converting to numpy
        if values_tensor.is_cuda:
            values_tensor = values_tensor.cpu()

        stats[name] = {
            'mean': torch.mean(values_tensor, dim=0).numpy(),
            'std': torch.std(values_tensor, dim=0).numpy(),
            '5%': np.percentile(values_tensor.numpy(), 5, axis=0),
            '95%': np.percentile(values_tensor.numpy(), 95, axis=0)
        }
    return stats


def main(y, total_counts, num_initializations=5, seeds=None, file_prefix=None, use_global_prior=True, input_conc=10, save_to_file=True, K=50, loss_plot=True, lr = 0.05, num_epochs=100):

    """
    Main function to fit the Bayesian model.

    Parameters:
    y (torch.Tensor): A tensor representing observed junction counts.
    total_counts (torch.Tensor): A tensor representing the observed intron cluster counts.
    K (int, optional): The number of components in the mixture model. Default is 50.
    num_initializations (int, optional): Number of random initializations. Default is 5.
    seeds (list, optional): List of seeds for random initializations. Default is None, which will generate random seeds.
    """

    # If seeds are not provided, create a list of random seeds
    if seeds is None:
        seeds = [random.randint(1, 10000) for _ in range(num_initializations)]

    all_results = []
    print(("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"))
    if use_global_prior:
        print("Using prior for a and b per junction to model average behaviour!")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    else:
        print("Not using priors on a and b, running simpler non-hierarchical version!")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # if input_conc is infinity, print a message
    if input_conc == float('inf'):
        print("Using a fixed probability of success across all trials (infinite concentration parameter) with a binomial distribution.")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    else:
        print(f"Using a Beta-binomial distribution with concentration parameter {input_conc}.")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if input_conc is None:
            print("No input concentration parameter provided. Using default Gamma(2.0, 2.0) to initialize and learn bb concentration.")
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        else:
            print(f"Using a fixed concentration parameter of {input_conc} for the Beta-binomial distribution.")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++") 
    
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
        
        guide = AutoDiagonalNormal(model) # autoguide list and then add autodiagonalnormal to each part, poutine. block  

        # Assuming `model` is your model function
        trace = pyro.poutine.trace(model).get_trace(y, total_counts, K, use_global_prior)
#        trace.compute_log_prob()  # optional: to compute log probabilities for all sample sites
        print("Model trace:")
        print(trace.format_shapes())

        # Fit the model
        print("Fit the model")
        print(f"The bb conc going into the model is {input_conc}")
        losses = fit(y, total_counts, K, use_global_prior, input_conc, guide, patience=10, min_delta=0.01, lr=lr, num_epochs=num_epochs)
        if loss_plot:
            plt.plot(losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss Plot for Initialization {i+1}")
            plt.show()

        # Disable gradients for the guide parameters
        guide.requires_grad_(False)

        # Sample from the guide (posterior) multiple times
        num_samples = 100

        # Sample from the guide (posterior) multiple times
        print("Sample from the guide (posterior)")
        all_samples = collect_samples(guide, y, total_counts, K, use_global_prior, num_samples)

        quantiles = (guide.quantiles([0.25, 0.5, 0.75]))

        # Calculate summary statistics for each latent variable
        print("Calculate summary statistics")
        summary_stats = calculate_summary_stats(all_samples)

        # Print summary statistics for each latent variable
        for var, stats in summary_stats.items():
            print(f"Summary statistics for {var}: {stats}")

        # Append results
        all_results.append({
            'seed': seed,
            'losses': losses,
            'latent_vars': all_samples,  # store all sampled latent variables
            'summary_stats': summary_stats  # store computed summary statistics
        })

    print("All initializations complete. Returning results.")
    print("------------------------------------------------")

    if save_to_file:
        print("Saving results to file")

        # add date time K to file name
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if file_prefix is None:
            file_name = f"results_{date}_{K}_{num_initializations}_factors.pt"
        else:
            file_name = f"{file_prefix}_{date}_{K}_{num_initializations}_factors.pt"
        torch.save(all_results, file_name)
        print(f"Results saved to {file_name}")
        print("------------------------------------------------")

    return all_results, quantiles

if __name__ == "__main__":
    main()