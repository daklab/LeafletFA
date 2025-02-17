import numpy as np
import torch 
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

# Function for the logit transform
def logit(p):
    return torch.log(p / (1 - p))

# Function for the inverse logit transform (sigmoid)
def inverse_logit(z):
    return 1 / (1 + torch.exp(-z))

# Function to sample z from gaussian distribution using mu_kj and sigma_kj
def sample_z(mu_kj, sigma_kj, num_sample=10):
    return np.random.normal(mu_kj, sigma_kj, num_sample)

def plot_junc_dists(scores, index_junc, pis):
    """
    Plot Gaussian distributions for splice junctions.

    Parameters:
    - scores: DataFrame with columns `mu_k` and `loc_k` (for each k)
    - index_junc: Index of the junction to plot
    - pis: List or tensor of mixing proportions
    """
    K = len(pis)  # Number of factors

    # Extract dynamically based on K
    mu_cols = [f"mu_{k}" for k in range(K)]
    var_cols = [f"loc_{k}" for k in range(K)]  # Assuming loc_k represents std deviation, not variance

    # Extract means and variances, ensuring conversion to float
    mus = torch.tensor(scores.loc[index_junc, mu_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    std_devs = torch.tensor(scores.loc[index_junc, var_cols].to_numpy(dtype=np.float32), dtype=torch.float32)

    # Convert std deviations to variances (assuming loc_k is standard deviation)
    variances = std_devs ** 2  

    # Compute combined mean and variance
    combined_mean, combined_variance = compute_combined_parameters(mus, variances, pis)
    
    print(f"The combined mean is: {combined_mean}")
    print(f"The combined variance is: {combined_variance}")

    # Append combined values for plotting
    mus_list = mus.tolist() + [combined_mean.item()]
    variances_list = variances.tolist() + [combined_variance.item()]
    plot_gaussian(mus_list, variances_list)
    
def compute_combined_parameters(mu, variance, pi):
    device = mu.device if hasattr(mu, 'device') else 'cpu'
    mu = torch.tensor(mu, device=device) if isinstance(mu, np.ndarray) else mu.to(device)
    variance = torch.tensor(variance, device=device) if isinstance(variance, np.ndarray) else variance.to(device)
    pi = torch.tensor(pi, device=device) if isinstance(pi, np.ndarray) else pi.to(device)

    variance = torch.clamp(variance.float(), min=1e-5)
    mu, pi = mu.float(), pi.float()
    
    precisions = 1.0 / variance
    weighted_precisions = pi * precisions
    combined_precision = torch.sum(weighted_precisions)
    combined_variance = 1.0 / combined_precision
    combined_mean = combined_variance * torch.sum((pi * mu) / variance)
    
    return combined_mean, combined_variance

def gaussian_log_prob(x, mu, variance):
    device = mu.device if hasattr(mu, 'device') else 'cpu'
    x = torch.tensor(x, device=device) if isinstance(x, (float, int, np.ndarray)) else x.to(device)
    variance = variance + 1e-10
    return -0.5 * (np.log(2 * np.pi) + torch.log(variance) + ((x - mu) ** 2) / variance)

def compute_log_Pj_H0(mu_kj, variance_kj, pi_k):
    """
    Compute log P_j(H0) with proper weighting of individual components
    """

    # Convert inputs to tensors and ensure float type
    device = mu_kj.device if hasattr(mu_kj, 'device') else 'cpu'
    
    mu_kj = torch.tensor(mu_kj, dtype=torch.float32, device=device) if isinstance(mu_kj, np.ndarray) else mu_kj.float().to(device)
    variance_kj = torch.tensor(variance_kj, dtype=torch.float32, device=device) if isinstance(variance_kj, np.ndarray) else variance_kj.float().to(device)
    pi_k = torch.tensor(pi_k, dtype=torch.float32, device=device) if isinstance(pi_k, np.ndarray) else pi_k.float().to(device)
    
    # Get combined parameters
    combined_mean, combined_variance = compute_combined_parameters(mu_kj, variance_kj, pi_k)
    
    # Calculate weighted log probabilities at zero for each component (DOES THIS MAKE SENSE?! mathematically?)
    log_pdfs_zero = gaussian_log_prob(torch.tensor(0.0), mu_kj, variance_kj)
    weighted_sum_log_pdfs_zero = torch.sum(pi_k * log_pdfs_zero)
    
    # Calculate log probability at zero for combined distribution
    combined_log_pdf_zero = gaussian_log_prob(torch.tensor(0.0), combined_mean, combined_variance)
    
    # Return weighted difference
    return weighted_sum_log_pdfs_zero - combined_log_pdf_zero

def compute_albf(mu_kj, variance_kj, pi_k):
    device = mu_kj.device if hasattr(mu_kj, 'device') else 'cpu'
   
    # Convert to tensors and move to device
    mu_kj = torch.from_numpy(mu_kj).to(device) if isinstance(mu_kj, np.ndarray) else mu_kj.to(device)
    variance_kj = torch.from_numpy(variance_kj).to(device) if isinstance(variance_kj, np.ndarray) else variance_kj.to(device)
    pi_k = torch.from_numpy(pi_k).to(device) if isinstance(pi_k, np.ndarray) else pi_k.to(device)

    # Force to 1D
    mu_kj = mu_kj.view(-1)
    variance_kj = variance_kj.view(-1)
    pi_k = pi_k.view(-1)
    
    log_Pj_H0 = compute_log_Pj_H0(mu_kj, variance_kj, pi_k)
    albf = -log_Pj_H0
    albf = torch.where(albf == -0.0, torch.tensor(0.0), albf)
    
    return albf.cpu().numpy(), log_Pj_H0.cpu()

def plot_gaussian(means, variances):
    """
    Plots Gaussian distributions given lists of means and variances.
    
    Parameters:
        means (list of float): List of mean values for the Gaussians.
        variances (list of float): List of variance values for the Gaussians.
    """
    if len(means) != len(variances):
        raise ValueError("Means and variances lists must have the same length.")
    
    print(f"The means are: {means}")
    print(f"The variances are: {variances}")

    # Compute standard deviations from variances
    std_devs = np.sqrt(variances)

    # Dynamically determine the x-range
    x_min = min(means) - 3 * max(std_devs)
    x_max = max(means) + 3 * max(std_devs)

    # Whichever is absolute bigger number set range to be that value on both sides 
    if abs(x_min) > abs(x_max):
        x_max = abs(x_min)
    else:
        x_min = -abs(x_max)

    x = np.linspace(x_min, x_max, 1000)
    
    # Plot Gaussian distributions
    fig, ax = plt.subplots(figsize=(6, 6))
    for mu, sigma in zip(means, std_devs):
        ax.plot(x, norm.pdf(x, mu, sigma))
        # Sample z from the Gaussian distribution
        z = np.random.normal(mu, sigma, 5)
        # Get PSI values from z
        psi_values = inverse_logit(torch.tensor(z))
        print(f"The mean of the sampled PSI values is: {psi_values.mean()}")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Gaussian Distributions")
    plt.show()

def compute_z_score_dss(psis_loc, psis_scale, pi_k, junction_ids):
    """
    Compute z-score based differential splicing score comparing each factor
    to the overall weighted mean, and calculate corresponding p-values.
    
    Parameters:
    - psis_loc: Location parameters from VI (mean of PSI), shape (K x J)
    - psis_scale: Scale parameters from VI (standard error of PSI), shape (K x J)
    - pi_k: Factor contributions vector, shape (K,)
    - junction_ids: List/array of junction identifiers, shape (J,)
    
    Returns:
    - pd.DataFrame: Z-scores and p-values with shape (J x 2K), indexed by junction_ids
    """
    from scipy.stats import chi2

    # Convert to numpy if they're torch tensors
    if hasattr(psis_loc, 'detach'):
        psis_loc = psis_loc.detach().cpu().numpy()
        psis_scale = psis_scale.detach().cpu().numpy()
    
    if hasattr(pi_k, 'detach'):
        pi_k = pi_k.detach().cpu().numpy()
    
    # Compute variances
    var_jk = psis_scale ** 2 # (K x J)
    print(f"The shape of var_jk is: {var_jk.shape}")
    
    # Compute overall weighted mean μ_j for each junction
    mu_j = np.dot(pi_k, psis_loc) # (J,)
    print(f"The length of mu_j is: {len(mu_j)}")

    # # Compute variance of weighted mean
    var_mu_j = np.sum((pi_k[:, None]**2) * var_jk, axis=0) # sum over K to get (J,)
    print(f"The length of var_mu_j is: {len(var_mu_j)}")
    
    # Compute z-scores
    K = len(pi_k)
    z_scores = np.zeros_like(psis_loc) # (K x J)
    
    for k in range(K):
        standard_error = np.sqrt(var_jk[k, :] + var_mu_j)  # (J,)
        z_scores[k, :] = (psis_loc[k, :] - mu_j) / standard_error  # (J,)
    
    # Transpose to get (J x K) shape and convert to DataFrame
    z_scores = z_scores.T
    
    # Calculate p-values
    chi_square_stats = z_scores ** 2
    p_values = 1 - chi2.cdf(chi_square_stats, df=1)

    # Create DataFrame with proper column names
    z_columns = [f"factor_{k}" for k in range(K)]
    p_columns = [f"factor_{k}_pvalue" for k in range(K)]
    # Convert junction_ids to strings if they aren't already
    junction_ids = pd.Index(junction_ids).astype(str)
    
    # Convert junction_ids to strings if they aren't already
        # Combine z-scores and p-values
    df = pd.DataFrame(
        np.hstack([z_scores, p_values]),
        index=junction_ids,
        columns=z_columns + p_columns
    )
        
    return df

def compute_confidence_weighted_fw_dss(df, pi_k):
    """
    Compute Confidence-Weighted Factor-Weighted Differential Splicing Score (FW-DSS) 

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'mu_k' and 'var_k' for each factor.
    - pi_k (np.array): Factor contributions (size K).

    Returns:
    - pd.Series: FW-DSS scores for each junction.
    """
    # Number of factors (K)
    K = len(pi_k)
    
    # Extract column names dynamically
    mu_cols = [f"loc_{k}" for k in range(K)]
    var_cols = [f"scale_{k}" for k in range(K)]  # Assuming 'loc_k' represents posterior variance

    # Convert to numpy arrays for efficient computation
    mu_jk = df[mu_cols].to_numpy()  # (J x K) matrix of mean junction usage per factor
    var_jk = df[var_cols].to_numpy()  # (J x K) matrix of variances

    # Compute population-wide mean for each junction
    mu_pop_j = np.dot(mu_jk, pi_k)  # ends up being a vector of size J

    # Compute Confidence-Weighted FW-DSS:
    confidence_weighted_variance = np.sum(pi_k * ((mu_jk - mu_pop_j[:, None])**2) / (1 + var_jk), axis=1)

    # We need to motivate the 1 here in the denominator... instead what we should do 
    # is make this more like a z-score more natural... 
    # we have z_jk = mu_jk - mu_j / standard error on (mu_jk - mu_j)
    # where standard error is sqrt(var_jk + var_j)

    # Return as pandas Series
    return pd.Series(confidence_weighted_variance, index=df.index, name="FW-DSS")

#def compute_DS(): 

    # for every factor, extract cell PHI values for that factor and correspondign PSI values for junctions in that factor 
    # multiple them to get the expected PSI value for that junction in that factor across all cells
    # then take the rest K-1 factors and do the same thing and get the expected PSI value for that junction in that factor across all cells

    # Then the effect sizes will be the differences between these two expected PSI values across conditions 
    # This will be a way to finding splice junction markers for each factor 



