import numpy as np
import torch 
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Function for the logit transform
def logit(p):
    return torch.log(p / (1 - p))

# Function for the inverse logit transform (sigmoid)
def inverse_logit(z):
    return 1 / (1 + torch.exp(-z))

# Function to sample z from gaussian distribution using mu_kj and sigma_kj
def sample_z(mu_kj, sigma_kj, num_sample=10):
    return np.random.normal(mu_kj, sigma_kj, num_sample)

def plot_junc_dists(albf_scores, index_junc, pis):
    mus = torch.tensor(albf_scores[["mu_0", "mu_1"]].iloc[index_junc].values)
    variances = torch.tensor(albf_scores[["loc_0", "loc_1"]].iloc[index_junc].values)
    print(f"The means are: {mus}")
    print(f"The variances are: {variances}")

    # compute combined mean and variance
    combined_mean, combined_variance = compute_combined_parameters(mus, variances, pis)
    print(f"The combined mean is: {combined_mean}")
    print(f"The combined variance is: {combined_variance}")

    # Make a list of original mus plus combined one and variances plus combined one to plot
    mus_list = mus.tolist()
    mus_list.append(combined_mean.item())
    variances_list = variances.tolist()
    variances_list.append(combined_variance.item())

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

def analyze_null_albf(df):
    """
    Analyze ALBF distribution for negative junctions
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'ALBF' and 'true_label' columns
    
    Returns
    -------
    dict
        Dictionary containing null distribution parameters
    """
    # Get ALBF values for negative junctions
    null_albf = df[df['true_label'] == 'negative']['ALBF']
    
    # Fit normal distribution to null ALBFs
    mu, std = stats.norm.fit(null_albf)
    
    # Test for normality
    _, norm_pval = stats.normaltest(null_albf)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of null ALBF values
    sns.histplot(null_albf, bins=30, color='blue', alpha=0.5, label='Observed')
    
    # Plot fitted normal distribution
    x = np.linspace(null_albf.min(), null_albf.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, std) * len(null_albf) * (null_albf.max() - null_albf.min()) / 30,
             'r-', label='Fitted Normal')
    
    plt.title(f'Distribution of ALBF Values for Negative Junctions\nμ={mu:.2f}, σ={std:.2f}, p={norm_pval:.2e}')
    plt.xlabel('ALBF')
    plt.ylabel('Count')
    plt.legend()
    
    # Calculate percentiles for potential thresholds
    percentiles = [90, 95, 99]
    thresholds = np.percentile(null_albf, percentiles)
    
    return {
        'mu': mu,
        'std': std,
        'normality_pvalue': norm_pval,
        'thresholds': dict(zip(percentiles, thresholds))
    }