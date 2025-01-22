import numpy as np
import torch 
import torch.nn.functional as F
import pandas as pd

def combined_mean_variance(means, variances, pis):
    eps = 1e-5  # Small value to ensure numerical stability
    variances = torch.clamp(variances, min=eps)
    inv_variances = 1 / variances

    # Convert pis to PyTorch tensor and ensure correct dimensions
    pis = torch.tensor(pis, dtype=torch.float32, device=means.device).view(-1, 1)

    # Weighting by pis
    # This computes a single combined Gaussian distribution from multiple components, weighted by their mixing proportions.
    weighted_inv_variances = pis * inv_variances
    combined_variance = 1 / torch.sum(weighted_inv_variances, dim=0)
    combined_mean = combined_variance * torch.sum(means * weighted_inv_variances, dim=0)
    
    return combined_mean, combined_variance

def combined_mean_variance_exclude(means, variances, pis, exclude_index=0):
    """
    Compute the combined mean and variance for all factors except the specified one.
    
    Args:
        means: Tensor of means for all factors.
        variances: Tensor of variances for all factors.
        pis: Tensor or NumPy array of mixing proportions (if applicable).
        exclude_index: The index of the factor to exclude from the combination (e.g., 0 to exclude factor 1).
    
    Returns:
        combined_mean: The combined mean of all factors except the excluded one.
        combined_variance: The combined variance of all factors except the excluded one.
    """
    eps = 1e-5  # Small value to ensure numerical stability
    variances = torch.clamp(variances, min=eps)

    # Convert pis to a PyTorch tensor if it's a NumPy array
    if isinstance(pis, np.ndarray):
        pis = torch.tensor(pis, dtype=torch.float32, device=means.device)
    
    # Exclude the factor at the specified index
    means_excluded = torch.cat([means[:exclude_index], means[exclude_index + 1:]])
    variances_excluded = torch.cat([variances[:exclude_index], variances[exclude_index + 1:]])
    pis_excluded = torch.cat([pis[:exclude_index], pis[exclude_index + 1:]])

    inv_variances = 1 / variances_excluded
    pis_excluded = pis_excluded.view(-1, 1)  # Ensure correct shape

    # Compute combined mean and variance for the remaining factors
    weighted_inv_variances = pis_excluded * inv_variances
    combined_variance = 1 / torch.sum(weighted_inv_variances, dim=0)
    combined_mean = combined_variance * torch.sum(means_excluded * weighted_inv_variances, dim=0)
    
    return combined_mean, combined_variance

def gaussian_log_pdf(x, mean, std):
    eps = 1e-10  # Small value to ensure numerical stability
    var = std ** 2 + eps
    log_denom = 0.5 * torch.log(2 * torch.pi * var)
    log_num = - (x - mean) ** 2 / (2 * var)
    return log_num - log_denom

def log_sum_exp(x, dim=0):
    """
    Numerically stable implementation of log(Σᵢ exp(xᵢ))
    
    To prevent overflow/underflow, we use the identity:
    log(Σᵢ exp(xᵢ)) = α + log(Σᵢ exp(xᵢ - α))
    where α = max(xᵢ)
    """
    # Find maximum value for numerical stability
    max_x, _ = torch.max(x, dim=dim, keepdim=True)
    
    # Subtract maximum (in exp space, this is division)
    x_shifted = x - max_x
    
    # Compute log-sum-exp with shifted values
    return max_x + torch.log(torch.sum(torch.exp(x_shifted), dim=dim))

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        print(f"{name} contains infinity values")

def likelihood_under_null(means, variances, pis):

    # This is my original function for this likelihood calculation 
    combined_mean, combined_variance = combined_mean_variance(means, variances, pis)
    combined_std = combined_variance ** 0.5

    # Calculate probability of 0 under the combined distribution
    # Calculate the log of the combined Gaussian PDF at zero
    combined_log_pdf_zero = gaussian_log_pdf(torch.tensor(0.0), combined_mean, combined_std)
    
    # Calculate probability of 0 under each component and weight by pis
    log_pdfs_zero = gaussian_log_pdf(torch.tensor(0.0), means, variances ** 0.5)
    sum_log_pdfs_zero = torch.sum(log_pdfs_zero, dim=0)
    
    # pis = torch.tensor(pis, dtype=torch.float32, device=means.device).view(-1, 1)  
    # weighted_sum_log_pdfs_zero = torch.sum(pis * log_pdfs_zero, dim=0)

    # Calculate the log likelihood under H0
    log_likelihood_H0 = sum_log_pdfs_zero - combined_log_pdf_zero
    
    # Check for NaN or infinity values
    check_for_nan_inf(combined_mean, "Combined Mean")
    check_for_nan_inf(combined_variance, "Combined Variance")
    check_for_nan_inf(combined_log_pdf_zero, "Combined Log PDF Zero")
    check_for_nan_inf(sum_log_pdfs_zero, "Sum Log PDFs Zero")
    check_for_nan_inf(log_likelihood_H0, "Log Likelihood H0")
    return log_likelihood_H0  # Convert log likelihood back to standard likelihood
 
def compute_albf(psis_mus, psis_loc, pis, eps = 1e-20):
    # This is the original function for computing ALBF
    # with somewhat messy handling of numerical stability  
    log_likelihood_H0 = likelihood_under_null(psis_mus, psis_loc, pis)
     
    #likelihood_H0 = torch.clamp(likelihood_H0, min=eps)
     
    # Compute ALBF for each junction
    albf = -log_likelihood_H0
    
    # Convert -0.0 to 0
    albf = torch.where(albf == -0.0, torch.tensor(0.0), albf)
    return albf, log_likelihood_H0 

def all_vs_one_differential_splicing_test(means, variances, pis, factor_index=0):
    """
    Perform an All-vs-1 differential splicing test comparing one factor to the combined effect of others.
    
    Args:
        means: Tensor of means for all factors.
        variances: Tensor of variances for all factors.
        pis: Tensor of mixing proportions (if applicable).
        factor_index: Index of the factor to compare against all others.
    
    Returns:
        albf: The computed ALBF for the differential splicing test.
        likelihood_H0: The likelihood under the null hypothesis.
    """
    # Get the mean and variance for the factor we're comparing (e.g., factor 1)
    mu_1 = means[factor_index]
    v_1 = variances[factor_index]
    
    # Calculate the combined mean and variance for all other factors (e.g., factors 2 through K)
    mu_combined, v_combined = combined_mean_variance_exclude(means, variances, pis, exclude_index=factor_index)
    
    # Now, we treat this as a comparison between two groups: the single factor vs. the combined group
    means_test = torch.stack([mu_1, mu_combined])
    variances_test = torch.stack([v_1, v_combined])
    pis_test = torch.tensor([0.5, 0.5], dtype=torch.float32, device=means.device)  # Equal weighting for the two groups
    
    # Compute ALBF and likelihood under the null hypothesis
    albf, likelihood_H0 = compute_albf(means_test, variances_test, pis_test)
    
    return albf

def all_vs_all_differential_splicing_test(means, variances, pis):
    """
    Perform an All-vs-All differential splicing test, comparing each factor to the combined effect of others.
    
    Args:
        means: Tensor of means for all factors.
        variances: Tensor of variances for all factors.
        pis: Tensor of mixing proportions (if applicable).
    
    Returns:
        markers_df: DataFrame with junction index, factor, and corresponding ALBF values.
    """
    n_factors = means.size(0)  # Number of factors (K)
    results = []

    for i in range(n_factors):
        albf = all_vs_one_differential_splicing_test(means, variances, pis, factor_index=i)

        # Iterate through each element of ALBF tensor to store individual ALBF values
        for idx, albf_val in enumerate(albf):
            result = {
                'Factor': f'Factor {i+1}',  # Label the factor (1-based index)
                'Junction_Index': idx,       # Index within the ALBF tensor
                'ALBF': albf_val.item()
            }
            results.append(result)

    # Create a DataFrame to store all results
    markers_df = pd.DataFrame(results)
    return markers_df