import numpy as np
import torch 

def combined_mean_variance(means, variances, pis):
    eps = 1e-5  # Small value to ensure numerical stability
    variances = torch.clamp(variances, min=eps)
    inv_variances = 1 / variances

    # Convert pis (NumPy array) to PyTorch tensor and ensure correct dimensions
    pis = torch.tensor(pis, dtype=torch.float32, device=means.device).view(-1, 1)

    # Weighting by pis
    weighted_inv_variances = pis * inv_variances
    combined_variance = 1 / torch.sum(weighted_inv_variances, dim=0)
    combined_mean = combined_variance * torch.sum(means * weighted_inv_variances, dim=0)
    
    return combined_mean, combined_variance

def gaussian_log_pdf(x, mean, std):
    eps = 1e-10  # Small value to ensure numerical stability
    var = std ** 2 + eps
    log_denom = 0.5 * torch.log(2 * torch.pi * var)
    log_num = - (x - mean) ** 2 / (2 * var)
    return log_num - log_denom

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        print(f"{name} contains infinity values")

def likelihood_under_null(means, variances, pis):
    combined_mean, combined_variance = combined_mean_variance(means, variances, pis)
    combined_std = combined_variance ** 0.5

    # Calculate the log of the combined Gaussian PDF at zero
    combined_log_pdf_zero = gaussian_log_pdf(torch.tensor(0.0), combined_mean, combined_std)
    
    # Calculate the weighted sum of logs of Gaussian PDFs evaluated at zero for each mean and std
    log_pdfs_zero = gaussian_log_pdf(torch.tensor(0.0), means, variances ** 0.5)
    pis = torch.tensor(pis, dtype=torch.float32, device=means.device).view(-1, 1)  # Ensure correct tensor type and shape
    weighted_sum_log_pdfs_zero = torch.sum(pis * log_pdfs_zero, dim=0)

    # Calculate the log likelihood under H0
    log_likelihood_H0 = weighted_sum_log_pdfs_zero - combined_log_pdf_zero
    
    # Check for NaN or infinity values
    check_for_nan_inf(combined_mean, "Combined Mean")
    check_for_nan_inf(combined_variance, "Combined Variance")
    check_for_nan_inf(combined_log_pdf_zero, "Combined Log PDF Zero")
    check_for_nan_inf(weighted_sum_log_pdfs_zero, "Weighted Sum Log PDFs Zero")
    check_for_nan_inf(log_likelihood_H0, "Log Likelihood H0")
    
    return torch.exp(log_likelihood_H0)  # Convert log likelihood back to standard likelihood

def compute_albf(psis_mus, psis_loc, pis, eps = 1e-10):

    likelihood_H0 = likelihood_under_null(psis_mus, psis_loc, pis)

    # Compute likelihood under H_0 for each junction
    likelihood_H0 = torch.clamp(likelihood_H0, min=eps)

    # Compute ALBF for each junction
    albf = -torch.log(likelihood_H0)
    
    # Convert -0.0 to 0
    albf = torch.where(albf == -0.0, torch.tensor(0.0), albf)
    
    return albf, likelihood_H0