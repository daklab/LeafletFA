import warnings
import numpy as np
import torch 
import torch.nn.functional as F
import pandas as pd
    
def combined_weighted_mean_variance(means, variances, pis):
    """
    Compute combined mean and variance by weighting individual estimates by their precision.
    This is not a mixture of Gaussians, but rather a precision-weighted combination where:
    - More precise estimates (smaller variance) get more weight
    - Estimates are weighted by both their precision and their mixing proportion
    
    Args:
        means: Tensor of mean estimates for each component (shape: [K])
        variances: Tensor of variance estimates for each component (shape: [K])
        pis: Weights for each component (shape: [K])
    
    Returns:
        combined_mean: The combined mean estimate (scalar)
        combined_variance: The combined variance estimate (scalar)
    """
    precisions = 1.0 / variances  # Compute precisions
    weighted_precisions = torch.tensor(pis) * precisions

    combined_variance = 1.0 / torch.sum(weighted_precisions)  # Inverse sum of weighted precisions
    combined_mean = combined_variance * torch.sum(weighted_precisions * means)  # Precision-weighted mean

    return combined_mean, combined_variance


def gaussian_log_pdf(x, mean, std):
    """
    Compute the log probability density function (log-PDF) of a Gaussian distribution.

    Args:
        x: The value(s) to evaluate (Tensor)
        mean: The mean of the Gaussian distribution (scalar)
        std: The standard deviation of the Gaussian (scalar)
    
    Returns:
        log_pdf: The log probability density value (Tensor)
    """
    var = std ** 2  # Convert std to variance
    log_pdf = -0.5 * (torch.log(2 * torch.pi * var) + ((x - mean) ** 2 / var))
    return log_pdf


def log_like_under_null(means, variances, pis):
    """
    Compute the log-likelihood under the null hypothesis (H0).
    
    Args:
        means: Tensor of mean estimates for each component (shape: [K])
        variances: Tensor of variance estimates for each component (shape: [K])
        pis: Weights for each component (shape: [K])
    
    Returns:
        log_Pj_H0: Log-likelihood under the null hypothesis (scalar)
    """
    combined_mean, combined_variance = combined_weighted_mean_variance(means, variances, pis)
    
    # Compute log probabilities at zero
    log_prob_components = torch.sum(gaussian_log_pdf(torch.tensor(0.0), means, torch.sqrt(variances)))
    log_prob_combined = gaussian_log_pdf(torch.tensor(0.0), combined_mean, torch.sqrt(combined_variance))

    # Compute log-likelihood under null hypothesis
    log_Pj_H0 = log_prob_components - log_prob_combined
    return log_Pj_H0

def compute_albf(psis_mus, psis_loc, pis):
    """
    Compute Approximate Log Bayes Factor (ALBF).
    
    ALBF = log P(H1) - log P(H0)
    
    - Positive ALBF: Evidence for H1 (differential splicing)
    - Negative ALBF: Evidence for H0 (no differential splicing)
    
    Args:
        psis_mus: Tensor of means for each component (shape: [K])
        psis_loc: Tensor of variances for each component (shape: [K])
        pis: Tensor of mixing proportions (shape: [K])
    
    Returns:
        albf: Approximate Log Bayes Factor (scalar)
        log_likelihood_H0: Log likelihood under null hypothesis (scalar)
    """

    # Compute log-likelihood under the null hypothesis (H0)
    log_likelihood_H0 = log_like_under_null(psis_mus, psis_loc, pis)

    # Under H1, the posterior integrates to 1, so log P(H1) = 0
    albf = -log_likelihood_H0
    return albf, log_likelihood_H0





