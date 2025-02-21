import numpy as np
import torch 
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.metrics import silhouette_score
from scipy.stats import chi2
from tqdm import tqdm
from typing import Tuple, Dict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
import sys 
import os 

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
    Compute factor specific z-score based differential splicing score comparing each factor
    to the overall weighted mean, and calculate corresponding p-values.
    
    Parameters:
    - psis_loc: Location parameters from gaussian variational param (mean of PSI), shape (K x J)
    - psis_scale: Scale parameters from gaussian variational param (standard error of PSI), shape (K x J)
    - pi_k: Latent factor contributions vector, shape (K,)
    - junction_ids: List of junction indices, shape (J,)
    
    Returns:
    - pd.DataFrame: Z-scores and p-values with shape (J x 2K), indexed by junction_ids
    """

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
    psis_loc = np.clip(psis_loc, 0.0001, 0.9999)
    mu_j = np.dot(pi_k, psis_loc) # (J,)
    print(f"The length of mu_j is: {len(mu_j)}")

    # # Compute variance of weighted mean
    var_mu_j = np.sum((pi_k[:, None]**2) * var_jk, axis=0) # sum over K to get (J,)
    print(f"The length of var_mu_j is: {len(var_mu_j)}")
    
    # Compute z-scores
    K = len(pi_k)
    z_scores = np.zeros_like(psis_loc) # (K x J)

    var_jk = np.maximum(var_jk, 1e-4)  # Prevents near-zero variance
    var_mu_j = np.maximum(var_mu_j, 1e-4)

    print("Min variance value:", np.min(var_jk + var_mu_j))
    print("Median variance value:", np.median(var_jk + var_mu_j))
    print("Max variance value:", np.max(var_jk + var_mu_j))
    print("Max absolute difference (μ_{jk} - μ_j):", np.max(np.abs(psis_loc - mu_j)))

    # Calculate z-scores for each factor comparing to overall weighted mean
    for k in range(K):
        standard_error = np.sqrt(var_jk[k, :] + var_mu_j + 1e-6) # (J,)
        z_scores[k, :] = (psis_loc[k, :] - mu_j) / standard_error  # (J,)
    
    # Transpose to get (J x K) shape and convert to DataFrame
    z_scores = z_scores.T
    
    # Calculate p-values using chi-square distribution
    chi_square_stats = z_scores ** 2
    p_values = 1 - chi2.cdf(chi_square_stats, df=1)

    # Create DataFrame with proper column names
    z_columns = [f"factor_{k}" for k in range(K)]
    p_columns = [f"factor_{k}_pvalue" for k in range(K)]
    junction_ids = pd.Index(junction_ids).astype(str) # Convert junction_ids to strings to match Anndata index
    
    # Combine z-scores and p-values
    df = pd.DataFrame(
        np.hstack([z_scores, p_values]),
        index=junction_ids,
        columns=z_columns + p_columns
    )
        
    return df

def calculate_silhouette_score(assign_post, cell_types):
    """Calculates silhouette score for the factor assignments."""
    return silhouette_score(assign_post, cell_types)

def analyze_differential_splicing(leaflet_model, 
                            factor_idx = 0, 
                            fdr_threshold: float=0.05, 
                            dist_sd: float=0.5,
                            min_effect_size: float=None,
                            eps = 1e-10):
    
    # Get posterior samples (sampled from variational distribution / guide)
    psi_samples = leaflet_model.psi_samples  # [n_samples, K, J]
    phi_samples = leaflet_model.phi_samples  # [n_samples, C, K]

    # Convert to numpy if they're torch tensors
    if hasattr(psi_samples, 'detach'):
        psi_samples = psi_samples.detach().cpu().numpy()
    if hasattr(phi_samples, 'detach'):
        phi_samples = phi_samples.detach().cpu().numpy()

    n_samples, n_cells, n_factors = phi_samples.shape
    _, _, n_junctions = psi_samples.shape

    # Initialize array to store effect sizes
    effect_sizes = np.zeros((n_samples, n_junctions))

    for s in tqdm(range(n_samples)):
        phi_s = phi_samples[s]  # [C, K]
        psi_s = psi_samples[s]  # [K, J]

        # Calculate factor k usage
        factor_usage = phi_s[:, factor_idx].reshape(-1, 1) * psi_s[factor_idx, :]  # (C, J)

        # Calculate other factors usage
        other_factor_indices = [k for k in range(n_factors) if k != factor_idx]
        if len(other_factor_indices) == 1:
            k = other_factor_indices[0]
            other_usage = phi_s[:, k].reshape(-1, 1) * psi_s[k, :]  # (C, J)
        else:
            other_usage = sum(
                phi_s[:, k].reshape(-1, 1) * psi_s[k, :] for k in other_factor_indices
            )  # (C, J)

        # Compute cell-wise effect sizes first and then aggregate:
        effect_sizes[s] = np.mean(factor_usage - other_usage, axis=0)

    # Calculate overall effect size and variance / aggregates posterior samples
    beta = np.mean(effect_sizes, axis=0)
    beta_vars = np.var(effect_sizes, axis=0)

    # Estimate delta if not provided (use posterior variance for robustness)
    # aka : call a junction significantly DS if the effect 
    # size is at least half a standard deviation away from the mean of beta... 
    if min_effect_size is None:
        min_effect_size = dist_sd * np.std(beta)

    print(f"Calculating probailities for effect size > {min_effect_size}...")
    # Calculate probabilities using effect sizes
    prob_greater = np.mean(np.abs(effect_sizes) > min_effect_size, axis=0)
    prob_lessoreq = 1 - prob_greater

    # Compute posterior expected False Discovery Proportion (FDP)
    sorted_idx = np.argsort(-prob_greater)
    sorted_probs = prob_greater[sorted_idx]

    # Calculate FDR 
    fdrs = np.cumsum(1 - sorted_probs) / (np.arange(len(sorted_probs)) + 1)
    max_discoveries = np.where(fdrs <= fdr_threshold)[0]
    n_significant = len(max_discoveries) if len(max_discoveries) > 0 else 0

    # Define significant junctions
    significant = np.zeros(n_junctions, dtype=bool)
    if n_significant > 0:
        significant[sorted_idx[:n_significant]] = True

    # Apply effect size threshold
    significant = significant & (np.abs(beta) > min_effect_size)
    print(f"Found {n_significant} significant junctions with effect size > {min_effect_size} at FDR < {fdr_threshold}")

    # Construct results dictionary
    results_dict = {
        'effect_sizes': beta,
        'effect_size_vars': beta_vars,
        'prob_greater': prob_greater,
        'prob_lessoreq': prob_lessoreq,
        'significant': significant,
        'delta': min_effect_size,
        'fdr_curve': fdrs,
        'n_significant': n_significant
    }

    # Create results DataFrame
    results_df = pd.DataFrame({
        'junction_idx': np.arange(n_junctions),
        'effect_size': beta,
        'effect_size_std': np.sqrt(beta_vars),
        'prob_greater': prob_greater,
        'prob_lessoreq': prob_lessoreq,
        'significant': significant,
        'delta': min_effect_size
    })

    return results_dict, results_df

def analyze_all_factors(
    leaflet_model,
    min_effect_size: float = None,
    dist_sd: float = 0.5,
    fdr_threshold: float = 0.05
) -> Dict[int, Tuple[Dict[str, np.ndarray], pd.DataFrame]]:
    """
    Analyze differential splicing for all factors.
    Note: For K=2 factors, only analyzes factor 0 vs 1 to avoid redundancy.
    
    Args:
        leaflet_model: LeafletFA model object
        n_samples: Number of bootstrap samples
        min_effect_size: Minimum effect size threshold
        fdr_threshold: FDR threshold
    
    Returns:
        Dictionary mapping factor index to (results_dict, results_df) tuple
    """
    n_factors = leaflet_model.psi_learned.shape[0]
    results = {}
    
    # For K=2, only analyze factor 0 vs 1
    if n_factors == 2:
        results[0] = analyze_differential_splicing(
            leaflet_model,
            factor_idx=0,
            min_effect_size=min_effect_size,
            fdr_threshold=fdr_threshold
        )
    else:
        # For K>2, analyze each factor vs others
        print(f"Analyzing differential splicing for {n_factors} factors...")
        for k in range(n_factors):
            print(f"Finding DS junctions in factor {k} vs. all others")
            results[k] = analyze_differential_splicing(
                leaflet_model,
                factor_idx=k,
                min_effect_size=min_effect_size,
                fdr_threshold=fdr_threshold, 
                dist_sd=dist_sd
            )
    return results

def calibration_test(leaflet_model, true_positive_junctions, min_effect_size=None, fdr_thresholds=[0.01, 0.05, 0.1, 0.2]):
    """
    Tests calibration of the Bayesian FDR control by comparing expected vs. observed FDR.

    Args:
        leaflet_model: LeafletFA model object.
        true_positive_junctions (set): Indices of true differentially spliced junctions (for empirical FDR calculation).
        fdr_thresholds (list): List of FDR thresholds to test.

    Returns:
        DataFrame with expected vs. observed FDR values.
    """
    results = []

    for fdr in fdr_thresholds:
        print(f"Running with FDR threshold: {fdr}")
        results_dict, results_df = analyze_differential_splicing(leaflet_model, fdr_threshold=fdr, min_effect_size=min_effect_size)

        # Extract discovered junctions
        discovered_junctions = set(results_df[results_df['significant']].junction_idx)

        # Compute observed FDR
        false_positives = discovered_junctions - true_positive_junctions
        observed_fdr = len(false_positives) / max(len(discovered_junctions), 1)  # Avoid division by zero

        # Store results
        results.append({'FDR_threshold': fdr, 'Observed_FDR': observed_fdr, 'Total Discoveries': len(discovered_junctions)})

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Plot expected vs. observed FDR
    plt.figure(figsize=(6, 5))
    plt.plot(df_results['FDR_threshold'], df_results['FDR_threshold'], '--', label="Ideal Calibration (y=x)")
    plt.scatter(df_results['FDR_threshold'], df_results['Observed_FDR'], color='red', label="Observed FDR")
    plt.xlabel("Expected FDR Threshold")
    plt.ylabel("Observed FDR")
    plt.legend()
    plt.title("FDR Calibration Test")
    plt.show()

    return df_results


def plot_precision_recall_curve(adata_input, results_df, output_dir=None):
    """
    Plots a Precision-Recall (PR) curve for differential splicing analysis.
    
    Args:
        adata_input: AnnData object containing true labels in `adjusted_true_label`.
        results_df: DataFrame containing posterior probabilities (`prob_greater`) from `analyze_differential_splicing`.

    Returns:
        Plots the PR curve.
    """
    # Ensure labels are binary (1 = True DS, 0 = Not DS)
    true_labels = (adata_input.var["adjusted_true_label"] == "positive").astype(int).values

    # Use posterior probability as score
    predicted_scores = results_df["prob_greater"].values

    # Compute precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_scores)

    # Compute area under PR curve (AUC-PR)
    auc_pr = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, marker='.', label=f'PR Curve (AUC = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Differential Splicing')
    plt.legend()
    plt.grid()
    
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    
    plt.show()
    return precision, recall, auc_pr

def plot_roc_curve(adata_input, results_df, output_dir=None):
    """
    Plots a Receiver Operating Characteristic (ROC) curve for differential splicing analysis.

    Args:
        adata_input: AnnData object containing true labels in `adjusted_true_label`.
        results_df: DataFrame containing posterior probabilities (`prob_greater`) from `analyze_differential_splicing`.

    Returns:
        Plots the ROC curve and prints the AUC.
    """
    # Ensure labels are binary (1 = True DS, 0 = Not DS)
    true_labels = (adata_input.var["adjusted_true_label"] == "positive").astype(int).values

    # Use posterior probability as score
    predicted_scores = results_df["prob_greater"].values

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)

    # Compute area under ROC curve (AUC-ROC)
    auc_roc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, marker='.', label=f'ROC Curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')  # Random chance line
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for Differential Splicing')
    plt.legend()
    plt.grid()

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))

    plt.show()  # Display plot after saving
    return fpr, tpr, auc_roc