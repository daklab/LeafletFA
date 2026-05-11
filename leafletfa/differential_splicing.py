import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from typing import Tuple, Dict

def calculate_silhouette_score(assign_post, cell_types):
    """Calculates silhouette score for the factor assignments."""
    return silhouette_score(assign_post, cell_types)

def plot_psi_distribution(psi_samples, junction_idx=None, ref_factor=None):

    # Ensure data is on CPU if using GPU tensors
    psi_samples = psi_samples.cpu()

    # Get the number of factors and junctions
    K = psi_samples.shape[1]
    J = psi_samples.shape[2]

    # Randomly select a junction if not provided
    if junction_idx is None:
        junction_idx = np.random.choice(J, 1)[0]

    # Randomly select a reference factor if not provided
    if ref_factor is None:
        ref_factor = np.random.choice(K, 1)[0]

    # Extract PSI values for the reference factor and non-reference factors
    psi_ref = psi_samples[:, ref_factor, junction_idx].numpy()  # Reference factor PSI
    psi_nonref = psi_samples[:, np.arange(K) != ref_factor, junction_idx].reshape(-1).numpy()  # Other factors PSI

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        "PSI Value": np.concatenate([psi_ref, psi_nonref]),
        "Group": ["Reference Factor"] * len(psi_ref) + ["Other Factors"] * len(psi_nonref)
    })

    # Create the plot
    plt.figure(figsize=(5, 5))

    # Plot side-by-side violin plots
    sns.violinplot(x="Group", y="PSI Value", data=df, inner="quartile", linewidth=1, palette={"Reference Factor": "skyblue", "Other Factors": "orange"})

    # Labels and title
    plt.ylabel("PSI values")
    plt.xlabel("")
    plt.title(f"PSI Distribution for Junction {junction_idx} (Ref: Factor {ref_factor})")

    # Show the plot
    plt.show()

    # Print means for reference
    print(f"The junction index is: {junction_idx}")
    print(f"The mean PSI of Reference Factor ({ref_factor}): {psi_ref.mean():.4f}")
    print(f"The mean PSI of Other Factors: {psi_nonref.mean():.4f}")

def compute_psi_effect_size(psi_samples, factor_idx, junction_idx, min_effect_size=0.1):
    """
    Computes the effect size for a given junction based on PSI values without weighting by factor usage.

    Args:
        psi_samples (torch.Tensor): Tensor of shape [S, K, J] representing sample-factor-junction usage.
        factor_idx (int): Index of the factor of interest.
        junction_idx (int): Index of the junction of interest.
        min_effect_size (float, optional): Threshold for significance. Default is 0.1.

    Returns:
        dict: Summary statistics of effect size for the given junction.
    """

    # Get the number of factors
    n_factors = psi_samples.shape[1]

    # Indices of all other factors
    other_factor_indices = [k for k in range(n_factors) if k != factor_idx]

    # Extract PSI for the factor of interest
    psi_j_factor = psi_samples[:, factor_idx, junction_idx]  # Shape: [S]

    # Extract PSI for all other factors and compute their mean
    psi_j_nonfactor = psi_samples[:, other_factor_indices, junction_idx]  # Shape: [S, K-1]
    psi_other_mean = psi_j_nonfactor.mean(dim=1)  # Mean PSI across other factors, shape: [S]

    # Compute log2 fold-change per sample
    psi_diff_samples = psi_j_factor - psi_other_mean  # Subtraction instead of log fold-change

    # Compute summary statistics across samples
    effect_size_mean = psi_diff_samples.mean()
    effect_size_var = psi_diff_samples.var()

    # Compute probability of effect size being greater than the threshold
    prob_greater = (torch.abs(psi_diff_samples) > min_effect_size).float().mean()
    num_greater = (torch.abs(psi_diff_samples) > min_effect_size).float().sum()
    prob_lessoreq = 1 - prob_greater

    # Return results in a dictionary
    return {
        'factor_idx': factor_idx,
        'junction_idx': junction_idx,
        'effect_size': effect_size_mean.item(),
        'effect_size_var': effect_size_var.item(),
        'prob_greater': prob_greater.item(),
        'num_greater': num_greater.item(),
        'prob_lessoreq': prob_lessoreq.item(),
        'psi_factor_mean': psi_j_factor.mean().item(),
        'psi_nonfactor_mean': psi_other_mean.mean().item()
    }

def compute_junctions_significance_psi(effect_sizes, fdr_threshold, min_effect_size=0.1):
    """
    Computes significant junctions based on PSI effect size probabilities and false discovery rate (FDR).

    Args:
        effect_sizes (pd.DataFrame): DataFrame containing results from compute_psi_effect_size.
        fdr_threshold (float): False discovery rate threshold.
        min_effect_size (float, optional): Minimum effect size threshold for significance. Default is 0.1.

    Returns:
        pd.DataFrame: DataFrame containing significant junctions and FDR calculations.
    """

    # Ensure required columns exist
    required_cols = ['prob_greater', 'effect_size', 'effect_size_var', 'junction_idx']
    for col in required_cols:
        if col not in effect_sizes.columns:
            raise ValueError(f"Missing column '{col}' in effect_sizes DataFrame.")

    # Extract values
    prob_greater = effect_sizes['prob_greater'].values # probability of effect size being greater than 0
    beta = effect_sizes['effect_size'].values
    beta_vars = effect_sizes['effect_size_var'].values
    n_junctions = len(effect_sizes)

    # Sort indices based on decreasing probability
    sorted_idx = np.argsort(-prob_greater)
    sorted_probs = prob_greater[sorted_idx]

    # Compute FDR using cumulative false discovery proportion
    fdrs = np.cumsum(1 - sorted_probs) / (np.arange(len(sorted_probs)) + 1)

    # Map FDR values back to original order
    fdrs_original_order = np.zeros(n_junctions)
    fdrs_original_order[sorted_idx] = fdrs  # Undo sorting

    # Identify significant junctions based on FDR threshold
    # Find the last rank where FDR is within threshold and apply effect size threshold separately
    significant = (fdrs_original_order <= fdr_threshold) & (np.abs(beta) > min_effect_size)

    # Count number of significant junctions
    n_significant = np.sum(significant)

    print(f"Found {n_significant} significant junctions with effect size > {min_effect_size} at FDR < {fdr_threshold}")

    # Construct results DataFrame
    results_df = pd.DataFrame({
        'factor_idx': effect_sizes['factor_idx'].values,
        'junction_idx': effect_sizes['junction_idx'].values,
        'effect_size': beta,
        'abs_effect_size': np.abs(beta),
        'effect_size_var': beta_vars,
        'num_greater': effect_sizes.get('num_greater', np.nan).values,
        'prob_greater': prob_greater,
        'prob_lessoreq': effect_sizes.get('prob_lessoreq', np.nan).values,
        'significant': significant,
        'delta': min_effect_size,
        'psi_factor_mean': effect_sizes.get('psi_factor_mean', np.nan).values,
        'psi_nonfactor_mean': effect_sizes.get('psi_nonfactor_mean', np.nan).values,
        'fdr_curve': fdrs_original_order,  # Now matches original order
        'n_significant': n_significant
    })

    return results_df


def analyze_all_factors_psi(
    psi_samples,
    top_junctions, 
    min_effect_size: float = 0.1,
    fdr_threshold: float = 0.05
) -> Dict[int, Tuple[Dict[str, np.ndarray], pd.DataFrame]]:
    """
    Analyze differential splicing for all factors using PSI values only.
    Identifies top differentially spliced (DS) junctions.

    Returns:
        Dictionary mapping factor index to (results_dict, results_df) tuple.
    """
    n_factors = psi_samples.shape[1]
    results = {}

    print(f"Analyzing differential splicing for {n_factors} factors...")

    for k in range(n_factors):
        print(f"Finding DS junctions in factor {k} vs. all others")
        top_fact_juncs = [] 
        factor_idx = k 
        for j in tqdm(top_junctions):
            results_dict = compute_psi_effect_size(
                psi_samples, factor_idx, j, min_effect_size=min_effect_size
            )
            top_fact_juncs.append(results_dict)

        # Convert to DataFrame
        top_fact_juncs_df = pd.DataFrame(top_fact_juncs)

        # Compute significance
        top_fact_juncs_df_SIG = compute_junctions_significance_psi(top_fact_juncs_df, fdr_threshold, min_effect_size)

        # Store results
        results[k] = (top_fact_juncs, top_fact_juncs_df_SIG)

    return results


def compute_differential_splicing_groups(
    adata, psi_samples, phi_samples, junction_idx, 
    group_1, group_2=None, groupby_column="cell_type_grouped", min_effect_size=0.1
):
    """
    Computes the effect size for a given junction between two groups (e.g., cell types or age groups).

    Args:
        adata (AnnData): The AnnData object containing cell metadata.
        psi_samples (torch.Tensor): Tensor of shape [S, K, J] representing sample-factor-junction usage.
        phi_samples (torch.Tensor): Tensor of shape [S, C, K] representing sample-cell-factor assignments.
        junction_idx (int): Index of the junction to analyze.
        group_1 (str): First group to compare (e.g., a specific cell type or age group).
        group_2 (str, optional): Second group to compare. If None, compares group_1 against all others.
        groupby_column (str, optional): Column in `adata.obs` to use for grouping (default: "cell_type_grouped").
        min_effect_size (float, optional): Threshold for significance. Default is 0.1.

    Returns:
        dict: Summary statistics of effect size for the given junction.
    """
    # Get cell indices for group_1
    cell_mask_1 = adata.obs[groupby_column] == group_1
    cell_indices_1 = np.where(cell_mask_1)[0]

    # Get cell indices for group_2 (if provided) or all others
    if group_2:
        cell_mask_2 = adata.obs[groupby_column] == group_2
        cell_indices_2 = np.where(cell_mask_2)[0]
    else:
        cell_mask_2 = ~cell_mask_1  # All other cells
        cell_indices_2 = np.where(cell_mask_2)[0]

    # Extract PHI for the selected groups
    phi_1 = phi_samples[:, cell_indices_1, :]  # Shape: [S, C1, K]
    phi_2 = phi_samples[:, cell_indices_2, :]  # Shape: [S, C2, K]

    # Extract PSI for the selected junction (same for all cells)
    psi_junction = psi_samples[:, :, junction_idx]  # Shape: [S, K]

    # Compute weighted PSI values
    weighted_psi_1 = (phi_1 * psi_junction.unsqueeze(1)).sum(dim=2)  # Shape: [S, C1]
    weighted_psi_2 = (phi_2 * psi_junction.unsqueeze(1)).sum(dim=2)  # Shape: [S, C2]

    # Step 1: Compute mean PSI per sample (across cells)
    mean_psi_1 = weighted_psi_1.mean(dim=1)  # Shape: [S] (S = number of samples)
    mean_psi_2 = weighted_psi_2.mean(dim=1)  # Shape: [S]

    # Step 2: Compute effect size per sample
    effect_size_per_sample = mean_psi_1 - mean_psi_2  # Shape: [S]

    # Compute summary statistics
    effect_size_mean = effect_size_per_sample.mean()  # Scalar value
    effect_size_var = effect_size_per_sample.var()  # Scalar value

    # Compute probability of effect size being greater than threshold
    prob_greater = (torch.abs(effect_size_per_sample) > min_effect_size).float().mean()
    num_greater = (torch.abs(effect_size_per_sample) > min_effect_size).float().sum()
    prob_lessoreq = 1 - prob_greater

    return {
        'junction_idx': junction_idx,
        'groupby_column': groupby_column,
        'group_1': group_1,
        'group_2': group_2 if group_2 else 'All Others',
        'effect_size': effect_size_mean.item(),
        'effect_size_var': effect_size_var.item(),
        'prob_greater': prob_greater.item(),
        'num_greater': num_greater.item(),
        'prob_lessoreq': prob_lessoreq.item(),
        'mean_psi_1': mean_psi_1.cpu().numpy(),
        'mean_psi_2': mean_psi_2.cpu().numpy()
    }


def compute_junctions_significance_groups(effect_sizes, fdr_threshold=0.05, min_effect_size=0.1):
    """
    Computes significant junctions based on effect size probabilities and false discovery rate (FDR).

    Args:
        effect_sizes (pd.DataFrame): DataFrame containing results from compute_junction_effect_size.
        fdr_threshold (float): False discovery rate threshold.
        min_effect_size (float, optional): Minimum effect size threshold for significance. Default is 0.1.

    Returns:
        pd.DataFrame: DataFrame containing significant junctions and FDR calculations.
    """

    # Ensure required columns exist
    required_cols = ['prob_greater', 'effect_size', 'effect_size_var', 'junction_idx']
    missing_cols = [col for col in required_cols if col not in effect_sizes.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in effect_sizes DataFrame: {missing_cols}")

    # Extract relevant values
    prob_greater = effect_sizes['prob_greater'].values
    beta = effect_sizes['effect_size'].values
    beta_vars = effect_sizes['effect_size_var'].values
    n_junctions = len(effect_sizes)

    # Sort indices based on decreasing probability of effect size significance
    sorted_idx = np.argsort(-prob_greater)
    sorted_probs = prob_greater[sorted_idx]

    # Compute FDR using cumulative false discovery proportion
    fdrs = np.cumsum(1 - sorted_probs) / (np.arange(len(sorted_probs)) + 1)
    
    # Determine significant junctions based on FDR threshold
    max_discoveries = np.where(fdrs <= fdr_threshold)[0]
    n_significant = len(max_discoveries) if len(max_discoveries) > 0 else 0

    # Initialize boolean mask for significance
    significant = np.zeros(n_junctions, dtype=bool)

    if n_significant > 0:
        significant[sorted_idx[:n_significant]] = True

    # Apply effect size threshold to final selection
    significant &= np.abs(beta) > min_effect_size
    n_significant = np.sum(significant)

    print(f"Found {n_significant} significant junctions with effect size > {min_effect_size} at FDR < {fdr_threshold}")

    # Construct results DataFrame
    results_df = pd.DataFrame({
        'junction_idx': effect_sizes['junction_idx'].values,
        'effect_size': beta,
        'group_1': effect_sizes['group_1'].values,
        'group_2': effect_sizes['group_2'].values,
        'abs_effect_size': np.abs(beta),
        'effect_size_var': beta_vars,
        'num_greater': effect_sizes['num_greater'].values,  
        'prob_greater': prob_greater,
        'prob_lessoreq': effect_sizes['prob_lessoreq'].values,
        'mean_psi_1': effect_sizes["mean_psi_1"].values,
        'mean_psi_2': effect_sizes["mean_psi_2"].values,
        'significant': significant,
        'delta': min_effect_size,
        'fdr_curve': fdrs,
        'n_significant': n_significant
    })

    return results_df



