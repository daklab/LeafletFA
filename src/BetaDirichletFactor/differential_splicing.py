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

def calculate_silhouette_score(assign_post, cell_types):
    """Calculates silhouette score for the factor assignments."""
    return silhouette_score(assign_post, cell_types)

# TO-DO: INSTEAD OF DOING DS BETWEEN FACTORS, DO DS BETWEEN CELL TYPES
# JUST LOOK AT FACTOR ACTITIES VIA PHI AND PSI FOR PHI[CELL_TYPE, FACTOR] AND PSI[FACTOR, JUNCTION]
# VERSUS PHI[CELL_TYPE, FACTOR] AND PSI[FACTOR, JUNCTION] FOR ALL OTHER CELL TYPES

def compute_junction_effect_size(psi_samples, phi_samples, factor_idx, junction_idx, min_effect_size=0.1):
    """
    Computes the effect size for a given junction across all samples and summarizes statistics.

    Args:
        psi_samples (torch.Tensor): Tensor of shape [S, K, J] representing sample-factor-junction usage.
        phi_samples (torch.Tensor): Tensor of shape [S, C, K] representing sample-cell-factor assignments.
        factor_idx (int): Index of the factor of interest.
        junction_idx (int): Index of the junction of interest.
        min_effect_size (float, optional): Threshold for significance. Default is 0.1.

    Returns:
        dict: Summary statistics of effect size for the given junction.
    """

    # Get the number of factors
    n_factors = phi_samples.shape[2]

    # Indices of all other factors
    other_factor_indices = [k for k in range(n_factors) if k != factor_idx]

    # Extract phi and psi for the factor of interest
    phi_j_factor = phi_samples[:, :, factor_idx]  # Shape: [S, C]
    psi_j_factor = psi_samples[:, factor_idx, junction_idx]  # Shape: [S]

    # Extract phi and psi for all other factors
    phi_j_nonfactor = phi_samples[:, :, other_factor_indices]  # Shape: [S, C, K-1]
    psi_j_nonfactor = psi_samples[:, other_factor_indices, junction_idx]  # Shape: [S, K-1]

    # Compute contributions from the factor of interest
    result_factor = phi_j_factor * psi_j_factor.unsqueeze(1)  # Shape: [S, C]

    # Compute sum of contributions from all other factors
    result_nonfactor = (phi_j_nonfactor * psi_j_nonfactor.unsqueeze(1)).sum(dim=2)  # Shape: [S, C]

    # Save factor and nonfactor mean values
    factor_mean = result_factor.mean()
    nonfactor_mean = result_nonfactor.mean()
    
    # get average cell usgae across samples [C]
    result_factor_avg = result_factor.mean(dim=0)
    result_nonfactor_avg = result_nonfactor.mean(dim=0)

    # Compute cell-wise effect sizes
    # effect_size_per_cell = result_factor - result_nonfactor  # Shape: [S, C]
    
    # This prevents the effect size from being dominated by non-factor contributions.
    effect_size_per_cell = (result_factor - result_nonfactor) / (result_nonfactor + 1e-6)

    # Compute effect size per sample (average across cells)
    effect_size_per_sample = effect_size_per_cell.mean(dim=1)  # Shape: [S]

    # Compute summary statistics across samples
    effect_size_mean = effect_size_per_sample.mean()
    effect_size_var = effect_size_per_sample.var()

    # Compute probability of effect size being greater than the threshold
    prob_greater = (torch.abs(effect_size_per_sample) > min_effect_size).float().mean()
    num_greater = (torch.abs(effect_size_per_sample) > min_effect_size).float().sum()
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
        'factor_mean': factor_mean.item(),
        'nonfactor_mean': nonfactor_mean.item(),
        'result_factor_avg': result_factor_avg.cpu().numpy(),
        'result_nonfactor_avg': result_nonfactor_avg.cpu().numpy()
    }

def compute_junctions_significance(effect_sizes, fdr_threshold, min_effect_size=0.1):
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
    for col in required_cols:
        if col not in effect_sizes.columns:
            raise ValueError(f"Missing column '{col}' in effect_sizes DataFrame.")

    # Extract values
    prob_greater = effect_sizes['prob_greater'].values
    beta = effect_sizes['effect_size'].values
    beta_vars = effect_sizes['effect_size_var'].values
    n_junctions = len(effect_sizes)

    # Sort indices based on decreasing probability
    sorted_idx = np.argsort(-prob_greater)
    sorted_probs = prob_greater[sorted_idx]

    # Compute FDR using cumulative false discovery proportion
    fdrs = np.cumsum(1 - sorted_probs) / (np.arange(len(sorted_probs)) + 1)
    
    # Identify significant junctions based on FDR threshold
    max_discoveries = np.where(fdrs <= fdr_threshold)[0]
    n_significant = len(max_discoveries) if len(max_discoveries) > 0 else 0

    # Initialize boolean array for significance
    significant = np.zeros(n_junctions, dtype=bool)

    # Apply FDR threshold first
    if n_significant > 0:
        significant[sorted_idx[:n_significant]] = True

    # Apply effect size threshold separately
    significant = significant & (np.abs(beta) > min_effect_size)

    # Count number of significant junctions
    n_significant = np.sum(significant)

    print(f"Found {n_significant} significant junctions with effect size > {min_effect_size} at FDR < {fdr_threshold}")

    # Construct results DataFrame
    results_df = pd.DataFrame({
        'factor_idx': effect_sizes['factor_idx'].values,
        'junction_idx': effect_sizes['junction_idx'].values,
        'junction_id_index': effect_sizes['junction_idx'].values,
        'effect_size': beta,
        'abs_effect_size': np.abs(beta),
        'effect_size_var': beta_vars,
        'num_greater': effect_sizes.get('num_greater', np.nan).values,  # Handles missing columns
        'prob_greater': prob_greater,
        'prob_lessoreq': effect_sizes.get('prob_lessoreq', np.nan).values,
        'significant': significant,
        'delta': min_effect_size,
        'factor_mean': effect_sizes.get('factor_mean', np.nan).values,
        'nonfactor_mean': effect_sizes.get('nonfactor_mean', np.nan).values,
        'fdr_curve': fdrs,
        'n_significant': n_significant, 
        'result_factor_avg': effect_sizes['result_factor_avg'].values,
        'result_nonfactor_avg': effect_sizes['result_nonfactor_avg'].values
    })

    return results_df

def analyze_all_factors(
    psi_samples,
    phi_samples,
    top_junctions, 
    min_effect_size: float = None,
    dist_sd: float = 0.5,
    fdr_threshold: float = 0.05
) -> Dict[int, Tuple[Dict[str, np.ndarray], pd.DataFrame]]:
    """
    Analyze differential splicing for all factors.
    Allows selection of top differentially spliced (DS) junctions.

    Returns:
        Dictionary mapping factor index to (results_dict, results_df) tuple.
    """
    n_factors = phi_samples.shape[2]
    results = {}

    print(f"Analyzing differential splicing for {n_factors} factors...")

    for k in range(n_factors):
        print(f"Finding DS junctions in factor {k} vs. all others")
        top_fact_juncs = [] 
        factor_idx = k 
        for j in tqdm(top_junctions):
            results_dict = compute_junction_effect_size(
                psi_samples, phi_samples, factor_idx, j, min_effect_size=min_effect_size
            )
            top_fact_juncs.append(results_dict)

        # Convert to DataFrame
        top_fact_juncs_df = pd.DataFrame(top_fact_juncs)

        # Compute significance
        top_fact_juncs_df_SIG = compute_junctions_significance(top_fact_juncs_df, fdr_threshold, min_effect_size)

        print(f"Done calculations for factor {k}")

        # Store results
        results[k] = (top_fact_juncs, top_fact_juncs_df_SIG)

    return results


# TO-DO: FIX THE CALIBRATION TEST TO WORK WITH THE NEW ANALYSIS FUNCTION
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
