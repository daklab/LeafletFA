import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap
import matplotlib.patches as mpatches
from leafletfa import differential_splicing

try:
    import cost_correlation_assign
    _COST_CORR_AVAILABLE = True
except ImportError:
    _COST_CORR_AVAILABLE = False

def calculate_and_plot_correlations(assign_matrices):
    """Calculates and plots the correlation matrix for assignments."""
    if not _COST_CORR_AVAILABLE:
        raise ImportError("cost_correlation_assign is not available in this environment.")
    corrs, matchings = cost_correlation_assign.compare_assignments(assign_matrices)
    
    plt.figure()
    sns.clustermap(corrs, annot=True)

    # Calculate and report correlation metrics
    avg_corr = average_pairwise_correlation(corrs)
    median_corr = median_pairwise_correlation(corrs)
    min_corr = min_pairwise_correlation(corrs)
    
    report = (f"Assignment Correlation Report:\n"
              f"Average Pairwise Correlation: {avg_corr:.4f}\n"
              f"Median Pairwise Correlation: {median_corr:.4f}\n"
              f"Minimum Pairwise Correlation: {min_corr:.4f}\n")
    
    print(report)
    return avg_corr, median_corr, min_corr

def save_results_dataframe(output_dir, results):
    """Saves a DataFrame with main results of the analysis."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'analysis_results.csv'), index=False)
    print(f"Results saved to {output_dir}/analysis_results.csv")

def average_pairwise_correlation(corrs):
    """Computes the average pairwise correlation."""
    i, j = np.tril_indices_from(corrs, k=-1)
    return np.mean(corrs[i, j])

def median_pairwise_correlation(corrs):
    """Computes the median pairwise correlation."""
    i, j = np.tril_indices_from(corrs, k=-1)
    return np.median(corrs[i, j])

def min_pairwise_correlation(corrs):
    """Computes the minimum pairwise correlation."""
    i, j = np.tril_indices_from(corrs, k=-1)
    return np.min(corrs[i, j])

def plot_umap(latent_space, output_dir, plot_name='umap.png'):
    """Plots UMAP of cell embeddings and saves it."""
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(latent_space)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP Projection of Latent Space")
    return embedding

def calculate_silhouette_score(assign_post, cell_types):
    """Calculates silhouette score for the factor assignments."""
    return silhouette_score(assign_post, cell_types)

def compute_and_plot_albf(psis_mus, psis_loc, psis, pi, K):
    
    # Compute ALBF
    l0 = []
    albf_values = []

    # Reshape psis_mus and psis_loc to be (J, K) instead of (K, J)
    psis_mus = psis_mus.T
    psis_loc = psis_loc.T
    J = psis_mus.shape[0]

    print(f"The first set of mus and locs for  the first junction are {psis_mus[0]} and {psis_loc[0]}")
    print(f"The learned pi vector is {pi}")

    for j in range(J):  # Iterate over all J junctions
        albf, log_pj_h0 = differential_splicing.compute_albf(psis_mus[j], psis_loc[j], pi)
        l0.append(log_pj_h0.item())
        albf_values.append(albf.item())

    l0 = torch.tensor(l0).detach().cpu()
    albf_values = torch.tensor(albf_values).detach().cpu()
    albf_df = pd.DataFrame(albf_values.numpy().flatten(), columns=["ALBF"])
    albf_df["junction_id_index"] = range(albf_df.shape[0])
    
    psis_df = pd.DataFrame(psis.T)
    psis_df["junction_id_index"] = psis_df.index
    psis_df = psis_df.merge(albf_df, on=["junction_id_index"])

    # Add mus and locs to psis_df for every junction to keep raw values     
    for k in range(K):         
        psis_df[f"mu_{k}"] = psis_mus[:, k].cpu().numpy()
        psis_df[f"loc_{k}"] = psis_loc[:, k].cpu().numpy()

    return psis_df