import numpy as np 
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import NMF, MiniBatchNMF


def get_NMF(adata_input, K, output_dir, true_juncs_layer="Junction_Counts", true_clusts_layer="Cluster_Counts"):
    
    """Evaluates model performance with different masking strategies and stores the results."""

    # Step 1: Evaluate MiniBatch NMF as a baseline
    W, H = run_minibatch_nmf_baseline(adata_input, n_components=K, true_juncs_layer=true_juncs_layer,true_clusts_layer=true_clusts_layer)

    # Step 2: Calculate silhouette score using NMF-based embeddings (H) and cell type labels
    cell_types = adata_input.obs['cell_type'].values
    NMF_silhouette = silhouette_score(W, cell_types)  # Silhouette score
    
    # Step 3: Plot clustermap of NMF-based embeddings
    plot_clustermap(W, adata_input, output_dir, plot_name='nmf_clustermap.png')

    # Step 4: UMAP projection for dimensionality reduction
    plot_umap(W, adata_input, output_dir, plot_name='nmf_umap.png', silhouette=NMF_silhouette)

    # Return the silhouette score for evaluation
    return NMF_silhouette


def run_minibatch_nmf_baseline(adata, mask=None, n_components=10, batch_size=512, max_iter=200, random_state=42, true_juncs_layer="Original_Junction_Counts", true_clusts_layer="Original_Cluster_Counts"):
    '''
    Run NMF (Non-negative Matrix Factorization) on the PSI values derived from the masked AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with masked junction and cluster counts.
    mask : torch.Tensor or np.ndarray
        Mask indicating the positions to evaluate (True for masked, False for unmasked).
    n_components : int, optional
        Number of components for the NMF model.
    max_iter : int, optional
        Maximum number of iterations for the NMF solver.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    nmf_model : NMF
        The fitted NMF model.
    W : np.ndarray
        The NMF basis matrix (similar to assignment in the factor model).
    H : np.ndarray
        The NMF coefficient matrix (similar to psi in the factor model).
    '''

    # Extract junction counts and cluster counts
    junction_counts = adata.layers[true_juncs_layer].toarray()  # Convert sparse to dense if necessary
    cluster_counts = adata.layers[true_clusts_layer].toarray()
    
    # Compute PSI (junction counts / cluster counts), avoiding division by zero
    psi_matrix = np.divide(junction_counts, cluster_counts, out=np.zeros_like(junction_counts, dtype=float), where=(cluster_counts != 0))
    minibatch_nmf_model = MiniBatchNMF(n_components=n_components, batch_size=batch_size, max_iter=max_iter, random_state=random_state)

    if mask == None:
        print(f"No masking here!")
        # Fit NMF model to the imputed PSI matrix
        W = minibatch_nmf_model.fit_transform(psi_matrix)  # Basis matrix (similar to assignment)
        H = minibatch_nmf_model.components_  # Coefficient matrix (similar to psi)

    else: 
        # Mask to cpu 
        mask_gen = np.array(mask.cpu())
        # Apply mask (if necessary) - this depends on how you want to mask the data
        masked_psi = psi_matrix * (1 - mask_gen)
        # Fit NMF model to the imputed PSI matrix
        # Fit the MiniBatch NMF model (factorizes into W and H matrices)
        W = minibatch_nmf_model.fit_transform(masked_psi)  # Basis matrix (similar to assignment)
        H = minibatch_nmf_model.components_  # Coefficient matrix (similar to psi)  

    # Evaluate NMF predictions
    print(f"Finished fitting NMF with {n_components} components.")
    return W, H