import numpy as np 
import pandas as pd
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

def sparse_sum(x, dim):
    """
    Compute the sum of a sparse matrix along a specified dimension and return a squeezed array.

    Parameters:
    x (spmatrix): A sparse matrix whose elements are to be summed.
    dim (int): The dimension along which the sum is computed. For example, `dim=0` sums along the rows, 
               while `dim=1` sums along the columns.

    Returns:
    ndarray: The resulting dense array with the sums, with single-dimensional entries removed from its shape.
    """
    return np.squeeze(np.asarray(x.sum(dim)))

def max_min_sampling(diffmap_components, n_waypoints, num_components=10, seed=42):

    # Set a random seed for reproducibility
    np.random.seed(seed)

    # Use only the first `num_components` diffusion components
    X_diff = diffmap_components[:, :num_components]
    
    # Step 1: Randomly initialize the first waypoint
    first_waypoint = np.random.choice(X_diff.shape[0], 1)[0]
    waypoints = [int(first_waypoint)]
    
    # Step 2: Initialize distance array to track the minimum distance from each cell to the nearest waypoint
    distances = np.linalg.norm(X_diff - X_diff[first_waypoint, :], axis=1)
    
    # Step 3: Iteratively add waypoints
    for _ in range(1, n_waypoints):
        # Find the cell with the maximum distance to the nearest waypoint
        next_waypoint = np.argmax(distances)
        waypoints.append(next_waypoint)
        
        # Update the minimum distances for all cells based on the new waypoint
        new_distances = np.linalg.norm(X_diff - X_diff[next_waypoint, :], axis=1)
        distances = np.minimum(distances, new_distances)
    
    return np.array(waypoints)

def initialize_psi_phi_with_random_non_zero(rho_hat, metacell_dict, epsilon=0.01):
    """
    Initialize psi and phi using waypoints based on observed rho_hat, choosing random non-zero values for each junction.

    Parameters:
    rho_hat (ndarray): Observed probabilities (n_cells, n_junctions).
    metacell_dict (dict): Dictionary of metacells with waypoints as keys and cell indices as values.
    epsilon (float): Small value to use as fallback if all junction values are zero.

    Returns:
    psi (ndarray): Initialized psi values (n_junctions, n_metacells).
    phi (ndarray): Initialized phi values (n_cells, n_metacells) with each row summing to 1.
    """

    # Step 1: Convert rho_hat to dense if it's sparse
    if hasattr(rho_hat, "toarray"):
        rho_hat = rho_hat.toarray()

    n_junctions = rho_hat.shape[1]  # Number of junctions
    n_metacells = len(metacell_dict)  # Number of metacells

    # Initialize psi with zeros (n_junctions, n_metacells)
    psi = np.zeros((n_junctions, n_metacells))

    # Function to select a random non-zero value from the array of cells for a given junction
    def random_non_zero(arr, epsilon):
        non_zero_vals = arr[arr > 0]  # Find non-zero values
        if len(non_zero_vals) == 0:   # If no non-zero values, return epsilon
            return epsilon
        return np.random.choice(non_zero_vals)  # Randomly choose one of the non-zero values

    # Iterate over each metacell and initialize psi for each junction in that metacell
    for i, (metacell, cell_indices) in tqdm(enumerate(metacell_dict.items())):
        junction_matrix = rho_hat[cell_indices, :]  # Get the junction data for the cells in the metacell

        # Apply random_non_zero for each junction across the metacell and assign the result to psi
        psi[:, i] = np.apply_along_axis(random_non_zero, 0, junction_matrix, epsilon)

    # Step 3: Solve for phi (n_cells, n_metacells) by solving the equation rho_hat = Phi * psi
    # Use least squares to solve for phi
    phi = np.linalg.lstsq(psi, rho_hat.T, rcond=None)[0].T  # (n_cells, n_metacells)

    # Step 4: Clip phi values between epsilon and 1 - epsilon
    phi = np.clip(phi, epsilon, 1 - epsilon)

    # Step 5: Normalize phi such that for each cell, the proportions sum to 1
    phi /= phi.sum(axis=1, keepdims=True)

    return psi, phi

def assign_nearest_cells(waypoints, diffmap_components, num_nearest=5):
    X_diff = diffmap_components
    assigned_cells = set()  # To keep track of already assigned cells
    metacell_dict = {}  # Dictionary to store metacell groupings

    for i, waypoint in enumerate(waypoints):
        # Compute distances from the waypoint to all cells
        distances = np.linalg.norm(X_diff - X_diff[waypoint, :], axis=1)
        
        # Sort cells by distance, and select the nearest ones excluding already assigned cells
        sorted_indices = np.argsort(distances)
        nearest_cells = [idx for idx in sorted_indices if idx not in assigned_cells][:num_nearest]
        
        # Assign the cells to the current metacell
        metacell_dict[f"metacell_{i}"] = nearest_cells
        
        # Add these cells to the set of assigned cells
        assigned_cells.update(nearest_cells)
    
    return metacell_dict

def generate_initializations(rho_hat, waypoints_dict, metacell_dicts, epsilon=0.01):
    """
    Generate multiple psi and phi initializations based on different waypoint sets and metacell dictionaries.

    Parameters:
    rho_hat (ndarray): Observed probabilities (n_cells, n_junctions).
    waypoints_dict (dict): Dictionary with keys as waypoint sizes and values as waypoint sets.
    metacell_dicts (dict): Dictionary with keys as waypoint sizes and values as metacell dictionaries.
    epsilon (float): Small value to use as fallback if all junction values are zero.

    Returns:
    psi_list (list): List of psi initializations.
    phi_list (list): List of phi initializations.
    """

    psi_list = []  # To store psi initializations
    phi_list = []  # To store phi initializations

    # Iterate over each set of waypoints and corresponding metacell dictionary
    for n_waypoints in waypoints_dict.keys():
        
        print(f"Finding {n_waypoints} waypoints from the diffusion components!")

        waypoints = waypoints_dict[n_waypoints]
        metacell_dict = metacell_dicts[n_waypoints]

        # Initialize psi and phi for the current metacell dictionary
        psi, phi = initialize_psi_phi_with_random_non_zero(rho_hat, metacell_dict, epsilon)

        # Store the initialization
        psi_list.append(psi)
        phi_list.append(phi)

    return psi_list, phi_list

# Function to transform the junction_id and Cluster to the desired format
def transform_row(row):
    junction_id = row['junction_id']
    cluster = row['Cluster']
    
    # Split the junction_id into chromosome, start, end, and strand
    chrom, start, end, strand = junction_id.split('_')
    
    # Format as required
    formatted = f"{chrom[3:]}:{start}:{end}:clu_{cluster}_{strand}"
    
    return formatted

def calculate_centered_psi(junction_counts, cluster_counts, rho=0.1):
    """
    Calculates centered PSI values and other related matrices based on the input junction counts and cluster counts.
    
    Weighting is used to:
    - Account for variability in junction counts using the beta-binomial variance model.
    - Adjust the PSI values based on the reliability of the cluster (or junction) counts.

    Parameters:
    - junction_counts: A sparse matrix (COO format) with junction counts.
    - cluster_counts: A sparse matrix (COO format) with cluster counts (ATSE counts).
    - rho: A float value (default: 0.1) representing the overdispersion parameter for the beta-binomial variance model.
    
    Returns:
    - Y_sparse: A sparse matrix with centered PSI values.
    - psi: A sparse matrix with calculated PSI values.
    - w_psi: A sparse matrix with weighted PSI values.
    - junc_means: A dense array with the mean junction usage ratios.
    """
    
    # Step 1: Calculate PSI (junction usage ratios) only for non-zero cluster counts
    non_zero_mask = cluster_counts.data != 0
    junc_ratio_data = junction_counts.data[non_zero_mask] / cluster_counts.data[non_zero_mask]

    psi = coo_matrix((junc_ratio_data, (junction_counts.row[non_zero_mask], junction_counts.col[non_zero_mask])),
                     shape=junction_counts.shape)  # Create a sparse matrix with calculated PSI values

    # Step 2: Calculate observation weights (w) using the beta-binomial variance model
    w = junction_counts.copy()  # Observation weights are based on junction counts
    w.data = cluster_counts.data / (1. + (cluster_counts.data - 1) * rho)  # Beta-binomial variance model

    # Step 3: Calculate weighted PSI (w_psi)
    w_psi = w.copy()  # Copy weight matrix
    w_psi.data *= psi.data  # Multiply weights by PSI data

    # Step 4: Calculate mean junction usage ratios (junc_means)
    junc_means = sparse_sum(w_psi, 0) / sparse_sum(w, 0)  # Weighted mean junction usage ratio per junction

    # Step 5: Center Y by subtracting the mean junction usage ratio from each PSI value
    Y_data = psi.data - junc_means[psi.col]  # Center PSI values

    # Step 6: Create a sparse matrix for Y_data using the row, col indices of the original psi matrix
    Y_sparse = coo_matrix((Y_data, (psi.row, psi.col)), shape=psi.shape)
    
    return Y_sparse

def plot_PCA_with_waypoints(adata, waypoints_dict, color_by='tissue', n_waypoints=20, 
                            save_plot=False,
                            waypoint_color='red', first_waypoint_color='blue', size=10):
    """
    Plots PCA with cells colored by a specified annotation (e.g., 'tissue', 'cell_type') 
    and overlays waypoints on top.
    """

    # Check if waypoints exist
    waypoints = waypoints_dict.get(n_waypoints)
    if waypoints is None:
        raise ValueError(f"No waypoints found for {n_waypoints}. Please check your waypoints_dict.")
    
    # Extract PCA coordinates
    pca_coords = adata.obsm['X_pca']
    waypoint_coords_pca = pca_coords[waypoints]

    # Create scatter plot
    sc.pl.pca(adata, color=color_by, size=size, show=False)  # Standard PCA plot

    # Overlay waypoints
    plt.scatter(waypoint_coords_pca[1:, 0], waypoint_coords_pca[1:, 1], s=40, 
                color=waypoint_color, edgecolors='black', label=f'{n_waypoints} Waypoints')

    plt.scatter(waypoint_coords_pca[0, 0], waypoint_coords_pca[0, 1], s=80, 
                color=first_waypoint_color, edgecolors='black', label='First Waypoint')

    # Add legend
    plt.legend()
    plt.show()

    # Save and show plot
    if save_plot:
        filename = f"PCA_with_waypoints_{n_waypoints}_waypoints.pdf"
        plt.savefig(filename, format='pdf')
    