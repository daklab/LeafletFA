import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist
import torch 

### Implementing Diffusion Components and Max-Min Sampling:

## Step 1: Compute Diffusion Components
adata = sc.datasets.pbmc3k()  # Replace actual data

# Preprocessing: Normalize and find highly variable genes
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat')
adata = adata[:, adata.var.highly_variable]

# Calculate the diffusion map
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
sc.tl.diffmap(adata)

# The diffusion components are now stored in adata.obsm['X_diffmap']
diffmap_components = adata.obsm['X_diffmap']

## Step 2: Perform Max-Min Sampling
# With the diffusion components (diffmap_components), perform Max-Min 
# sampling to spread out waypoints across the data space. The goal is to iteratively add 
# waypoints, where each new waypoint maximizes the minimum distance to the current set of waypoints.

def max_min_sampling(diffmap_components, n_waypoints, num_components=3):
    # Use only the first `num_components` diffusion components
    X_diff = diffmap_components[:, :num_components]
    
    # Step 1: Randomly initialize the first waypoint
    first_waypoint = np.random.choice(X_diff.shape[0], 1)
    waypoints = [first_waypoint]
    
    # Step 2: Iteratively add waypoints
    for _ in range(1, n_waypoints):
        distances = []
        for cell in range(X_diff.shape[0]):
            # Compute distance from this cell to the closest waypoint
            min_distance = min(
                np.linalg.norm(X_diff[cell, :] - X_diff[wp, :])
                for wp in waypoints
            )
            distances.append(min_distance)
        
        # Step 3: Select the point with the maximum of these minimum distances
        next_waypoint = np.argmax(distances)
        waypoints.append(next_waypoint)

    return waypoints

# Example usage
n_waypoints = 10  # Number of waypoints you want
waypoints = max_min_sampling(diffmap_components, n_waypoints)

## Step 3: Use Waypoints for Initialization

# Assign cells near each waypoint to the corresponding latent factor. You can 
# initialize the assign matrix such that cells close to a waypoint have higher probabilities of belonging to that factor.

def initialize_assign_via_waypoints(waypoints, diffmap_components, K):
    # Get the coordinates of waypoints in diffusion space
    waypoint_coords = diffmap_components[waypoints, :]

    # Calculate distances from each cell to the waypoints
    distances = cdist(diffmap_components, waypoint_coords)

    # Invert distances to create higher probability for closer waypoints
    inv_distances = 1 / (distances + 1e-6)  # Avoid division by zero

    # Normalize the distances to get a probability distribution
    assign_init = inv_distances / inv_distances.sum(axis=1, keepdims=True)

    # Assign to K clusters
    return assign_init

# Example usage
assign_init = initialize_assign_via_waypoints(waypoints, diffmap_components, K=10)
assign_init = torch.tensor(assign_init, dtype=torch.float)

def initialize_psi_via_waypoints(adata, waypoints, K):
    # Extract junction usage for waypoint cells
    waypoint_data = adata.X[waypoints, :]  # Assuming adata.X contains junction counts

    # Initialize psi by averaging junction usage of the cells close to each waypoint
    psi_init = np.zeros((K, waypoint_data.shape[1]))

    for k, wp in enumerate(waypoints):
        psi_init[k, :] = waypoint_data[k, :]

    # Normalize psi to be valid probabilities (0 to 1)
    psi_init = psi_init / psi_init.sum(axis=1, keepdims=True)
    
    return torch.tensor(psi_init, dtype=torch.float)

# Example usage
psi_init = initialize_psi_via_waypoints(adata, waypoints, K=10)
