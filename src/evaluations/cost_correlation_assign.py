import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def compare_assignments(assign_matrices):

    """
    Compare the assignments [K by N (# cells) latent variable, vector of cell factor assignment proportions] 
    across multiple initializations using the Hungarian algorithm to find the optimal matching and correlation 
    of factor proportions for each cell.
    """

    num_matrices = len(assign_matrices)
    corrs = np.zeros((num_matrices, num_matrices))
    matchings = {}

    for i in range(num_matrices):
        for j in range(i + 1, num_matrices):

            # Compute correlation matrix between two sets of assignments
            corr_matrix = np.corrcoef(assign_matrices[i].T, assign_matrices[j].T)
            
            # Extract the KxK block that represents the inter-correlations between the assignments
            K = assign_matrices[i].shape[1]  # Number of states
            corr_submatrix = corr_matrix[:K, K:K*2]  # Extracting the cross-correlation part

            # Create cost matrix for maximum correlation assignment (minimizing negative correlation)
            # Cost Matrix: The negative of the correlation matrix is used as the cost matrix because the 
            # linear sum assignment function minimizes the cost. 
            # By negating the correlations, high correlations (close to 1) become low costs (close to -1), which 
            # the algorithm will then minimize, effectively maximizing the correlation.
            cost_matrix = -corr_submatrix

            # Matching: The linear_sum_assignment function from SciPy returns the optimal row and column indices 
            # that minimize the total cost. These indices represent the optimal matching of latent states across the two matrices being compared.

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Compute the optimal correlation based on the assignment
            aligned_j = assign_matrices[j][:, col_ind]
            corrs[i, j] = np.corrcoef(assign_matrices[i].flatten(), aligned_j.flatten())[0, 1]
            corrs[j, i] = corrs[i, j]  # Make the matrix symmetric

            # Store the matched pairs
            matchings[(i, j)] = (row_ind, col_ind)
            print(f"Initialization {i+1} vs Initialization {j+1}: Matched pairs {list(zip(row_ind, col_ind))}")

    return corrs, matchings
