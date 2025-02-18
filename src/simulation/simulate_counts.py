import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os
from tqdm import tqdm
import torch 
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pdb
import scipy.sparse as sp

def preprocess_adata(adata, cell_type_column, cluster_layer="Cluster_Counts"):
    """
    Preprocesses the AnnData object by filtering unevenly distributed splicing events (ATSEs).
    
    If a cell type column is specified, the function filters out junctions that have
    low expression across the cell types.
    """
    if cell_type_column is not None:
        
        print(f"Filtering ATSEs to remove those very unevenly distributed across cell types!")
        cluster_counts = adata.layers[cluster_layer]
        cell_types = adata.obs[cell_type_column].values
        unique_cell_types = np.unique(cell_types)
        unique_clusters = adata.var_names

        expression_counts = pd.DataFrame(0, index=unique_cell_types, columns=unique_clusters)
        for cell_type in tqdm(unique_cell_types):
            cells_in_type = (cell_types == cell_type)
            counts_in_type = cluster_counts.toarray()[cells_in_type, :].sum(axis=0)
            expression_counts.loc[cell_type] = counts_in_type

        non_zero_counts = (expression_counts >= 5).sum(axis=0)
        threshold = len(expression_counts) * 0.2
        filtered_clusters = non_zero_counts[non_zero_counts > threshold].index

        filtered_expression_counts = expression_counts[filtered_clusters]
        juncs_keep = list(filtered_expression_counts)
        adata_filtered = adata[:, juncs_keep].copy()
        adata_filtered.var['junction_id_index'] = np.arange(adata_filtered.n_vars)
        adata_filtered.var_names = adata_filtered.var['junction_id_index'].astype(str)
        return adata_filtered
    else:
        print("No cell type column specified. Skipping filtering of ATSEs.")
        return adata.copy()
    
# write function that takes in Cluster name 
def check_SS_cluster(juncs_c):
        
    # keep only rows where either start or end appear twice
    start_dup = juncs_c[juncs_c.duplicated(subset=['start'], keep=False)]
    end_dup = juncs_c[juncs_c.duplicated(subset=['end'], keep=False)]
    juncs_c = juncs_c[(juncs_c["start"].isin(start_dup["start"])) | (juncs_c["end"].isin(end_dup["end"]))]

    # if num rows in juncs_c is 3 then return cluster name 
    if len(juncs_c == 3):       
        cluster_name = juncs_c["event_id"].iloc[0]
        return cluster_name
    else:
        pass

def simulate_junc_counts(adata_input, psi_prior_shape1=0.5, psi_prior_shape2=0.5, proportion_negative=0.5):
    
    """
    Simulate junction counts while keeping the cluster counts of observed data.
    Ensure that simulated junction counts have the same nnz as cluster counts.
    
    Parameters
    ----------
    adata_input : AnnData
        The input AnnData object.
    proportion_negative : float
        Proportion of negative ASEs to simulate.
    psi_prior_shape1 : float
        Shape parameter for the Beta distribution for sampling PSI.
    psi_prior_shape2 : float
        Shape parameter for the Beta distribution for sampling PSI.
    
    Returns
    -------
    sim_junc_counts : sparse matrix
        Simulated junction counts matrix that matches nnz of cluster counts.
    cell_type_psi_df : pd.DataFrame
        DataFrame containing cell-type-specific PSI values for each junction.
    """
    
    # Get the categorical values and convert to numpy array of codes
    cell_type_labels = adata_input.obs["cell_type"].cat.codes.to_numpy()
    K = len(adata_input.obs["cell_type"].cat.categories)
    print(f"The proportion of negative ASEs to set is: {proportion_negative}")

    # Get the cluster counts matrix
    cluster_counts = adata_input.layers["cell_by_cluster_matrix"]
    N, P = cluster_counts.shape  # number of cells, number of junctions
    
    print("The number of cell types is:", K)
    print("The number of cells is:", N)
    print("The number of junctions is:", P)

    # Ensure cluster counts are in COO format
    cluster_counts_coo = coo_matrix(cluster_counts)
    
    # Number of intron clusters 
    num_clusters = len(adata_input.var.event_id.unique())

    # Determine the number of negatives and positives
    num_negative = int(proportion_negative * num_clusters)
    num_positive = num_clusters - num_negative

    # Create the labels array
    cluster_labels = np.array(([0] * num_negative) + ([1] * num_positive))

    # Shuffle the labels to randomize their assignment
    np.random.shuffle(cluster_labels)

    # Count occurrences of each label
    count_negatives = np.count_nonzero(cluster_labels == 0)
    count_positives = np.count_nonzero(cluster_labels == 1)

    # Print the counts
    print("Number of negative labels (0):", count_negatives)
    print("Number of positive labels (1):", count_positives)

    # Make a mapping of Cluster ID to cluster_labels
    cluster_labels_dict = dict(zip(adata_input.var.event_id.unique(), cluster_labels))
    
    # initiate empty dataframe cell_type_psi_df to which we will append the simulated PSI values for each junction in each cell type
    cell_type_psi_df = pd.DataFrame()

    for clust in tqdm(adata_input.var.event_id.unique()):
        
        clust_label = cluster_labels_dict[clust]

        # Get junctions in cluster and order them by start and end 
        juncs_c = adata_input.var[adata_input.var["event_id"] == clust].sort_values(by=['start', 'end'])

        # Ensure exactly 3 junctions are present
        if len(juncs_c) != 3:
            continue  # Skip clusters that do not have exactly 3 junctions

        # Assign J1, J2, and J3 to junctions where J1+J2 correspond to exon inclusion and J3 corresponds to exon skipping
        juncs_c["junction"] = ["J1", "J3", "J2"] 

        # Sample PSI values (by default shape parameters are set to the same values for all junctions regardless of cell state...)
        probs = torch.distributions.beta.Beta(psi_prior_shape1, psi_prior_shape2).sample([3, K])
    
        if clust_label == 0: 
            # Make PSI values similar across cell types for negative control
            J3_prob = probs[1,0] # Just take sampled J3 probability from the first cell state 
            probs[1,] = J3_prob  # Same value across all cell types
            probs[0,] = (1 - J3_prob) / 2
            probs[2,] = (1 - J3_prob) / 2
            sample_label = "negative"

        elif clust_label == 1:
            # For positive cases, split cell types into two groups
            # Half will favor high PSI, half will favor low PSI
            num_high = K // 2  # number of cell types that will have high PSI
            num_low = K - num_high  # remaining will have low PSI

            # Sample PSIs from different distributions for each group
            J3_probs_low = torch.distributions.beta.Beta(1, 3).sample([num_low])  # favor low PSI
            J3_probs_high = torch.distributions.beta.Beta(3, 1).sample([num_high])  # favor high PSI
    
            # Combine into one vector of length K
            J3_probs = torch.cat([J3_probs_low, J3_probs_high])

            # Randomly shuffle to avoid having all high/low PSIs grouped together
            J3_probs = J3_probs[torch.randperm(K)]

            # Set the probabilities
            probs[1,] = J3_probs  # J3 probabilities across K cell types some will be low and some will be high
            probs[0,] = (1 - J3_probs) / 2  # J1 probabilities
            probs[2,] = (1 - J3_probs) / 2  # J2 probabilities
            sample_label = "positive"

        probs_df = pd.DataFrame(probs.cpu().numpy())  
        probs_df["junction_id_index"] = juncs_c.index
        probs_df["junction_id"] = juncs_c["junction_id"].values
        probs_df["sample_label"] = sample_label
        probs_df["event_id"] = juncs_c["event_id"].values[0]
        cell_type_psi_df = pd.concat([cell_type_psi_df, probs_df])

    cell_type_psi_df.junction_id_index = cell_type_psi_df.junction_id_index.astype(int)
    cell_type_psi_df = cell_type_psi_df.sort_values(by=['junction_id_index'])

    # Assert that 'junction_id_index' in 'cell_type_psi_df' matches the index in 'adata_input.var'
    assert (cell_type_psi_df['junction_id_index'].values == adata_input.var.index.values.astype(int)).all(), \
        "Mismatch between 'junction_id_index' in 'cell_type_psi_df' and index in 'adata_input.var'."

    print("Assertion passed: 'junction_id_index' matches the index in 'adata_input.var'.")

    # Keep just the first K columns of cell_type_psi_df
    cols_keep = cell_type_psi_df.columns[:K]
    cell_type_psi = torch.tensor(cell_type_psi_df[cols_keep].to_numpy())
    print("Done simulating PSI!")
    
    # Simulate junction counts using a binomial distribution
    sim_junc_counts = cluster_counts_coo.copy() 
    
    # Simulate junction counts using a binomial distribution
    sim_junc_counts.data = torch.distributions.binomial.Binomial(
        total_count=torch.tensor(cluster_counts_coo.data), 
        probs=cell_type_psi[
            cluster_counts_coo.col,  # Junction index
            cell_type_labels[cluster_counts_coo.row]  # Cell index 
        ]
    ).sample().cpu().numpy()  # Ensure compatibility with numpy

    # Group by cluster and normalize junction counts within each cell-cluster pair
    cluster_info = adata_input.var[["event_id", "junction_id_index"]]

    for cluster_id in tqdm(cluster_info["event_id"].unique(), desc="Processing ATSEs"):
        # Get the junction indices for this cluster
        junction_indices = cluster_info[cluster_info["event_id"] == cluster_id]["junction_id_index"].values

        # Create a mask for the current cluster based on the junction indices
        cluster_mask = np.isin(cluster_counts_coo.col, junction_indices)

        # Get the rows (cells) and columns (junctions) for this cluster
        relevant_rows = cluster_counts_coo.row[cluster_mask]
        relevant_cols = cluster_counts_coo.col[cluster_mask]
    
        # Get the simulated junction counts for these cell-cluster combinations
        junction_counts = sim_junc_counts.data[cluster_mask]

        # Compute the new cluster counts by summing the simulated junction counts for each cell
        new_cluster_counts = np.bincount(relevant_rows, weights=junction_counts, minlength=cluster_counts_coo.shape[0])

        # Update the cluster counts (in the sparse matrix) to the new sums
        cluster_counts_coo.data[cluster_mask] = new_cluster_counts[relevant_rows]

    print("Done normalizing junction counts by cluster!")
    print("Done simulating junction counts!")       
        
    return sim_junc_counts, cluster_counts_coo, cell_type_psi_df

def quick_clust_plot(clust, adata_input):

    simple_data_junc = adata_input.var[adata_input.var.event_id == clust]
    junc_indices = simple_data_junc["junction_id_index"].values

    # Get cell specific junction usage ratios 
    junc_data = adata_input[:, junc_indices].layers["junc_ratio"]

    junc_data_dense = junc_data.toarray() if hasattr(junc_data, 'toarray') else junc_data

    junc_df = pd.DataFrame(junc_data_dense, index = adata_input.obs["cell_id_index"], columns=simple_data_junc["junction_id_index"])
    junc_df["cell_id_index"] = junc_df.index 
    junc_df = junc_df.melt(id_vars="cell_id_index" , var_name="junction_id_index", value_name="junc_ratio")

    # Filter out NaN values (these indicate that the junction is not expressed in that cell)
    junc_df = junc_df.dropna(subset=["junc_ratio"])
    junc_df['cell_id_index'] = junc_df['cell_id_index'].astype(adata_input.obs.index.dtype)

    if adata_input.obs["cell_id_index"].dtype != junc_df['cell_id_index'].dtype:
        adata_input.obs["cell_id_index"] = adata_input.obs["cell_id_index"].astype(junc_df['cell_id_index'].dtype)

    junc_df = junc_df.merge(adata_input.obs, on="cell_id_index")
    junc_df = junc_df.merge(adata_input.var)

    print(junc_df[["junction_id_index", "junction_id", "difference", "true_label"]].drop_duplicates())
    print(junc_df.cell_type.value_counts())

    sns.violinplot(data=junc_df, x="junc_ratio", y="cell_type", hue="junction_id_index")
    plt.title(f"{junc_df.true_label[0]}")
    num_nz_cells = junc_df.cell_id_index.nunique()
    print("Number of non-zero cells:", num_nz_cells)
    return junc_df

def simulate_and_prepare_data(adata_input, K, float_type, proportion_negative=0.5, cell_type_column=None):
    
    """Load, filter, and simulate data, returning tensors for the model.
    
    input_file: should be an Anndata object 
    K: number of cell types we want to simulate
    """

    # Group by 'Cluster' and count 'junction_id'
    cluster_junc_counts =  adata_input.var.groupby(["event_id"]).agg({"junction_id": "count"}).reset_index()

    # Filter clusters with exactly 3 junctions
    clusts_keep = cluster_junc_counts[cluster_junc_counts["junction_id"] == 3]

    # Filter 'var' in the AnnData object based on the clusters to keep
    adata_input = adata_input[:, adata_input.var["event_id"].isin(clusts_keep["event_id"])].copy()

    # Extract 'chr', 'start', and 'end' from 'junction_id' using .loc to avoid SettingWithCopyWarning
    adata_input.var.loc[:, "chr"] = adata_input.var["junction_id"].str.split("_").str[0]
    adata_input.var.loc[:, "start"] = adata_input.var["junction_id"].str.split("_").str[1]
    adata_input.var.loc[:, "end"] = adata_input.var["junction_id"].str.split("_").str[2]

    print(f"Cluster_Counts nnz: {adata_input.layers['cell_by_cluster_matrix'].count_nonzero()}")
    print(f"Junction_Counts nnz: {adata_input.layers['cell_by_junction_matrix'].count_nonzero()}")

   # Reset the junction_id_index column to maintain consistency
    print(f"The number of unique junctions included in the simulation data is: {len(adata_input.var.junction_id.unique())}")
    print(f"The number of unique clusters included in the simulation data is: {len(adata_input.var.event_id.unique())}")

    # Reset the junction_id_index column to match new reset index order 
    adata_input.var.reset_index(drop=True, inplace=True)
    adata_input.var["junction_id_index"] = range(len(adata_input.var))

    clusters_SS = []

    for cluster, juncs_c in tqdm( adata_input.var.groupby("event_id")):
        result = check_SS_cluster(juncs_c)
        if result:
            clusters_SS.append(result)

    # Subset the data to just clusters that contain exon skipping events 
    adata_input = adata_input[:, adata_input.var["event_id"].isin(clusters_SS)].copy()
    
    # Reset the junction_id_index column after subsetting
    adata_input.var["junction_id_index"] = range(len(adata_input.var))
    adata_input.var.reset_index(drop=True, inplace=True)

    print(f"Cluster_Counts nnz: {adata_input.layers['cell_by_cluster_matrix'].count_nonzero()}")
    print(f"Junction_Counts nnz: {adata_input.layers['cell_by_junction_matrix'].count_nonzero()}")

    # Check if the cell_type_column exists in adata_input
    if cell_type_column in adata_input.obs.columns:
        # Use the existing cell type column to assign dummy variables
        adata_input.obs["cell_type"] = adata_input.obs[cell_type_column]
    else:
        # If the cell_type_column does not exist, randomly assign a synthetic "cell type" to each cell
        adata_input.obs["cell_type"] = np.random.choice(range(K), size=len(adata_input.obs))

    # Randomly assign a synthetic "cell type" to each cell 
    if adata_input.obs["cell_type"].dtype != 'category':
        adata_input.obs["cell_type"] = adata_input.obs["cell_type"].astype("category")

    # Simulate junction counts!!! 
    sim_junc_counts, sim_cluster_counts, cell_type_psi_df = simulate_junc_counts(adata_input, proportion_negative = proportion_negative)
    
    if K == 2:
        # Calculate the absolute difference between two cell types
        cell_type_psi_df["difference"] = np.abs(cell_type_psi_df[0] - cell_type_psi_df[1])
        threshold=0.1
    else:
        # Calculate the standard deviation across all cell types
        cell_type_psi_df["difference"] = cell_type_psi_df.iloc[:, :K].std(axis=1)
        threshold = cell_type_psi_df[cell_type_psi_df["sample_label"] == "positive"]["difference"].quantile(0.05)

    # Relabel clusters based on the calculated threshold
    # Not doing this ATM ! 
    # relabel_clusts = cell_type_psi_df[
    #    (cell_type_psi_df["sample_label"] == "positive") & (cell_type_psi_df["difference"] < threshold)
    #    ].Cluster.unique()
    
    # Create the 'true_label' column, initially copying from 'sample_label'
    cell_type_psi_df["true_label"] = cell_type_psi_df["sample_label"]
    
    # Update 'true_label' for clusters that should be relabeled to 'negative'
    # cell_type_psi_df.loc[cell_type_psi_df["Cluster"].isin(relabel_clusts), "true_label"] = "negative"
    cell_type_psi_df.sort_values(by=["junction_id_index"], inplace=True)

    # Print the value counts for 'true_label' and 'sample_label' for comparison
    print("True label counts:\n", cell_type_psi_df["true_label"].value_counts())
    print("Sample label counts:\n", cell_type_psi_df["sample_label"].value_counts())

    # Remove Junction_Counts and remake it using simulated junction counts
    del adata_input.layers["cell_by_junction_matrix"]
    del adata_input.layers["cell_by_cluster_matrix"]

    adata_input.layers["cell_by_junction_matrix"]  = coo_matrix(sim_junc_counts)
    adata_input.layers["cell_by_cluster_matrix"]  = coo_matrix(sim_cluster_counts)

    adata_input.layers["junc_ratio"] = adata_input.layers["cell_by_junction_matrix"] / adata_input.layers["cell_by_cluster_matrix"]
    
    cell_type_psi_df.reset_index(inplace=True)

    print(f"Cluster_Counts nnz: {adata_input.layers['cell_by_cluster_matrix'].count_nonzero()}")
    print(f"Junction_Counts nnz: {adata_input.layers['cell_by_junction_matrix'].count_nonzero()}")

    # Add junction and cluster info to adata.var 
    common_columns = adata_input.var.columns.intersection(cell_type_psi_df.columns)
    
    adata_input.var = pd.merge(
        adata_input.var,
        cell_type_psi_df,
        on=common_columns.tolist())    

    # Make input for model 
    device = float_type["device"]

    # Extract cell indices and junction indices from the sparse junction layer
    cell_index_array = np.array(adata_input.layers["cell_by_junction_matrix"].row)
    junc_index_array = np.array(adata_input.layers["cell_by_junction_matrix"].col)

    # Convert the row and col of the sparse matrix to torch tensors
    cell_index_tensor = torch.tensor(cell_index_array, dtype=torch.int32, device=device)
    junc_index_tensor = torch.tensor(junc_index_array, dtype=torch.int32, device=device)

    # Convert the data of the sparse matrix to torch tensor
    ycount_array = np.array(adata_input.layers["cell_by_junction_matrix"].data)
    ycount = torch.tensor(ycount_array, **float_type)

    # Create the ycount sparse matrix (Junction counts)
    ycount_lookup = torch.sparse_coo_tensor(
            indices=torch.stack([cell_index_tensor, junc_index_tensor]), 
            values=ycount,
            size=(len(adata_input.obs), len(adata_input.var))
            ).to_sparse_csr()

    # Extract the cluster layer (total counts)
    coo2 = adata_input.layers["cell_by_cluster_matrix"]
    total_counts_tensor = torch.tensor(coo2.data, **float_type)

    # Create the tcount sparse matrix (Total counts)
    tcount_lookup = torch.sparse_coo_tensor(
            indices=torch.stack([cell_index_tensor, junc_index_tensor]), 
            values=total_counts_tensor,
            size=(len(adata_input.obs), len(adata_input.var))
        ).to_sparse_csr()

    # Convert sparse matrices to COO format, ensure they are on the GPU, and follow the specified float type
    full_y_tensor = ycount_lookup.to_sparse_coo()
    full_total_counts_tensor = tcount_lookup.to_sparse_coo()

    print("Data successfully simulated and prepared!")
    return full_y_tensor, full_total_counts_tensor, adata_input, cell_type_psi_df


    