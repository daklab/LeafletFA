import torch
import torch.distributions as distributions

import pandas as pd
import numpy as np
import copy
torch.cuda.empty_cache()

from dataclasses import dataclass
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import argparse
from scipy.stats import binom
from tqdm import tqdm
import sklearn.cluster
from scipy.stats import binom

@dataclass
class IndexCountTensor():
    ycount_lookup: torch.Tensor
    tcount_lookup: torch.Tensor 
    ycount_lookup_T: torch.Tensor
    tcount_lookup_T: torch.Tensor 
        
def make_torch_data(final_data, **float_type):

    device = float_type["device"]
            
    # note these are staying on the CPU! 
    print("The number of cells going into training data is:")
    print(len(final_data.cell_id_index.unique()))

    # initiate instance of data class containing junction and cluster indices for non-zero clusters 
    junc_index_tensor = torch.tensor(final_data['junction_id_index'].values, dtype=torch.int64, device=device)
    cell_index_tensor = torch.tensor(final_data['cell_id_index'].values, dtype=torch.int64, device=device)
    print(len(cell_index_tensor.unique()))
    
    ycount = torch.tensor(final_data.junc_count.values, **float_type) 
    tcount = torch.tensor(final_data.clustminjunc.values, **float_type)

    M = len(cell_index_tensor)

    ycount_lookup = torch.sparse_coo_tensor(
        torch.stack([cell_index_tensor, junc_index_tensor]), 
        ycount).to_sparse_csr()

    ycount_lookup_T = torch.sparse_coo_tensor( # this is a hack since I can't figure out tranposing sparse matrices :( maybe will work in newer pytorch? 
        torch.stack([junc_index_tensor, cell_index_tensor]), 
        ycount).to_sparse_csr()

    tcount_lookup = torch.sparse_coo_tensor( 
        torch.stack([cell_index_tensor, junc_index_tensor]), 
        tcount).to_sparse_csr()
    
    tcount_lookup_T = torch.sparse_coo_tensor(
        torch.stack([junc_index_tensor, cell_index_tensor]), 
        tcount).to_sparse_csr()

    my_data = IndexCountTensor(ycount_lookup, tcount_lookup, ycount_lookup_T, tcount_lookup_T)
    
    return cell_index_tensor, junc_index_tensor, my_data


# load data 
def load_cluster_data(input_file=None, input_folder=None, celltypes = None, num_cells_sample = None, 
                      max_intron_count=None, remove_singletons=True, has_genes="no"):

    """
    Load and preprocess cluster data from HDF5 files, either from a single file or a directory of files.
    It filters data based on cell types, samples a specified number of cells, removes outliers based on max intron count,
    and constructs sparse matrices for junction and cluster counts.
    
    Parameters:
    - input_file (str, optional): Path to a single HDF5 file to load.
    - input_folder (str, optional): Path to a folder containing multiple HDF5 files to load and concatenate.
    - celltypes (list of str, optional): List of cell types to include in the analysis.
    - num_cells_sample (int, optional): Number of cells to randomly sample from the dataset.
    - max_intron_count (int, optional): Maximum allowable intron count for filtering outliers.
    - has_genes (str, optional): Indicates whether gene IDs are included in the dataset ('yes' or 'no').
    
    Returns:
    - final_data (DataFrame): Processed data including junction and cluster counts, cell types, and ratios.
    - coo_counts_sparse (csr_matrix): Sparse matrix of junction counts.
    - coo_cluster_sparse (csr_matrix): Sparse matrix of cluster counts.
    - cell_ids_conversion (DataFrame): Mapping of cell_id_index to cell_id and cell_type.
    - junction_ids_conversion (DataFrame): Mapping of junction_id_index to junction_id, and optionally gene_id.
    """

    # Load data
    if input_file:
        summarized_data = pd.read_hdf(input_file, 'df')
    elif input_folder:
        print("Reading in data from folder:", input_folder)
        summarized_data = pd.concat(
            [pd.read_hdf(os.path.join(input_folder, file), 'df') for file in os.listdir(input_folder) if file.endswith(".h5")],
            ignore_index=True
        )
        print("Finished reading in data from folder.")
    else:
        raise ValueError("Either input_file or input_folder must be provided.")

    # Filter by cell types
    if celltypes:
        print("Filtering by cell types:", celltypes)
        summarized_data = summarized_data[summarized_data["cell_type"].isin(celltypes)]
    
    # Sample cells
    if num_cells_sample:
        summarized_data = summarized_data.sample(n=num_cells_sample)

    # Remove singleton clusters
    if remove_singletons:
        print("Removing singleton clusters...")
        cluster_counts = summarized_data.groupby("Cluster")["junction_id"].nunique()
        valid_clusters = cluster_counts[cluster_counts > 1].index
        summarized_data = summarized_data[summarized_data["Cluster"].isin(valid_clusters)]
   
    # Remove outliers based on max intron count
    if max_intron_count:
        print("Filtering clusters with max intron count greater than", max_intron_count)
        summarized_data = summarized_data[summarized_data["Cluster_Counts"] <= max_intron_count]

    # Re-index cell_id and junction_id
    summarized_data["cell_id_index"] = pd.factorize(summarized_data["cell_id"])[0]
    summarized_data["junction_id_index"] = pd.factorize(summarized_data["junction_id"])[0]

    # Prepare conversion tables
    cell_ids_conversion = summarized_data[["cell_id_index", "cell_id", "cell_type"]].drop_duplicates().sort_values("cell_id_index")
    junction_columns = ["junction_id_index", "junction_id", "Cluster"]
    if has_genes == "yes":
        junction_columns.append("gene_id")
    junction_ids_conversion = summarized_data[junction_columns].drop_duplicates().sort_values("junction_id_index")
    
    # Create sparse matrices
    coo = summarized_data[["cell_id_index", "junction_id_index", "junc_count", "Cluster_Counts", "Cluster", "junc_ratio"]]
    coo_counts_sparse = coo_matrix((coo["junc_count"], (coo["cell_id_index"], coo["junction_id_index"])))
    coo_cluster_sparse = coo_matrix((coo["Cluster_Counts"], (coo["cell_id_index"], coo["junction_id_index"])))
    
    # Construct final data
    final_data = pd.DataFrame({
        "cell_id_index": coo_counts_sparse.row,
        "junction_id_index": coo_counts_sparse.col,
        "junc_count": coo_counts_sparse.data,
    })

    final_data = final_data.merge(
        summarized_data[["cell_id_index", "junction_id_index", "junction_id", "Cluster", "Cluster_Counts"]], 
        on=["cell_id_index", "junction_id_index"], 
        how="left"
    )
    
    final_data["clustminjunc"] = final_data["Cluster_Counts"] - final_data["junc_count"]
    final_data["juncratio"] = final_data["junc_count"] / final_data["Cluster_Counts"]
    final_data = final_data.merge(cell_ids_conversion, on="cell_id_index", how="left")
    
    print("Final data prepared with {} junctions and {} cells.".format(
        len(final_data["junction_id_index"].unique()), 
        len(final_data["cell_id_index"].unique())
    ))

    return(final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion)
