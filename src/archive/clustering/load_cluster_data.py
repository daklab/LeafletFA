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

def make_torch_adata(adata, cluster_layer="Cluster_Counts", junction_layer="Junction_Counts", **float_type):
    device = float_type["device"]
        
    # Convert the row and col of the sparse matrix to numpy arrays first, then to tensors
    cell_index_array = np.array(adata.layers[junction_layer].row)
    junc_index_array = np.array(adata.layers[junction_layer].col)
    
    cell_index_tensor = torch.tensor(cell_index_array, dtype=torch.int64, device=device)
    junc_index_tensor = torch.tensor(junc_index_array, dtype=torch.int64, device=device)
    
    print(f"Unique cell indices: {len(cell_index_tensor.unique())}")

    # Convert the data of the sparse matrix directly to a numpy array, then to a tensor
    ycount_array = np.array(adata.layers[junction_layer].data)  # Non-zero junction counts
    
    # Create ycount tensor directly (already non-zero by construction)
    ycount = torch.tensor(ycount_array, **float_type)

    # Convert the row-column pairs of junction_layer to a set for fast lookup
    junction_pairs = set(zip(cell_index_array, junc_index_array))

    # Extract the cluster_layer COO matrix (coo2)
    coo2 = adata.layers[cluster_layer]

    # Create a mask by checking if the row-col pairs in coo2 exist in junction_pairs
    mask = np.array([(r, c) in junction_pairs for r, c in zip(coo2.row, coo2.col)])
    mask = np.array([(r, c) in junction_pairs for r, c in zip(coo2.row, coo2.col)])

    # Subset coo2's data, rows, and cols based on the mask
    subset_data = coo2.data[mask]

    # Ensure that subset_data aligns with the indices in ycount
    tcount = torch.tensor(subset_data, **float_type)

    # Construct sparse matrices using coo format and convert to CSR for efficiency
    ycount_lookup = torch.sparse_coo_tensor(
        indices=torch.stack([cell_index_tensor, junc_index_tensor]), 
        values=ycount,
        size=(len(adata.obs), len(adata.var))
    ).to_sparse_csr()

    # Transpose the sparse matrix by swapping indices
    ycount_lookup_T = torch.sparse_coo_tensor(
        indices=torch.stack([junc_index_tensor, cell_index_tensor]), 
        values=ycount,
        size=(len(adata.var), len(adata.obs))
    ).to_sparse_csr()

    tcount_lookup = torch.sparse_coo_tensor(
        indices=torch.stack([cell_index_tensor, 
                             junc_index_tensor]),
        values=tcount,
        size=(len(adata.obs), len(adata.var))
    ).to_sparse_csr()

    tcount_lookup_T = torch.sparse_coo_tensor(
        indices=torch.stack([junc_index_tensor, 
                             cell_index_tensor]),
        values=tcount,
        size=(len(adata.var), len(adata.obs))
    ).to_sparse_csr()

    my_data = IndexCountTensor(ycount_lookup, tcount_lookup, ycount_lookup_T, tcount_lookup_T)
    return cell_index_tensor, junc_index_tensor, my_data

        
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

