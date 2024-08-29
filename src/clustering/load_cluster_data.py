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

def make_torch_adata(adata, **float_type):
    device = float_type["device"]

    # Initiate tensors for cell and junction indices
    cell_index_tensor = torch.tensor(adata.layers["Junction_Counts"].row, dtype=torch.int64, device=device)
    junc_index_tensor = torch.tensor(adata.layers["Junction_Counts"].col, dtype=torch.int64, device=device)
    print(len(cell_index_tensor.unique()))
    
    # Save sparse counts should be just non-zero cluster counts values 
    ycount = torch.tensor(adata.layers["Junction_Counts"].data, **float_type)
    tcount = torch.tensor(adata.layers["Cluster_Counts"].data, **float_type)

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

