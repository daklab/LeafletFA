import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/utils')
from collections import defaultdict
from scipy.sparse import coo_matrix
import numpy as np
import anndata as ad
import random
import datetime
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

def merge_dictionaries(dict1, dict2):
    """
    Merge two nested dictionaries.
    If a key does not exist in dict1, initialize it before adding.
    """
    for key, subdict in dict2.items():
        if key not in dict1:
            dict1[key] = {}  # Initialize sub-dictionary if it doesn't exist
        for subkey, value in subdict.items():
            if subkey not in dict1[key]:
                dict1[key][subkey] = 0  # Initialize count if it doesn't exist
            dict1[key][subkey] += value

def read_single_junction_file_filt(junc_file, relevant_junction_ids, sequencing_type="single_cell"):
    """
    Read a single junction file and return a dictionary of junctions that match the relevant junction IDs.

    Parameters:
    - junc_file: Path to the junction file.
    - relevant_junction_ids: Set of relevant junction IDs to filter.
    - sequencing_type: Type of sequencing data ('single_cell' or 'bulk').

    Returns:
    - junc_dict: Dictionary containing cell counts and total read scores for relevant junctions.
    """

    # Specify dtypes to reduce memory usage
    dtypes = {0: str, 1: 'int32', 2: 'int32', 3: str, 4: 'int32', 5: str,
              6: 'int32', 7: 'int32', 8: str, 9: 'int32', 10: str, 11: str}
    juncs = pd.read_csv(junc_file, sep="\t", header=None, dtype=dtypes)
    
    # Add column names here:
    col_names = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", 
         "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    if sequencing_type == "single_cell":
        col_names += ["num_cells_wjunc", "cell_readcounts"]
    juncs.columns = col_names 

    # Extract block sizes and clean up junction coordinates
    juncs[['block_add_start', 'block_subtract_end']] = juncs["blockSizes"].str.extract(r'(\d+),(\d+)').astype(int)
    juncs["chromStart"] += juncs['block_add_start']
    juncs["chromEnd"] -= juncs['block_subtract_end']
    
    # Create a junction_id (including strand)
    juncs['junction_id'] = juncs['chrom'] + '_' + juncs['chromStart'].astype(str) + '_' + juncs['chromEnd'].astype(str) + '_' + juncs['strand']

    # Filter the DataFrame based on relevant junction IDs
    juncs = juncs[juncs['junction_id'].isin(relevant_junction_ids)]

    # Dictionary to store cell counts and scores for each junction
    cell_junction_counts = defaultdict(lambda: defaultdict(int))

    # Process each junction record
    for _, row in juncs.iterrows():
        junction_id = row['junction_id']
        
        if sequencing_type == "single_cell":
            # For Smart-seq2, there's only one cell per file
            cell_id = row['cell_readcounts'].split(":")[0]  # Get cell ID
            read_count = row['score']
            cell_junction_counts[cell_id][junction_id] += read_count
        elif sequencing_type == "10x":
            # For 10x, multiple cells and their read counts may be in the 'cell_readcounts' column
            for cell_info in row['cell_readcounts'].split(','):
                cell_id, read_count = cell_info.split(":")
                read_count = int(read_count)
                cell_junction_counts[cell_id][junction_id] += read_count

    return cell_junction_counts

def read_and_process_file(junc_file, relevant_junction_ids, intron_clusts, sequencing_type="single_cell"):
    """
    Read and process a single junction file, returning junction and cluster counts for relevant junctions.
    """
    file_junction_counts = read_single_junction_file_filt(junc_file, relevant_junction_ids, sequencing_type)
    
    # Prepare to store the results in regular dictionaries
    cell_junction_counts = {}
    cell_cluster_counts = {}

    # Precompute junction-to-cluster mapping
    junction_to_cluster = dict(zip(intron_clusts['junction_id'], intron_clusts['Cluster']))

    # Accumulate junction and cluster counts
    for cell_id, junctions in file_junction_counts.items():
        if cell_id not in cell_junction_counts:
            cell_junction_counts[cell_id] = {}
            cell_cluster_counts[cell_id] = {}
        for junction_id, count in junctions.items():
            if junction_id not in cell_junction_counts[cell_id]:
                cell_junction_counts[cell_id][junction_id] = 0
            cell_junction_counts[cell_id][junction_id] += count
            
            # Get the corresponding cluster for the junction
            if junction_id in junction_to_cluster:
                cluster = junction_to_cluster[junction_id]
                if cluster not in cell_cluster_counts[cell_id]:
                    cell_cluster_counts[cell_id][cluster] = 0
                cell_cluster_counts[cell_id][cluster] += count

    return cell_junction_counts, cell_cluster_counts

def process_files_and_build_matrices_parallel(junction_files, relevant_junction_ids, intron_clusts, sequencing_type="single_cell", max_workers=8):
    """
    Process junction files in parallel and build sparse matrices for cell-by-junction and cell-by-cluster data.
    """

    # 1. Check format of junc_files and convert to list if necessary
    if isinstance(junction_files, list):
        pass
    elif "," in junction_files:
        junction_files = junction_files.split(",")
    else:
        junction_files = [junction_files]

    # Initialize regular dictionaries
    cell_junction_counts = {}
    cell_cluster_counts = {}

    # Parallelize the processing of junction files
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for junc_file in junction_files:
            futures.append(executor.submit(read_and_process_file, junc_file, relevant_junction_ids, intron_clusts, sequencing_type))
        
        # Aggregate the results as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Merging results"):
            junc_counts, clust_counts = future.result()
            merge_dictionaries(cell_junction_counts, junc_counts)
            merge_dictionaries(cell_cluster_counts, clust_counts)

    print(f"Building the sparse cell-by-junction counts matrix!")
    # Build the sparse matrix for cell-by-junction
    cells = sorted(cell_junction_counts.keys())
    junctions = sorted(relevant_junction_ids)
    cell_idx = {cell: i for i, cell in enumerate(cells)}
    junc_idx = {junc: i for i, junc in enumerate(junctions)}

    rows, cols, data = [], [], []
    for cell, juncs in cell_junction_counts.items():
        for junc, count in juncs.items():
            rows.append(cell_idx[cell])
            cols.append(junc_idx[junc])
            data.append(count)

    # Create the cell-by-junction sparse matrix
    cell_by_junction_matrix = coo_matrix((data, (rows, cols)), shape=(len(cells), len(junctions)))

    print(f"Building the sparse cell-by-junction cluster counts matrix!")
    # Now create the sparse matrix for cell-by-cluster
    rows, cols, data = [], [], []
    for cell, clusts in cell_cluster_counts.items():
        for cluster, clust_count in clusts.items():
            # Get the junctions corresponding to this cluster
            junctions_in_cluster = intron_clusts[intron_clusts['Cluster'] == cluster]['junction_id'].values
            
            # For each junction in this cluster, assign the cluster count to the corresponding junction index
            for junction_id in junctions_in_cluster:
                if junction_id in junc_idx:  # Only add if the junction is relevant
                    rows.append(cell_idx[cell])  # Consistent cell index
                    cols.append(junc_idx[junction_id])  # Use junction index for this cluster
                    data.append(clust_count)  # Add the cluster count, not the junction count

    # Create the cell-by-cluster sparse matrix, same shape as cell_by_junction_matrix
    cell_by_cluster_matrix = coo_matrix((data, (rows, cols)), shape=(len(cells), len(junctions)))
    print(f"Returning sparse matrices and orders of cells and junctions!")
    
    return cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions

def sanity_check(cell_by_junction_matrix, cell_by_cluster_matrix, cell_junction_counts, cell_cluster_counts, intron_clusts, cells, junctions):
    # Pick a random cell
    random_cell = random.choice(cells)

    # Ensure the cell has junctions
    if random_cell not in cell_junction_counts or not cell_junction_counts[random_cell]:
        print(f"Cell {random_cell} has no junctions.")
        return

    # Pick a random junction from that cell
    random_junction = random.choice(list(cell_junction_counts[random_cell].keys()))

    # Get the row and column indices
    cell_idx = cells.index(random_cell)
    junc_idx = junctions.index(random_junction)

    # Extract the values from the sparse matrices
    junction_count_matrix_value = cell_by_junction_matrix.toarray()[cell_idx, junc_idx]
    cluster_count_matrix_value = cell_by_cluster_matrix.toarray()[cell_idx, junc_idx]

    # Extract the expected values from cell_junction_counts and cell_cluster_counts
    expected_junction_count = cell_junction_counts[random_cell][random_junction]

    # Find the cluster corresponding to the random_junction
    cluster = intron_clusts[intron_clusts['junction_id'] == random_junction]['Cluster'].values[0]
    expected_cluster_count = cell_cluster_counts[random_cell][cluster]

    # Print comparison results
    print(f"Sanity Check for Cell: {random_cell}, Junction: {random_junction}")
    print(f"Junction Count - Sparse Matrix: {junction_count_matrix_value}, Expected: {expected_junction_count}")
    print(f"Cluster Count  - Sparse Matrix: {cluster_count_matrix_value}, Expected: {expected_cluster_count}")

    # Add assert statements to ensure the values are correct
    assert junction_count_matrix_value == expected_junction_count, "Junction counts mismatch!"
    assert cluster_count_matrix_value == expected_cluster_count, "Cluster counts mismatch!"

    print("Sanity check passed!")


def create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions, metadata_subset, intron_clusts, save_file=False, prefix="ATSE_Anndata_Object"):
    """
    Create an AnnData object using the cell-by-junction and cell-by-cluster matrices,
    combined with the metadata for cells.
    
    Parameters:
    -----------
    - cell_by_junction_matrix: Sparse matrix of cells by junctions.
    - cell_by_cluster_matrix: Sparse matrix of cells by clusters.
    - cells: List of cell IDs.
    - metadata_subset: DataFrame with metadata for the cells.
    
    Returns:
    --------
    adata: AnnData object containing the junction and cluster matrices, along with cell metadata.
    """

    # Convert cells to a DataFrame for joining with metadata
    cell_df = pd.DataFrame({'cell_id': cells})

    # Merge cell_df with metadata to get the corresponding metadata for cells in the matrix
    metadata_matched = pd.merge(cell_df, metadata_subset, on='cell_id', how='inner')
    metadata_matched = metadata_matched.set_index('cell_id').reindex(cells)

   # Prepare junction metadata (var) from intron_clusts
    junction_var = intron_clusts[['junction_id', 'gene_id', 'num_cells_with_junc', 'total_read_counts', 
                                  'Cluster', 'Count']].copy()
    junction_var.rename(columns={'Count': 'CountJuncs'}, inplace=True)

    # Reorder the junction metadata to match the order of relevant_junction_ids in cell_by_junction_matrix
    junction_var = junction_var.set_index('junction_id').reindex(junctions)

    # Create the AnnData object
    adata = ad.AnnData(X=cell_by_junction_matrix, 
                       obs=metadata_matched,  # Cell metadata
                       var=junction_var)  # Junction metadata (features/variables)

    # Add cell-by-cluster matrix as a layer in the AnnData object
    adata.layers["cell_by_junction_matrix"] = cell_by_junction_matrix.tocsr()  
    adata.layers["cell_by_cluster"] = cell_by_cluster_matrix.tocsr()  
    
    # Convert the COO matrix to CSR format since COO can't be used when saving Anndata file 
    adata.X = adata.X.tocsr()  

    if save_file:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        adata_path = f"{prefix}_{current_time}.h5ad"
        # Save the AnnData object to the h5ad file with gzip compression
        adata.write_h5ad(adata_path, compression='gzip')
    
    else: 
        return adata
    
    print(f"Done generating splicing Anndata Object!")
