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
import os

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

def read_single_junction_file_filt(junc_file, relevant_junction_ids, sequencing_type="smart_seq", junc_idx=None):
    """
    Read a single junction file and return a dictionary of junctions that match the relevant junction IDs.

    Parameters:
    - junc_file: Path to the junction file.
    - relevant_junction_ids: Set of relevant junction IDs to filter.
    - sequencing_type: Type of sequencing data ('smart_seq' or 'bulk').
    - junc_idx: Dictionary mapping junction IDs to their indices.

    Returns:
    - junc_dict: Dictionary containing cell counts and total read scores for relevant junctions.
    """

    # Specify dtypes to reduce memory usage
    dtypes = {0: str, 1: 'int32', 2: 'int32', 3: str, 4: 'int32', 5: str,
              6: 'int32', 7: 'int32', 8: str, 9: 'int32', 10: str, 11: str}
    juncs = pd.read_csv(junc_file, sep="\t", header=None, dtype=dtypes)
    
    try:
        juncs = pd.read_csv(junc_file, sep="\t", header=None, dtype=dtypes)
    except pd.errors.EmptyDataError:
        print(f"Empty or invalid file encountered: {junc_file}")
        return {}, {}

    # Add column names here:
    col_names = ["chrom", "chromStart", "chromEnd", "name", "score", "strand", 
         "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
    if sequencing_type in ["smart_seq", "10x", "split-seq", "easysci"]:
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
        junction_index = junc_idx[junction_id]

        if sequencing_type == "smart_seq":
            # For Smart-seq2, there's only one cell per file
            cell_id = row['cell_readcounts'].split(":")[0]  # Get cell ID
            read_count = row['score']
            cell_junction_counts[cell_id][junction_index] += read_count

        elif sequencing_type == "10x":
            # For 10x, multiple cells and their read counts may be in the 'cell_readcounts' column
            for cell_info in row['cell_readcounts'].split(','):
                cell_id, read_count = cell_info.split(":")
                read_count = int(read_count)
                cell_junction_counts[cell_id][junction_index] += read_count

    return cell_junction_counts

def read_and_process_file(junc_file, relevant_junction_ids, sequencing_type="smart_seq", junc_idx=None, cluster_idx=None, cluster_idx_flip=None):
    """
    Read and process a single junction file, returning junction and cluster counts for relevant junctions.
    """

    # First ensure junc_file exists before reading it
    if os.path.exists(junc_file):
        
        file_junction_counts = read_single_junction_file_filt(junc_file, relevant_junction_ids, sequencing_type, junc_idx)
        
        # Prepare to store the results in regular dictionaries
        cell_junction_counts = {}
        cell_cluster_counts = {}

        # Accumulate junction counts as usual
        for cell_id, junctions in file_junction_counts.items():
            if cell_id not in cell_junction_counts:
                cell_junction_counts[cell_id] = {}
                cell_cluster_counts[cell_id] = {}

            # Keep track of cluster counts for the current cell
            cluster_totals = {}

            for junction_index, count in junctions.items():

                # Update cell_junction_counts using the junction index
                if junction_index not in cell_junction_counts[cell_id]:
                    cell_junction_counts[cell_id][junction_index] = 0
                cell_junction_counts[cell_id][junction_index] += count

                # Get the corresponding cluster index for this junction
                cluster_name = cluster_idx[junction_index]

                # Accumulate counts for clusters, but don't add them directly yet
                if cluster_name not in cluster_totals:
                    cluster_totals[cluster_name] = 0
                cluster_totals[cluster_name] += count

            # Second pass: loop through all cluster_totals (observed clusters) and add zero counts for unobserved junctions in that cluster
            for cluster_name in cluster_totals.keys():
                # Find all junctions in this cluster
                junctions_in_cluster = cluster_idx_flip[cluster_name]
                for junction_index in junctions_in_cluster:
                    # Add a zero count for junctions not observed in this cell
                    if junction_index not in cell_junction_counts[cell_id]:
                        cell_junction_counts[cell_id][junction_index] = 0

                    # Add the total cluster count for this junction
                    if junction_index not in cell_cluster_counts[cell_id]:
                        cell_cluster_counts[cell_id][junction_index] = 0
                    cell_cluster_counts[cell_id][junction_index] += cluster_totals[cluster_name]  # Use cluster_name here

        return cell_junction_counts, cell_cluster_counts
    else:
        print(f"Warning: The file {junc_file} does not exist. Skipping.")
        return {}, {}  # Return empty dictionaries if file does not exist

def process_files_and_build_matrices_parallel(junction_files, relevant_junction_ids, intron_clusts, sequencing_type="smart_seq", max_workers=8):
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

    junctions = sorted(relevant_junction_ids)
    junction_to_cluster = dict(zip(intron_clusts['junction_id'], intron_clusts['event_id']))
    ordered_clusters = [junction_to_cluster[junc] for junc in junctions if junc in junction_to_cluster]

    junc_idx = {junction: i for i, junction in enumerate(junctions)}
    junc_idx_flip = {i: junction for i, junction in enumerate(junctions)}
    cluster_idx = {i: cluster for i, cluster in enumerate(ordered_clusters)}
    cluster_idx_flip = defaultdict(list)
    for i, cluster in enumerate(ordered_clusters):
        cluster_idx_flip[cluster].append(i)  # Append the index to the cluster's list
    
    # Initialize regular dictionaries
    cell_junction_counts = {}
    cell_cluster_counts = {}

    # Parallelize the processing of junction files
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for junc_file in junction_files:
            if not os.path.exists(junc_file) or os.stat(junc_file).st_size == 0:
                print(f"Skipping invalid or empty file: {junc_file}")
                continue
            futures.append(executor.submit(read_and_process_file, junc_file, relevant_junction_ids, sequencing_type, junc_idx, cluster_idx, cluster_idx_flip))
        
        # Aggregate the results as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Merging results", miniters=100):
            junc_counts, clust_counts = future.result()
            merge_dictionaries(cell_junction_counts, junc_counts)
            merge_dictionaries(cell_cluster_counts, clust_counts)

    # Remove any null entries from dictionary if there are any 
    cell_junction_counts = {cell: junctions for cell, junctions in cell_junction_counts.items() if junctions is not None}
    cell_cluster_counts = {cell: clusters for cell, clusters in cell_cluster_counts.items() if clusters is not None}
    
    # Build the sparse matrix for cell-by-junction
    cells = sorted(cell_junction_counts.keys())
    cell_idx = {i: cell for i, cell in enumerate(cells)}

    # Replace keys to be cell_indices 
    cell_id_to_index = {v: k for k, v in cell_idx.items()}

    # Replace the cell id keys with the corresponding indices
    indexed_cell_junction_counts = {
        cell_id_to_index[cell]: junctions for cell, junctions in cell_junction_counts.items() if cell in cell_id_to_index
    }

    indexed_cell_cluster_counts = {
        cell_id_to_index[cell]: clusters for cell, clusters in cell_cluster_counts.items() if cell in cell_id_to_index
    }

    # Flatten dictionaries for cell-by-junction
    flat_cells_junc = np.array([cell for cell in indexed_cell_junction_counts for junc in indexed_cell_junction_counts[cell]])
    flat_junctions = np.array([junc for cell in indexed_cell_junction_counts for junc in indexed_cell_junction_counts[cell]])
    flat_counts_junc = np.array([indexed_cell_junction_counts[cell][junc] for cell in indexed_cell_junction_counts for junc in indexed_cell_junction_counts[cell]])
    
    # Build sparse matrix for cell-by-junction
    cell_by_junction_matrix = coo_matrix((flat_counts_junc, (flat_cells_junc, flat_junctions)), shape=(len(cells), len(junc_idx)))

    # Flatten dictionaries for cell-by-cluster
    flat_cells_clust = np.array([cell for cell in indexed_cell_cluster_counts for clust in indexed_cell_cluster_counts[cell]])
    flat_clusters = np.array([clust for cell in indexed_cell_cluster_counts for clust in indexed_cell_cluster_counts[cell]])
    flat_counts_clust = np.array([indexed_cell_cluster_counts[cell][clust] for cell in indexed_cell_cluster_counts for clust in indexed_cell_cluster_counts[cell]])

    # Build sparse matrix for cell-by-cluster
    cell_by_cluster_matrix = coo_matrix((flat_counts_clust, (flat_cells_clust, flat_clusters)), shape=(len(cells), len(cluster_idx)))

    return cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions, cell_idx, junc_idx, cluster_idx, cluster_idx_flip

def create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cell_idx, junc_idx, metadata, intron_clusts, save_file=False, meta_cell_column="cell_id", prefix="ATSE_Anndata_Object"):
    """
    Create an AnnData object using the cell-by-junction and cell-by-cluster matrices,
    combined with the metadata for cells.
    
    Parameters:
    -----------
    - cell_by_junction_matrix: Sparse matrix of cells by junctions.
    - cell_by_cluster_matrix: Sparse matrix of cells by clusters.
    - cell_idx: Dictionary mapping cell indices to cell names.
    - junc_idx: Dictionary mapping junction IDs to indices.
    - cluster_idx_flip: Dictionary mapping cluster names to lists of junction indices.
    - metadata: DataFrame with metadata for the cells.
    - intron_clusts: DataFrame with intron cluster information.
    - save_file: Boolean flag indicating whether to save the AnnData object.
    - meta_cell_column: Column in metadata representing cell identifiers.
    - prefix: Prefix for the filename when saving the AnnData object.
    
    Returns:
    --------
    adata: AnnData object containing the junction and cluster matrices, along with cell metadata.
    """

    # Prepare cell metadata (obs)
    cells = list(cell_idx.values())  # Get the list of cells
    if meta_cell_column not in metadata.columns:
        raise ValueError(f"Column '{meta_cell_column}' not found in metadata.")

    metadata_matched = metadata.set_index(meta_cell_column).reindex(cells)

    # Add cell_id_index to metadata_matched
    metadata_matched['cell_id_index'] = metadata_matched.index.map({v: k for k, v in cell_idx.items()})

    # Reset the index to make 'cell_id' a regular column
    metadata_matched = metadata_matched.reset_index()

    # Prepare junction metadata (var) from intron_clusts
    # Check if 'gene_id' column is present, then include it; otherwise, skip it
    columns_to_include = ['junction_id', 'event_id', 'total_score', 'splice_motif', 'label_5_prime',
       'label_3_prime']

    # Add 'gene_id' to columns if it exists in intron_clusts
    if 'gene_id' in intron_clusts.columns:
        columns_to_include.insert(1, 'gene_id')  # Insert 'gene_id' at the correct position if present

    # Create a copy of the selected columns
    junction_var = intron_clusts[columns_to_include].copy()
    junction_var.rename(columns={'total_score': 'CountJuncs'}, inplace=True)

    # Reorder the junction metadata to match the order of relevant_junction_ids in cell_by_junction_matrix
    junctions = list(junc_idx.keys())  # Use the junction IDs in order
    junction_var = junction_var.set_index('junction_id').reindex(junctions)

    # Add junction_id_index to junction_var
    junction_var = junction_var.reset_index()
    junction_var['junction_id_index'] = junction_var['junction_id'].map(junc_idx)

    # Create the AnnData object
    adata = ad.AnnData(X=cell_by_junction_matrix, 
                       obs=metadata_matched,  # Cell metadata
                       var=junction_var)  # Junction metadata (features/variables)

    # Add cell-by-cluster matrix as a layer in the AnnData object
    adata.layers["cell_by_junction_matrix"] = cell_by_junction_matrix.tocsr()  
    adata.layers["cell_by_cluster_matrix"] = cell_by_cluster_matrix.tocsr()  
    
    # Ensure cell_id and junction_id are of type string
    metadata_matched.index = metadata_matched.index.astype(str)
    junction_var['junction_id'] = junction_var['junction_id'].astype(str)

    # Convert the COO matrix to CSR format since COO can't be used when saving AnnData file 
    adata.X = adata.X.tocsr()  
    adata.obs.head()
    print(adata.obs.dtypes)

    if save_file:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        adata_path = f"{prefix}_{current_time}.h5ad"
        # Save the AnnData object to the h5ad file with gzip compression
        adata.write_h5ad(adata_path, compression='gzip')
        print(f"AnnData object saved as {adata_path}")
    
    else: 
        return adata
    
    print(f"Done generating splicing AnnData Object!")