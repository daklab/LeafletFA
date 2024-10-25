# Import necessary libraries
import os
import pandas as pd
import random
from scipy.sparse import csr_matrix, coo_matrix
import anndata as ad
import numpy as np
import sys 

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering')
import find_intron_clusters_v2
# Reload the module if you've made changes and want to update it
import importlib
importlib.reload(find_intron_clusters_v2)

import prep_anndata_object
from prep_anndata_object import *
importlib.reload(prep_anndata_object)

# Paths
juncs_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/junctions/"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet"
gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"

# Metadata file
metadata_path = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/metadata/tabula-muris-senis-full-metadata.csv"
metadata = pd.read_csv(metadata_path)

# Filter metadata for FACS method
metadata = metadata[metadata['method'] == 'facs']
metadata_subset = metadata.copy()

# Function to format cell IDs based on the month group
def format_cell_id(index, group):
    if group == '3m':
        parts = index.replace('.', '-', 1).replace('_', '-', 1).split('.')
        corrected_part = parts[1].replace('-', '_', 1)
        return parts[0] + '-' + corrected_part + '-1-1'
    else:
        return index.split('.')[0]

# Apply the formatting to the metadata subset
metadata_subset['cell_id'] = metadata_subset.apply(lambda row: format_cell_id(row['index'], row['age']), axis=1)
cell_ids_set = set(metadata_subset['cell_id'].values)

# Keep only important columns in metadata_subset
metadata_subset = metadata_subset[['cell_id', 'age', 'batch', 'cell_ontology_class', 'method', 'mouse.id', 'sex', 'subtissue', 'tissue']]

# Get all junction files and filter
all_junc_files = []
all_dirs = os.listdir(juncs_path)
print(all_dirs)
for dir in all_dirs:
    dir_path = os.path.join(juncs_path, dir)
    if os.path.isdir(dir_path):
        junc_files = os.listdir(dir_path)
        junc_files = [os.path.join(dir_path, x) for x in junc_files]
        all_junc_files.extend(junc_files)

# Filter the files using the set
portion_in_list2 = [x for x in all_junc_files if os.path.splitext(os.path.basename(x))[0] in cell_ids_set]
portion_in_list2 = [x + "/junctions_with_barcodes.bed" for x in portion_in_list2]

# Shuffle the list
random.shuffle(portion_in_list2)

print(len(portion_in_list2))

# Define additional parameters
output_file = os.path.join(output_path, "tabula_senis_test_intron_clusters")
junc_bed_file = os.path.join(output_path, "tabula_senis_test_intron_clusters.bed")
sequencing_type = "single_cell"
min_intron = 50
max_intron = 500000
min_junc_reads = 10 
min_num_cells_wjunc = 5
max_workers = 10
batch_size = 20
run_clustering = False

# Check if output files already exist to skip
print("Running intron clustering for test data...")

if run_clustering:
    intron_clusts_file = find_intron_clusters_v2.main(
            junc_files=portion_in_list2,
            gtf_file=gtf_file,
            output_file=output_file,
            sequencing_type=sequencing_type,
            junc_bed_file=junc_bed_file,
            threshold_inc=0.01,
            min_intron=min_intron,
            max_intron=max_intron,
            min_junc_reads=min_junc_reads,
            singleton=False,
            min_num_cells_wjunc=min_num_cells_wjunc,
            filter_shared_ss=True,
            max_workers=max_workers,
            batch_size=batch_size,
            run_notebook=False
        )

else:
    # Read intron clusts file 
    intron_clusts_file="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/tabula_senis_test_intron_clusters_50_500000_10_20240927_single_cell.gz"

print("Reading in obtained intron cluster (ATSE file!)")
intron_clusts = pd.read_csv(intron_clusts_file, sep="}")
relevant_junction_ids = set(intron_clusts['junction_id'])

# Extract single cell junction and cluster counts 
print("Process single cell junction counts and assemble sparse matrices!")
cell_by_junction_matrix, cell_by_cluster_matrix, cells, junctions, cell_idx, junc_idx, cluster_idx, cluster_idx_flip = process_files_and_build_matrices_parallel(portion_in_list2, relevant_junction_ids, intron_clusts, sequencing_type="smart_seq")

# Save as Anndata object!
print("Save Anndata object!")
create_anndata_object(cell_by_junction_matrix, cell_by_cluster_matrix, cell_idx, junc_idx, metadata_subset, intron_clusts, save_file=True, prefix="ATSE_Anndata_Object")