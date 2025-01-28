from find_intron_clusters_v3 import JunctionReader, JunctionAnalyzer, GenomeDB, ATSEAnalyzer, create_analysis_summary

# Import necessary libraries
import os
import pandas as pd
import random
from scipy.sparse import csr_matrix, coo_matrix
import anndata as ad
import numpy as np
import sys 
import datetime

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering')
import find_intron_clusters_v2
# Reload the module if you've made changes and want to update it
import importlib
importlib.reload(find_intron_clusters_v2)

import find_intron_clusters_v3
importlib.reload(find_intron_clusters_v3)
from find_intron_clusters_v3 import JunctionReader, JunctionAnalyzer, GenomeDB, ATSEAnalyzer, create_analysis_summary

import prep_anndata_object
from prep_anndata_object import *
importlib.reload(prep_anndata_object)

# Paths
juncs_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/junctions/"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/"

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
junction_files = portion_in_list2

# Write the junction files to a text file in output_path 
with open(os.path.join(output_path, "junction_files.txt"), 'w') as f:
    for item in junction_files:
        f.write("%s\n" % item)

# for testing purposes, only use the first 1000 junction files
junction_files = portion_in_list2[0:1000]

print(f"Number of junction files going in to the analysis: {len(junction_files)}")
gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
fasta_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/fasta/genome.fa"

min_intron = 50
max_intron = 500000
min_junc_reads = 100 
min_num_cells_wjunc = 10
batch_size = 32
num_workers = 10

# Initialize the junction reader
reader = JunctionReader(batch_size=batch_size, min_cells=min_num_cells_wjunc, min_reads=min_junc_reads, min_intron=min_intron, max_intron=max_intron, num_workers=num_workers)

# Process and filter junctions
junctions = reader.process_files(junction_files)
filtered_junctions = reader.SJ_QC(junctions)

# Initialize genome database
genome_db = GenomeDB(db_name="gencodeVM19", gtf_file=gtf_file, fasta_file=fasta_file)

# Initialize junction analyzer
analyzer = JunctionAnalyzer(fasta_file=fasta_file, db=genome_db.get_db())

# Ensure that the junctions have canonical splice sites
junctions = analyzer.check_splice_sites(filtered_junctions)
canonical_junctions = analyzer.filter_canonical(junctions)

# Annotate junctions 5' and 3' with trancript exons 
print(f"Keep in mind, splice junction annotation with exon boundaries is the longest step!")
annotated_junctions = analyzer.check_junction_annotation(canonical_junctions)

# Filter junctions that are not annotated
filtered_junctions = analyzer.filter_annotated(annotated_junctions)

# Summarize the analysis
create_analysis_summary(reader, len(junction_files), filtered_junctions)

# Initialize the ATSE analyzer
atse_analyzer = ATSEAnalyzer()

# Build initial splice graph 
sgraph, stats = atse_analyzer.build_splice_graph(filtered_junctions)

# Find ATSEs using the splice graph
ATSE_groups, sorted_counts = atse_analyzer.find_atse_groups(sgraph)

# Try classifyiing ATSEs 
ATSE_lablled, event_counts = atse_analyzer.classify_events(sgraph, ATSE_groups)
print(event_counts) 

# Save the ATSEs to a file
# Save file as today's date and time and "test_atse_file.txt" 
today = datetime.datetime.now().strftime("%Y-%m-%d")
atse_file = f"{today}_test_atse_file.txt"

output_file = os.path.join(output_path, atse_file)
atse_analyzer.save_atse_file(ATSE_lablled, filtered_junctions, output_file)

## to submit:
# cd /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap
# conda activate LeafletSC
# script_path=/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/testing.py
# sbatch --wrap="python $script_path" --mem=300G --time=3-00:00:00 -J TMSLeafletFA -p bigmem
