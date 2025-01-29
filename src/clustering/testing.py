# Import necessary libraries
import os
import pandas as pd
import anndata as ad
import numpy as np
import sys 
import datetime
import pickle
import gffutils 

# Add the directory containing find_intron_clusters_v3.py to Python path
LEAFLET_SRC = '/gpfs/commons/home/kisaev/Leaflet-private/src'
sys.path.append(LEAFLET_SRC)

# Now import your modules
from clustering.find_intron_clusters_v3 import (
    JunctionReader,
    JunctionAnalyzer, 
    GenomeDB, 
    ATSEAnalyzer
)

import visualization.IsovizPy as ja

# Set up paths for genome files and junction files (Regtools based)
gtf_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/genes/genes.gtf"
fasta_file = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/genome_files/gencode.vM19/fasta/genome.fa"
combined_junctions = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_processing_20250128/results"
output_path = "/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/"
junction_files=pd.read_csv("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/junction_files.txt", header=None, sep="\t")
print(f"Number of junction files going in to the analysis: {len(junction_files)}")

# Set up parameters for junction processing
min_intron = 50
max_intron = 500000
min_junc_reads = 20 
min_num_cells_wjunc = 5
batch_size = 32
num_workers = 10

# Initialize genome database
genome_db = GenomeDB(db_name="gencodeVM19", gtf_file=gtf_file, fasta_file=fasta_file)

# Initialize the junction reader
reader = JunctionReader(batch_size=batch_size, 
                        min_cells=min_num_cells_wjunc, 
                        min_reads=min_junc_reads, 
                        min_intron=min_intron, 
                        max_intron=max_intron, 
                        num_workers=num_workers)

# For testing, sample 100 junction files
junction_files = junction_files.sample(20)

# Process and filter junctions
# Note: if you have a large number of junctions (e.g. > 1000),
# The reader.process_files() step should be done seperately in an outside job 
# by breaking up the junction files into smaller chunks and using slurm array
# this results in a massive speedup
junctions = reader.process_files(junction_files[0].values)

# Filter junctions based on most basic QC parameters
filtered_junctions = reader.SJ_QC(junctions)

# Initialize junction analyzer
analyzer = JunctionAnalyzer(fasta_file=fasta_file, db=genome_db.get_db(), tolerance=100)

# Ensure that the junctions have canonical splice sites
junctions = analyzer.check_splice_sites(filtered_junctions)
canonical_junctions = analyzer.filter_canonical(junctions)

# Annotate junctions 5' and 3' with trancript exons 
annotated_junctions = analyzer.check_junction_annotation(canonical_junctions)

# Filter junctions that are not annotated
filtered_junctions = analyzer.filter_annotated(annotated_junctions, annotation_status_include="unanno_also")

# Initialize the ATSE analyzer
atse_analyzer = ATSEAnalyzer()

# Build initial splice graph 
sgraph, stats = atse_analyzer.build_splice_graph(filtered_junctions)

# Find ATSEs using the splice graph
ATSE_groups, sorted_counts = atse_analyzer.find_atse_groups(sgraph)

# Try classifyiing ATSEs (in progress, may not be totally accurate...)
ATSE_lablled, event_counts = atse_analyzer.classify_events(sgraph, ATSE_groups)
print(event_counts) 

# # Save the ATSEs to a file
# today = datetime.datetime.now().strftime("%Y-%m-%d")
# atse_file = f"{today}_test_atse_file.txt"
# 
# output_file = os.path.join(output_path, atse_file)
# atse_analyzer.save_atse_file(ATSE_lablled, filtered_junctions, output_file)
# 
# # Visualize random ATSEs
# p=pd.read_csv("/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaSenis/Leaflet/ATSEmap/output/2025-01-29_test_atse_file.txt.gz", sep="\t")
# db = gffutils.FeatureDB("gencodeVM19", keep_order=True)
# juncs = p[p["event_id"]=="ATSE_16"]
# juncs["usage_ratio"] = 0
# juncs["Cluster"] = juncs["event_id"]
# splice_junctions = ja.convert_junction_ids(juncs)
# 
# # Check junction annotations
# junction_annotation_results = ja.check_junction_annotation(splice_junctions, db)
# 
# # Extract unique transcript IDs from junction_labels
# unique_transcripts = list({transcript for label in junction_annotation_results for transcript in label['transcripts']})
# 
# # Fetch transcript exon coordinates and determine plot boundaries
# transcript_data = ja.fetch_transcripts_and_annotations(db, unique_transcripts)
# region_start, region_end = ja.determine_region_boundaries(splice_junctions)
# ja.plot_exons_and_junctions(db, transcript_data, splice_junctions, region_start-200, region_end-400, base_width=10, trans_height=0.2, show_usage=False, show_junc_lines=True)
