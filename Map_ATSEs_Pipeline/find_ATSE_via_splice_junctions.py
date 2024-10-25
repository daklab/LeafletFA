import pandas as pd
import numpy as np
import argparse
import pyranges as pr
from tqdm import tqdm
import time
import warnings
import gzip
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from gtfparse import read_gtf #initially tested with version 1.3.0)
import logging
import sys
sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/utils')
from read_gtf import read_gtf_file  
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=FutureWarning, module="pyranges")
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Read in file that lists junctions for all samples, \
                                 one file per line and no header')

parser.add_argument('--junc_files', dest='junc_files',
                    help='path that has all junction files along with counts in single cells or bulk samples, \
                    make sure path ends in "/" Can also be a comma separated list of paths. If you have a complex folder structure, \
                        provide the most root folder that contains all the junction files. The script will recursively search for junction files with the suffix provided in the next argument.')

parser.add_argument('--sequencing_type', dest='sequencing_type',
                    default='single_cell',
                    help='were the junction obtained using data from single cell or bulk sequencing? \
                        options are "single_cell" or "bulk". Default is "single_cell"')

parser.add_argument('--gtf_file', dest='gtf_file', 
                    default = None,
                    help='a path to a gtf file to annotate high confidence junctions, \
                    ideally from long read sequencing, if not provided, then the script will not \
                        annotate junctions based on gtf file')

parser.add_argument('--output_file', dest='output_file', 
                    default='intron_clusters.txt',
                    help='name of the output file to save intron cluster file to')

parser.add_argument('--junc_bed_file', dest='junc_bed_file', 
                    default='juncs.bed',
                    help='name of the output bed file to save final list of junction coordinates to')

parser.add_argument('--threshold_inc', dest='threshold_inc',
                    default=0.005,
                    help='threshold to use for removing clusters that have junctions with low read counts \
                        (proportion of reads relative to intron cluster) at either end, default is 0.01')

parser.add_argument('--min_intron_length', dest='min_intron_length',
                    default=50,
                    help='minimum intron length to consider, default is 50')

parser.add_argument('--max_intron_length', dest='max_intron_length',
                    default=500000,
                    help='maximum intron length to consider, default is 500000')

parser.add_argument('--min_junc_reads', dest='min_junc_reads',
                    default=1,
                    help='minimum number of reads to consider a junction, default is 1')

parser.add_argument('--keep_singletons', dest='keep_singletons', 
                    default=False,
                    help='Indicate whether you would like to keep "clusters" composed of just one junction.\
                          Default is False which means do not keep singletons')

parser.add_argument('--junc_suffix', dest='junc_suffix', #set default param to *.junc, 
                    default='*.juncs', 
                    help='suffix of junction files')

parser.add_argument('--min_num_cells_wjunc', dest='min_num_cells_wjunc',
                    default=1,
                    help='minimum number of cells that have a junction to consider it, default is 1')

parser.add_argument('--run_notebook', dest='run_notebook',
                    default=False,
                    help='Indicate whether you would like to run the script in a notebook and return the table in session.\
                          Default is False')

parser.add_argument('--filter_shared_ss', dest='filter_shared_ss',
                    default=True,
                    help='Indicate whether you would like to filter clusters by junctions with shared splice sites.\
                          Default is True')

parser.add_argument('--max_workers', type=int, default=8, 
                    help='Maximum number of parallel workers for processing files. Default is 8')

parser.add_argument('--batch_size', type=int, default=1000, 
                        help='Batch size for processing junction files in parallel. Default is 1000')

args = parser.parse_args(args=[])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                      Utilities
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def process_gtf(gtf_file): #make this into a seperate script that processes the gtf file into gr object that can be used in the main scriptas input 
    """
    Process the GTF file into a pyranges object.

    Parameters:
    - gtf_file (str): Path to the GTF file.

    Returns:
    - gtf_exons_gr (pyranges.GenomicRanges): Processed pyranges object.
    """

    print("The gtf file you provided is " + gtf_file)
    print("Reading the gtf may take a minute...")

    # calculate how long it takes to read gtf_file and report it 
    start_time = time.time()
    #[1] extract all exons from gtf file provided 
    gtf = read_gtf(gtf_file, result_type="pandas") #to reduce the speed of this, can just get rows with exon in the feature column (preprocess this before running package)? check if really necessary
    end_time = time.time()

    print("Reading gtf file took " + str(round((end_time-start_time), 2)) + " seconds")
    # assert that gtf is a non empty dataframe otherwise return an error
    if gtf.empty or type(gtf) != pd.DataFrame:
        raise ValueError("The gtf file provided is empty or not a pandas DataFrame. Please provide a valid gtf file and ensure you have the \
                         latest version of gtfparse installed by running 'pip install gtfparse --upgrade'")
    
    # Convert the seqname column to a string in gtf 
    gtf["seqname"] = gtf["seqname"].astype(str)

    # Make a copy of the DataFrame
    gtf_exons = gtf[(gtf["feature"] == "exon")].copy()

    if gtf_exons['seqname'].str.contains('chr').any():
        gtf_exons.loc[gtf_exons['seqname'].str.contains('chr'), 'seqname'] = gtf_exons['seqname'].map(lambda x: x.lstrip('chr').rstrip('chr'))

    if not set(['seqname', 'start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'transcript_id', 'exon_id']).issubset(gtf_exons.columns):
        # print the columns that the file is missing
        missing_cols = set(['seqname', 'start', 'end', 'score', 'strand', 'gene_id', 'gene_name', 'transcript_id', 'exon_id']).difference(gtf_exons.columns)
        print("Your gtf file is missing the following columns: " + str(missing_cols))

        # if the missing column is just exon_id, we can generate it
        if "exon_id" in missing_cols:
            # add exon_id to gtf_exons
            print("Adding exon_id column to gtf file")
            gtf_exons.loc[:, "exon_id"] = gtf_exons["transcript_id"] + "_" + gtf_exons["start"].astype(str) + "_" + gtf_exons["end"].astype(str)
        else:
            pass

    # Convert the DataFrame to a PyRanges object
    gtf_exons_gr = pr.from_dict({"Chromosome": gtf_exons["seqname"], "Start": gtf_exons["start"], "End": gtf_exons["end"], "Strand": gtf_exons["strand"], "gene_id": gtf_exons["gene_id"], "gene_name": gtf_exons["gene_name"], "transcript_id": gtf_exons["transcript_id"], "exon_id": gtf_exons["exon_id"]})

    # Remove rows where exon start and end are the same or when gene_name is empty
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.Start == gtf_exons_gr.End)]
    gtf_exons_gr = gtf_exons_gr[ ~ (gtf_exons_gr.gene_name == "")]

    # When do I need to do this? depends on gtf file used? base 0 or 1? probably need this to be a parameter 
    gtf_exons_gr.Start = gtf_exons_gr.Start-1

    # Drop duplicated positions on same strand 
    gtf_exons_gr = gtf_exons_gr.drop_duplicate_positions(strand=True) # Why are so many gone after this? 

    # Print the number of unique exons, transcript ids, and gene ids
    print("The number of unique exons is " + str(len(gtf_exons_gr.exon_id.unique())))
    print("The number of unique transcript ids is " + str(len(gtf_exons_gr.transcript_id.unique())))
    print("The number of unique gene ids is " + str(len(gtf_exons_gr.gene_id.unique())))
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return(gtf_exons_gr)

def filter_junctions_by_shared_splice_sites(df):
    """
    Filter junctions by shared splice sites.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    # Function to apply to each group (cluster)
    def filter_group(group):

        # Find duplicated start and end positions within the group
        duplicated_starts = group['Start'].duplicated(keep=False)
        duplicated_ends = group['End'].duplicated(keep=False)
        
        # Keep rows where either start or end position is duplicated (this results in at least two junctions in every cluster)
        return group[duplicated_starts | duplicated_ends]
    
    # Group by 'Cluster' and apply the filtering function
    filtered_df = df.groupby('Cluster').apply(filter_group).reset_index(drop=True)
    return filtered_df.Cluster.unique()

def read_single_junction_file(junc_file, min_intron=50, max_intron=500000, sequencing_type="single_cell"):
    """
    Read a single junction file and return a dictionary of junctions.
    """
    try:
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
        juncs["intron_length"] = juncs["chromEnd"] - juncs["chromStart"]

        # Filter based on intron length and chromosomes
        juncs = juncs[(juncs["intron_length"] >= min_intron) & (juncs["intron_length"] <= max_intron)]
        standard_chromosomes_pattern = r'^(?:chr)?(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)$'
        juncs = juncs[juncs['chrom'].str.match(standard_chromosomes_pattern)]

        # Create a junction_id (including strand)
        juncs['junction_id'] = juncs['chrom'] + '_' + juncs['chromStart'].astype(str) + '_' + juncs['chromEnd'].astype(str) + '_' + juncs['strand']

        # Create a dictionary of junction counts and appearances
        junc_dict = {}
        for _, row in juncs.iterrows():
            junction_id = row['junction_id']
            score = row['score']
            if junction_id not in junc_dict:
                junc_dict[junction_id] = {'cells': 0, 'total_score': 0}
            junc_dict[junction_id]['cells'] += 1
            junc_dict[junction_id]['total_score'] += score

        return junc_dict

    except Exception as e:
        logging.error(f"Could not read in {junc_file}: {e}")
        return None
    
def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        dict1[key]['cells'] += value['cells']
        dict1[key]['total_score'] += value['total_score']

# Define a function to replace the lambda
def junction_default_dict():
    return {'cells': 0, 'total_score': 0}

def process_junction_files_chunk(file_chunk, min_intron, max_intron, sequencing_type):
    batch_dict = defaultdict(junction_default_dict)
    
    for junc_file in file_chunk:
        junc_dict = read_single_junction_file(junc_file, min_intron, max_intron, sequencing_type)
        if junc_dict:
            for junction_id, data in junc_dict.items():
                batch_dict[junction_id]['cells'] += data['cells']
                batch_dict[junction_id]['total_score'] += data['total_score']
    
    return batch_dict

# Main function to process junction files in parallel
def read_junction_files_parallel(junc_files, min_intron=50, max_intron=500000, sequencing_type="single_cell", max_workers=10, batch_size=100):

    logging.info(f"Found {len(junc_files)} junction files to process.")
    
    # Split the files into batches
    batched_sampled = [junc_files[i:i + batch_size] for i in range(0, len(junc_files), batch_size)]

    logging.info(f"Number of chunks to process: {len(batched_sampled)}")
    logging.info(f"Processing junction files in batches of {batch_size}!")
    
    junction_summary = defaultdict(junction_default_dict)

    # Use ProcessPoolExecutor to process batches in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_chunk in batched_sampled:
            futures.append(executor.submit(process_junction_files_chunk, file_chunk, min_intron, max_intron, sequencing_type))

        # Merge the results as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Merging results"):
            batch_result = future.result()
            merge_dicts(junction_summary, batch_result)

    print(f"Done merging all junction files!")
    return junction_summary

def convert_dict_to_dataframe(junction_summary):
    """
    Convert the junction summary dictionary into a pandas DataFrame.

    Parameters:
    -----------
    junction_summary (dict): The dictionary containing junction information.

    Returns:
    --------
    pd.DataFrame: A DataFrame with junction data.
    """
    df = pd.DataFrame.from_dict(junction_summary, orient='index').reset_index()
    df.columns = ['junction_id', 'num_cells_with_junc', 'total_read_counts']
    return df

def clean_up_juncs(all_juncs, min_num_cells_wjunc, min_junc_reads):
    
    # Step 1: Filter junctions based on min_num_cells_wjunc and min_junc_reads
    junctions_to_keep = all_juncs[(all_juncs['num_cells_with_junc'] >= min_num_cells_wjunc) & 
                                  (all_juncs['total_read_counts'] >= min_junc_reads)]
    
    # Step 2: Return the filtered DataFrame
    return junctions_to_keep

def mapping_juncs_exons(juncs_gr, gtf_exons_gr, singletons):
    print("Annotating junctions with known exons based on input gtf file")
    
    # For each junction, the start of the junction should equal end of exons and end of junction should equal start of exon 
    juncs_gr = juncs_gr.k_nearest(gtf_exons_gr, strandedness="same", ties="different", k=2, overlap=False)
    # Ensure distance parameter is still 1 
    juncs_gr = juncs_gr[abs(juncs_gr.Distance) == 1]

    # Group juncs_gr by gene_id and ensure that each junction has Start and End aligning with at least one End_b and Start_b respectively
    grouped_gr = juncs_gr.df.groupby("gene_id")
    juncs_keep = []
    for name, group in grouped_gr:
        group = group[(group.Start.isin(group.End_b)) & (group.End.isin(group.Start_b))]
        # Save junctions that are found here after filtering for matching start and end positions
        juncs_keep.append(group.junction_id.unique())

    # Flatten the list of lists
    juncs_keep = [item for sublist in juncs_keep for item in sublist]
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(juncs_keep)]
    
    print("The number of junctions after assessing distance to exons is " + str(len(juncs_gr.junction_id.unique())))
    if len(juncs_gr.junction_id.unique()) < 5000:
        print("There are less than 5000 junctions after assessing distance to exons. Please check your gtf file and ensure that it is in the correct format (start and end positions are not off by 1).", flush=True)
    
    print("Clustering intron splicing events by gene_id")
    juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id', 'gene_id']].drop_duplicate_positions()
    clusters = juncs_coords_unique.cluster(by="gene_id", slack=-1, count=True)
    print("The number of clusters after clustering by gene_id is " + str(len(clusters.Cluster.unique()))) 

    if not singletons:
        # Remove singletons 
        clusters = clusters[clusters.Count > 1]
        # Update juncs_gr to only include junctions that are part of clusters
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
        # Update juncs_coords_unique to only include junctions that are part of clusters
        juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
        print("The number of junctions after removing singletons is " + str(len(juncs_coords_unique.junction_id.unique())))
        print("The number of clusters after removing singletons is " + str(len(clusters.Cluster.unique()))) 
        return juncs_gr, juncs_coords_unique, clusters
    else:
        return juncs_gr, juncs_coords_unique, clusters

def basepair_to_kilobase(bp):
    return bp / 1000  # Convert base pairs to kilobases

def refine_clusters(clust_info):
    # For all start positions that are same for each cluster get the sum total_read_counts
    clust_info_5ss = clust_info.groupby(['Cluster', 'Start']).agg({'total_read_counts': 'sum'}).reset_index()
    clust_info_3_ss = clust_info.groupby(['Cluster', 'End']).agg({'total_read_counts': 'sum'}).reset_index()
    
    # Rename columns in 5ss to be total 5ss counts
    clust_info_5ss.rename(columns={'total_read_counts': 'total_5ss_counts'}, inplace=True)
    clust_info_3_ss.rename(columns={'total_read_counts': 'total_3ss_counts'}, inplace=True)
    
    # Remove Start and End column from each
    clust_info = clust_info.merge(clust_info_5ss, on=['Cluster', 'Start'])
    clust_info = clust_info.merge(clust_info_3_ss, on=['Cluster', 'End'])

    # Give each junction a 5ss fraction and 3ss fraction and then add column total_read_counts 
    clust_info['5SS_usage'] = clust_info['total_read_counts'] / clust_info['total_5ss_counts']
    clust_info['3SS_usage'] = clust_info['total_read_counts'] / clust_info['total_3ss_counts']
    clust_info["min_usage"] = clust_info[["5SS_usage", "3SS_usage"]].min(axis=1)
    print("Done refining clusters!")
    return(clust_info)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
#        Run analysis and obtain intron clusters
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main(junc_files, gtf_file, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, min_num_cells_wjunc, filter_shared_ss, max_workers, batch_size, run_notebook):
    # 1. Check format of junc_files and convert to list if necessary
    if isinstance(junc_files, list):
        pass
    elif "," in junc_files:
        junc_files = junc_files.split(",")
    else:
        junc_files = [junc_files]

    # 2. Convert parameters to integers outside the loop
    min_intron = int(min_intron)
    max_intron = int(max_intron)
    min_junc_reads = int(min_junc_reads)
    min_num_cells_wjunc = int(min_num_cells_wjunc)

    # 3. Run read_junction_files function to read in all junction files
    all_juncs = read_junction_files_parallel(junc_files, min_intron, max_intron, sequencing_type, max_workers=max_workers, batch_size=batch_size)
    all_juncs_df = convert_dict_to_dataframe(all_juncs)

    # 3. Clean up junctions (filter by min_num_cells_wjunc, min_junc_reads requirements!)
    all_juncs_clean = clean_up_juncs(all_juncs_df, min_num_cells_wjunc, min_junc_reads)

    # Step 1: Split the 'junction_id' column into 'chrom', 'chromStart', 'chromEnd', and 'strand'
    all_juncs_clean[['chrom', 'chromStart', 'chromEnd', 'strand']] = all_juncs_clean['junction_id'].str.split('_', expand=True)
    all_juncs_clean['chrom'] = all_juncs_clean['chrom'].str.replace('chr', '', regex=False)

    # Step 2: Convert 'chromStart' and 'chromEnd' to integers (they are strings after splitting)
    all_juncs_clean['chromStart'] = all_juncs_clean['chromStart'].astype(int)
    all_juncs_clean['chromEnd'] = all_juncs_clean['chromEnd'].astype(int)

    # 4. Make gr object from ALL junctions across all cell types add also total counts and number of cells with junction --> num_cells_with_junc  total_read_counts
    juncs_gr = pr.from_dict({
        "Chromosome": all_juncs_clean["chrom"],
        "Start": all_juncs_clean["chromStart"],
        "End": all_juncs_clean["chromEnd"],
        "Strand": all_juncs_clean["strand"],
        "num_cells_with_junc": all_juncs_clean["num_cells_with_junc"],
        "total_read_counts": all_juncs_clean["total_read_counts"],
        "junction_id": all_juncs_clean["junction_id"]
    })

    # 7. If gtf_file is not empty, read it in and process it
    if gtf_file is not None:
        gtf_exons_gr = process_gtf(gtf_file)
        print("Done extracting exons from gtf file")
    else:
        pass
    
    # 8. Annotate junctions based on gtf file (if gtf_file is not empty)
    if gtf_file is not None:
        juncs_gr, juncs_coords_unique, clusters = mapping_juncs_exons(juncs_gr, gtf_exons_gr, singleton) 
    else:
        print("Clustering intron splicing events by coordinates")
        juncs_coords_unique = juncs_gr[['Chromosome', 'Start', 'End', 'Strand', 'junction_id']].drop_duplicate_positions()
        clusters = juncs_coords_unique.cluster(slack=-1, count=True)
        print("The number of clusters after clustering by coordinates is " + str(len(clusters.Cluster.unique())))
        if not singleton:
            clusters = clusters[clusters.Count > 1]
            # Update juncs_gr to include only clusters that are in clusters
            juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
            juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
            print("The number of clusters after removing singletons is " + str(len(clusters.Cluster.unique())))

    print("The number of junctions after gtf file mapping " + str(len(juncs_coords_unique.junction_id.unique())))

    # 9. Now for each cluster we want to check that each junction shares a splice site with at least one other junction in the cluster
    if filter_shared_ss:
        clusts_keep = filter_junctions_by_shared_splice_sites(clusters.df)
        # Update clusters, juncs_gr, and juncs_coords_unique to only include clusters
        clusters = clusters[clusters.Cluster.isin(clusts_keep)]
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]

    juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
    
    print("The number of clusters after filtering for shared splice sites is " + str(len(clusters.Cluster.unique())))
    print("The number of junctions after filtering for shared splice sites is " + str(len(juncs_coords_unique.junction_id.unique())))

    # 10. Update our all_juncs file to only include junctions that are part of clusters
    all_juncs_df = all_juncs_df[all_juncs_df.junction_id.isin(juncs_coords_unique.junction_id)]

    # 11. Refine intron clusters based on splice sites found in them
    print("Refining intron clusters to account for junction usage ratio threshold...")
    juncs_counts = juncs_gr.df[['junction_id', 'Start', 'End', "total_read_counts"]].drop_duplicates()
    clust_info = clusters.df[['Cluster', 'junction_id']].drop_duplicates()
    clust_info = clust_info.merge(juncs_counts)
    junc_scores_all = refine_clusters(clust_info)
    junc_scores_all = junc_scores_all[junc_scores_all.min_usage >= threshold_inc]
    # Add 5ss and 3ss usage of each junction to all_juncs
    all_juncs_df = all_juncs_df.merge(junc_scores_all[['junction_id', 'total_5ss_counts', 'total_3ss_counts', "5SS_usage", "3SS_usage"]], on='junction_id')

    # Remove junctions that are in junc_scores_all from juncs_gr, clusters, all_juncs and juncs_coords_unique
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(junc_scores_all.junction_id)]
    clusters = clusters[clusters.junction_id.isin(junc_scores_all.junction_id)]
    all_juncs_df = all_juncs_df[all_juncs_df.junction_id.isin(junc_scores_all.junction_id)]
    juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(junc_scores_all.junction_id)]
    print("The number of clusters after removing low confidence junctions is " + str(len(clusters.Cluster.unique())))

    # 12. Re-cluster introns after low confidence junction removal
    print("Reclustering intron splicing events after low confidence junction removal")
    juncs_gr = juncs_gr.drop_duplicate_positions()
    clusters = juncs_gr.cluster(by="gene_id", slack=-1, count=True)
        
    # 13. Remove singletons if there are new ones 
    if not singleton:
        clusters = clusters[clusters.Count > 1]
        # Update juncs_gr and juncs_coords_unique
        juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
        juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
        print("The number of clusters after removing singletons is " + str(len(clusters.Cluster.unique())))

    # 14. Confirm that junctions in each cluster share splice sites
    print("Confirming that junctions in each cluster share splice sites")
    clusts_keep = filter_junctions_by_shared_splice_sites(clusters.df)
    # Update clusters, juncs_gr, and juncs_coords_unique to only include clusters
    clusters = clusters[clusters.Cluster.isin(clusts_keep)]
    juncs_gr = juncs_gr[juncs_gr.junction_id.isin(clusters.junction_id)]
    juncs_coords_unique = juncs_coords_unique[juncs_coords_unique.junction_id.isin(clusters.junction_id)]
    all_juncs_df = all_juncs_df[all_juncs_df.junction_id.isin(juncs_coords_unique.junction_id)]
    print("The number of clusters after filtering for shared splice sites is " + str(len(clusters.Cluster.unique())))

    if gtf_file is not None:
        clusts_unique = clusters.df[["Cluster", "junction_id", "gene_id", "gene_name", "Count"]].drop_duplicates()
        juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id", "Start_b", "End_b", "gene_id", "gene_name", "transcript_id", "exon_id"]]
        juncs_gr = juncs_gr.drop_duplicate_positions()
        juncs_gr.to_bed(junc_bed_file, chain=True)
        print("Saved final list of junction coordinates to " + junc_bed_file)
    else:
        clusts_unique = clusters.df[["Cluster", "junction_id", "Count"]].drop_duplicates()
        juncs_gr = juncs_gr[["Chromosome", "Start", "End", "Strand", "junction_id"]]
        juncs_gr = juncs_gr.drop_duplicate_positions()
        juncs_gr.to_bed(junc_bed_file, chain=True)
        print("Saved final list of junction coordinates to " + junc_bed_file)
    
    # Merge juncs_gr with corresponding cluster id
    all_juncs_df = all_juncs_df.merge(clusts_unique, how="left")

    # Get final list of junction coordinates and save to bed file for visualization
    print("The number of clusters to be finally evaluated is " + str(len(all_juncs_df.Cluster.unique()))) 
    print("The number of junctions to be finally evaluated is " + str(len(all_juncs_df.junction_id.unique())))

    # Assert unique number of junctions and clusters in all_juncs_df and clusters_df is the same 
    assert len(all_juncs_df.junction_id.unique()) == len(clusters.df.junction_id.unique())
    assert len(all_juncs_df.Cluster.unique()) == len(clusters.df.Cluster.unique()) 
    
    # 15. Save the final list of intron clusters to a file
    date = time.strftime("%Y%m%d")
    output = output_file + "_" + str(min_intron) + "_" + str(max_intron) + "_" + str(min_junc_reads) + "_" + date + "_" + str(sequencing_type)
    output_file_name = output + '.gz'  # Construct the full output file name
    
    with gzip.open(output_file_name, mode='wt', encoding='utf-8') as f:
        all_juncs_df.to_csv(f, index=False, sep="}")
    
    print("You can find the output file here: " + output_file_name)
    print("Finished obtaining intron cluster files!")
    
    # Return the final list of intron clusters if running in notebook
    if run_notebook:
        return all_juncs_df
    else:
        return output_file_name  # Return the output file name

if __name__ == '__main__':
    gtf_file=args.gtf_file
    junc_files=args.junc_files
    output_file=args.output_file
    sequencing_type=args.sequencing_type
    junc_bed_file=args.junc_bed_file
    threshold_inc = float(args.threshold_inc)
    min_intron=args.min_intron_length
    max_intron=args.max_intron_length
    min_junc_reads=args.min_junc_reads
    junc_suffix=args.junc_suffix
    min_num_cells_wjunc=args.min_num_cells_wjunc
    filter_shared_ss=bool(args.filter_shared_ss)
    max_workers = args.max_workers
    batch_size = args.batch_size
    singleton = args.keep_singletons == "True"
    run_notebook = args.run_notebook
    
    # Print out all user defined arguments that were chosen 
    print("The following arguments were chosen:")
    print("gtf_file: " + str(gtf_file))
    print("junc_files: " + junc_files)
    print("output_file: " + output_file)
    print("sequencing_type: " + sequencing_type)
    print("junc_bed_file: " + junc_bed_file)
    print("threshold_inc: " + str(threshold_inc))
    print("min_intron: " + str(min_intron))
    print("max_intron: " + str(max_intron))
    print("min_junc_reads: " + str(min_junc_reads))
    print("junc_suffix: " + junc_suffix)
    print("min_num_cells_wjunc: " + str(min_num_cells_wjunc))
    print("singleton: " + str(singleton))
    print("shared_ss: " + str(filter_shared_ss))
    print(f"Max workers: {max_workers}")
    print(f"Batch size: {batch_size}")

    main(junc_files, gtf_file, output_file, sequencing_type, junc_bed_file, threshold_inc, min_intron, max_intron, min_junc_reads, singleton, junc_suffix, min_num_cells_wjunc, filter_shared_ss, run_notebook)
