#!/usr/bin/env python3
import pandas as pd
from typing import Dict, List, Tuple
import networkx as nx
from tqdm import tqdm
from pyfaidx import Fasta
import gffutils
import logging
import os 
import time
import sqlite3
from collections import defaultdict
import networkx as nx
import random 
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

class JunctionReader:
    def __init__(self, 
                min_intron: int = 50, 
                max_intron: int = 500000, 
                sequencing_type: str = "single_cell", 
                batch_size: int = 10,
                min_cells: int = 2,
                min_reads: int = 100, 
                num_workers: int = 4):
        
        self.min_intron = min_intron
        self.max_intron = max_intron
        self.sequencing_type = sequencing_type
        self.dtypes = {0: str, 1: 'int32', 2: 'int32', 3: str, 4: 'int32', 5: str,
                      6: 'int32', 7: 'int32', 8: str, 9: 'int32', 10: str, 11: str}
        self.batch_size = batch_size
        self.min_cells = min_cells
        self.min_reads = min_reads
        self.num_workers = num_workers
        self.combined_junctions = {}

    def parse_file(self, file_path: str) -> Dict[str, Dict]:
        """
        Read and process a junction file, returning a dictionary of junctions with cell information.

        Args:
            file_path (str): Path to the junction file

        Returns:
            Dict[str, Dict]: Dictionary containing junction information with cell-specific details
        """
        try:
            juncs = pd.read_csv(file_path, sep="\t", header=None, dtype=self.dtypes)

            col_names = ["chrom", "chromStart", "chromEnd", "name", "score", "strand",
                        "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
            if self.sequencing_type == "single_cell":
                col_names += ["num_cells_wjunc", "cell_readcounts"]
            juncs.columns = col_names[:len(juncs.columns)]

            juncs[['block_add_start', 'block_subtract_end']] = (
                juncs["blockSizes"].str.extract(r'(\d+),(\d+)').astype(int)
            )
            juncs["chromStart"] += juncs['block_add_start']
            juncs["chromEnd"] -= juncs['block_subtract_end']
            juncs["intron_length"] = juncs["chromEnd"] - juncs["chromStart"]

            juncs = juncs[
                (juncs["intron_length"] >= self.min_intron) & 
                (juncs["intron_length"] <= self.max_intron)
            ]
            standard_chromosomes_pattern = r'^(?:chr)?(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)$'
            juncs = juncs[juncs['chrom'].str.match(standard_chromosomes_pattern)]

            juncs['junction_id'] = (
                juncs['chrom'] + '_' + 
                juncs['chromStart'].astype(str) + '_' +
                juncs['chromEnd'].astype(str) + '_' + 
                juncs['strand']
            )

            junc_dict = {}
            for _, row in juncs.iterrows():
                junction_id = row['junction_id']
                if junction_id not in junc_dict:
                    junc_dict[junction_id] = {
                        'cells': 1,
                        'total_score': row['score'],
                        'chrom': row['chrom'],
                        'start': row['chromStart'],
                        'end': row['chromEnd'],
                        'strand': row['strand']
                    }
                else:
                    junc_dict[junction_id]['cells'] += 1
                    junc_dict[junction_id]['total_score'] += row['score']

            return junc_dict
        
        except Exception as e:
            logging.error(f"Could not read in {file_path}: {e}")
            return {}
        
    def process_files(self, file_list: List[str]) -> Dict[str, Dict]:
        """
        Process multiple junction files in parallel batches.

        Args:
            file_list: List of file paths

        Returns:
            Combined junction dictionary
        """
        def process_batch(batch_files):
            batch_junctions = {}
            for file_path in batch_files:
                try:
                    file_junctions = self.parse_file(file_path)
                    for j_id, j_data in file_junctions.items():
                        if j_id not in batch_junctions:
                            batch_junctions[j_id] = j_data
                        else:
                            batch_junctions[j_id]['cells'] += j_data['cells']
                            batch_junctions[j_id]['total_score'] += j_data['total_score']
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")
            return batch_junctions

        # Create batches
        batches = [file_list[i:i + self.batch_size] for i in range(0, len(file_list), self.batch_size)]
        total_batches = len(batches)

        print(f"\nProcessing {len(file_list)} files in {total_batches} batches")
        print(f"Using {self.num_workers} workers, batch size: {self.batch_size}\n")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]

            with tqdm(total=total_batches, desc="Processing batches") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_result = future.result()

                        # Merge results
                        for j_id, j_data in batch_result.items():
                            if j_id not in self.combined_junctions:
                                self.combined_junctions[j_id] = j_data
                            else:
                                self.combined_junctions[j_id]['cells'] += j_data['cells']
                                self.combined_junctions[j_id]['total_score'] += j_data['total_score']

                        pbar.update(1)

                    except Exception as e:
                        logging.error(f"Batch processing error: {str(e)}")
                        pbar.update(1)

        return self.combined_junctions

    def SJ_QC(self, junctions: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Filter junctions based on minimum cell and read requirements.

        Args:
            junctions: Dictionary of junctions
            min_cells: Minimum number of cells required
            min_reads: Minimum number of reads required

        Returns:
            Filtered junction dictionary
        """
        initial_count = len(junctions)

        # Filter based on cells and reads
        filtered_junctions = {
            j_id: j_data for j_id, j_data in junctions.items()
            if j_data['cells'] >= self.min_cells and j_data['total_score'] >= self.min_reads
        }

        # Log filtering results
        filtered_count = len(filtered_junctions)
        # To log also add which filters were used
        print(f"Minimum cells per junction to pass set to: {self.min_cells}")
        print(f"Minimum total reads per junction to pass set to: {self.min_reads}")
        print(f"Initial junctions before basic QC: {initial_count}")
        print(f"Filtered junctions: {filtered_count}")
        print(f"Removed junctions: {initial_count - filtered_count}")
        print(f"Percentage kept: {(filtered_count/initial_count)*100:.2f}%")

        return filtered_junctions

    def clear_combined_junctions(self):
        """Reset the combined junctions dictionary"""
        self.combined_junctions = {}

class JunctionAnalyzer:
    def __init__(self, fasta_file: str, db: gffutils.FeatureDB, window_size: int = 2, tolerance: int = 1):
        self.genome = Fasta(fasta_file)
        self.db = db
        self.window_size = window_size
        self.tolerance = tolerance
        self.canonical_motifs = {
            "GT-AG": ("GT", "AG"),
            "AT-AC": ("AT", "AC"),
            "GC-AG": ("GC", "AG")
        }

    @staticmethod
    def reverse_complement(seq: str) -> str:
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        return ''.join(complement.get(base, base) for base in reversed(seq))

    def get_splice_sequences(self, junction: Dict):
        chrom = self.check_chromosome(junction["chrom"])
        
        if junction["strand"] == "+":
            donor_seq = str(self.genome[chrom][junction["start"]:junction["start"]+self.window_size]).upper()
            acceptor_seq = str(self.genome[chrom][junction["end"]-self.window_size:junction["end"]]).upper()
        else:
            donor_seq = self.reverse_complement(str(self.genome[chrom][junction["end"]-self.window_size:junction["end"]]).upper())
            acceptor_seq = self.reverse_complement(str(self.genome[chrom][junction["start"]:junction["start"]+self.window_size]).upper())
            
        return donor_seq, acceptor_seq

    def check_chromosome(self, chrom: str) -> str:
        if chrom not in self.genome:
            return f"chr{chrom}" if not chrom.startswith("chr") else chrom.replace("chr", "")
        return chrom

    def get_motif_type(self, donor_seq: str, acceptor_seq: str) -> str:
        for motif_name, (donor, acceptor) in self.canonical_motifs.items():
            if donor_seq == donor and acceptor_seq == acceptor:
                return motif_name
        return "non-canonical"

    def check_splice_sites(self, junctions: Dict[str, Dict]) -> Dict[str, Dict]:
        """Add splice site annotations to junction dictionary"""
        for j_id, junction in junctions.items():
            donor_seq, acceptor_seq = self.get_splice_sequences(junction)
            junction["splice_motif"] = self.get_motif_type(donor_seq, acceptor_seq)
            junction["donor_seq"] = donor_seq
            junction["acceptor_seq"] = acceptor_seq
        return junctions

    def filter_canonical(self, junctions: Dict[str, Dict], only_canonical: bool = True) -> Dict[str, Dict]:
       """Filter junctions based on splice site motifs"""
       if not only_canonical:
           return junctions

       initial_count = len(junctions)
       motif_counts = {}
    
       for j_data in junctions.values():
           motif = j_data["splice_motif"]
           motif_counts[motif] = motif_counts.get(motif, 0) + 1
    
       canonical_junctions = {j_id: j_data for j_id, j_data in junctions.items() 
                            if j_data["splice_motif"] != "non-canonical"}
    
       filtered_count = len(canonical_junctions)
    
       print(f"Initial junctions before splice site filtering: {initial_count}")
       print("\nSplice site motif breakdown:")
       for motif, count in motif_counts.items():
           print(f"{motif}: {count} ({(count/initial_count)*100:.2f}%)")
       print(f"\nCanonical junctions kept: {filtered_count}")
       print(f"Non-canonical removed: {initial_count - filtered_count}")
       print(f"Percentage canonical: {(filtered_count/initial_count)*100:.2f}%")
    
       return canonical_junctions
    
    def check_junction_annotation(self, junctions: Dict[str, Dict]) -> Dict[str, Dict]:
        """Add annotation status for both ends of each junction with improved transcript mapping"""

        # Group junctions by chromosome to minimize database queries
        chrom_groups = {}
        for j_id, j in junctions.items():
            chrom_groups.setdefault(j["chrom"], []).append((j_id, j))

        for chrom, junc_group in tqdm(chrom_groups.items(), desc="Processing chromosomes"):
            # Get all relevant transcripts for this chromosome group
            min_pos = min(j["start"] for _, j in junc_group)
            max_pos = max(j["end"] for _, j in junc_group)

            # Single query for all transcripts in range
            transcripts = list(self.db.region(
                region=(chrom, min_pos - 1000, max_pos + 1000),
                featuretype="transcript"
            ))

            # Cache exon data and transcript data 
            exon_cache = {t.id: list(self.db.children(t, featuretype="exon", order_by="start"))
                         for t in transcripts}

            # Get gene info and transcripts
            gene_info = {}
            for t in transcripts:
                gene = list(self.db.parents(t, featuretype="gene"))[0]
                gene_info[t.id] = {
                    'gene_id': gene.id,
                    'gene_name': gene.attributes.get('gene_name', [None])[0],
                     'gene_type': gene.attributes.get('gene_biotype', 
                                  gene.attributes.get('gene_type', [None]))[0]
                }

            # Get trancsript types 
            transcript_types = {}
            for t in transcripts:
                transcript_types[t.id] = t.attributes.get('transcript_type', [None])[0]

            # Process each junction
            for j_id, junction in tqdm(junc_group, desc="Processing junctions"):
                start = junction["start"]
                end = junction["end"]
                strand = junction["strand"]

                # Initialize overall annotation labels
                label_5_prime, label_3_prime = "unannotated on 5'", "unannotated on 3'"
                position_off_5_prime, position_off_3_prime = None, None

                # Initialize transcript categorization
                both_ends_transcripts = set()
                only_5_prime_transcripts = set()
                only_3_prime_transcripts = set()

                # Track overall for all transcripts
                transcript_types_junc = set()
                genes_found = set()
                gene_types_found = set()  # New: Track gene types

                for transcript in transcripts:
                    exons = exon_cache[transcript.id]
                    found_5_prime = False
                    found_3_prime = False

                    if strand == "+":
                        # Check 5' end (start position)
                        for exon in exons:
                            if abs(exon.end - start) <= self.tolerance:
                                label_5_prime = "annotated on 5'"
                                position_off_5_prime = exon.end - start
                                break 
                        # Check 3' end (end position)
                        for exon in exons:
                            if abs(exon.start - end) <= self.tolerance:
                                label_3_prime = "annotated on 3'"
                                position_off_3_prime = exon.start - end
                                found_3_prime = True
                                break
                    else: # strand == "-"
                        # Check 5' end (end position for negative strand)
                        for exon in exons:
                            if abs(exon.start - end) <= self.tolerance:
                                label_5_prime = "annotated on 5'"
                                position_off_5_prime = exon.start - end
                                found_5_prime = True
                                break

                        # Check 3' end (start position for negative strand)
                        for exon in exons:
                            if abs(exon.end - start) <= self.tolerance:
                                label_3_prime = "annotated on 3'"
                                position_off_3_prime = exon.end - start
                                found_3_prime = True
                                break

                    # Add gene and transcript info if any end matches
                    if found_5_prime or found_3_prime:
                        if transcript.id in transcript_types:
                            transcript_types_junc.add(transcript_types[transcript.id])
                        if transcript.id in gene_info:
                            gene_data = gene_info[transcript.id]
                            genes_found.add((gene_data['gene_id'], gene_data['gene_name']))
                            if gene_data['gene_type']:
                                gene_types_found.add(gene_data['gene_type'])

                    # Categorize transcripts based on which junction ends they match
                    if found_5_prime and found_3_prime:
                        both_ends_transcripts.add(transcript.id)
                    elif found_5_prime:
                        only_5_prime_transcripts.add(transcript.id)
                    elif found_3_prime:
                        only_3_prime_transcripts.add(transcript.id)

                # Get all transcripts that match at least one end
                all_matching_transcripts = both_ends_transcripts.union(only_5_prime_transcripts, only_3_prime_transcripts)

                junction.update({
                    "label_5_prime": label_5_prime,
                    "label_3_prime": label_3_prime,
                    "position_off_5_prime": position_off_5_prime,
                    "position_off_3_prime": position_off_3_prime,
                    "transcripts": list(all_matching_transcripts),
                    "both_ends_transcripts": list(both_ends_transcripts),
                    "only_5_prime_transcripts": list(only_5_prime_transcripts),
                    "only_3_prime_transcripts": list(only_3_prime_transcripts),
                    "transcript_types": list(transcript_types_junc),
                    "gene_ids": [g[0] for g in genes_found],
                    "gene_names": [g[1] for g in genes_found],
                    "gene_types": list(gene_types_found)  # Added gene types to output
                })

        return junctions
    
    def filter_annotated(self, junctions: Dict[str, Dict], annotation_status_include: str = 'both') -> Dict[str, Dict]:
        """
        Filter junctions based on annotation status.
    
        Args:
            junctions: Dictionary of junctions
            annotation_status_include: Filtering option for junctions
                - 'both': Keep only junctions where both ends are annotated
                - 'either': Keep junctions where at least one end is annotated
                - 'unanno_also': Keep all junctions regardless of annotation

        Returns:
            Filtered junction dictionary
        """
        if annotation_status_include not in ['both', 'either', 'unanno_also']:
            raise ValueError("annotation_status_include must be one of: 'both', 'either', 'unanno_also'")
        
        initial_count = len(junctions)

        annotation_stats = {
        "both_ends": 0,
        "five_prime_only": 0,
        "three_prime_only": 0,
        "unannotated": 0,
        "multi_gene": 0
        }

        filtered_junctions = {}
        multi_gene_junctions = {}

        # Process each junction
        for j_id, j_data in junctions.items():
            
            # Initialize flags
            five_prime = False
            three_prime = False

            # Check position off for 5' and 3' ends if not None
            if j_data["position_off_5_prime"] is not None:
                five_prime = (0 <= j_data["position_off_5_prime"] <= 1)

            if j_data["position_off_3_prime"] is not None:
                three_prime = (0 <= j_data["position_off_3_prime"] <= 1)
    
            # Handle multi-gene junctions
            if len(j_data["gene_ids"]) > 1:
                annotation_stats["multi_gene"] += 1
                multi_gene_junctions[j_id] = j_data
                continue

            # Categorize junction
            if five_prime and three_prime:
                annotation_stats["both_ends"] += 1
                category = "both"
            elif five_prime:
                annotation_stats["five_prime_only"] += 1
                category = "five_prime"
            elif three_prime:
                annotation_stats["three_prime_only"] += 1
                category = "three_prime"
            else:
                annotation_stats["unannotated"] += 1
                category = "unannotated"

            # Apply filtering based on annotation_status_include
            keep_junction = False
            if annotation_status_include == 'both':
                keep_junction = (category == "both")
            elif annotation_status_include == 'either':
                keep_junction = (category in ["both", "five_prime", "three_prime"])
            elif annotation_status_include == 'unanno_also':
                keep_junction = True

            if keep_junction:
                # Add junction to filtered dictionary and include its category
                j_data["annotation_status"] = category
                filtered_junctions[j_id] = j_data

        filtered_count = len(filtered_junctions)
        
        # Save multi-gene junctions to file
        if multi_gene_junctions:
            multi_gene_file = f'multi_gene_junctions_{time.strftime("%Y%m%d-%H%M%S")}.csv'
            pd.DataFrame.from_dict(multi_gene_junctions, orient='index').to_csv(multi_gene_file)
            print(f"\nMulti-gene junctions saved to {multi_gene_file}")

        # Print statistics
        print(f"\nInitial junctions: {initial_count}")
        print("\nAnnotation breakdown:")
        for status, count in annotation_stats.items():
            print(f"{status}: {count} ({(count/initial_count)*100:.2f}%)")

        print(f"\nFiltering summary:")
        print(f"Annotation status filter: {annotation_status_include}")
        print(f"Junctions kept: {filtered_count}")
        print(f"Junctions removed: {initial_count - filtered_count}")
        print(f"Percentage kept: {(filtered_count/initial_count)*100:.2f}%")

        return filtered_junctions

class GenomeDB:
    def __init__(self, db_name: str, gtf_file: str = None, fasta_file: str = None, max_retries: int = 3):
        self.db_name = db_name
        self.gtf_file = gtf_file
        self.fasta_file = fasta_file
        self.genome = None
        self.db = None
        
        if self.fasta_file:
            print(f"Loading genome from {self.fasta_file}")
            self.genome = Fasta(self.fasta_file)
            
        if self.gtf_file and self.db_name:
            if not os.path.exists(self.db_name):
                print(f"Creating database {self.db_name} from {self.gtf_file}")
                self.db = self.create_db()
            else:
                print(f"Loading existing database {self.db_name}")
                for attempt in range(max_retries):
                    try:
                        self.db = gffutils.FeatureDB(self.db_name)
                        break
                    except sqlite3.OperationalError as e:
                        if attempt == max_retries - 1:
                            raise e
                        print(f"Database locked, retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(1)  # Wait before retry

    def create_db(self) -> gffutils.FeatureDB:
        return gffutils.create_db(
            self.gtf_file,
            dbfn=self.db_name,
            force=True,
            keep_order=True,
            merge_strategy="merge",
            sort_attribute_values=True,
            disable_infer_genes=True,
            disable_infer_transcripts=True,
        )
        
    def get_db(self) -> gffutils.FeatureDB:
        return self.db
    
# Class for ATSE mapping or junction grouping 
# Nodes are splice sites (donor/acceptor)
# Edges represent junctions
# Connected components form ATSEs
# Event classification based on graph topology
class ATSEAnalyzer:
    def __init__(self):
        self.events = {}
        
    def build_splice_graph(self, junctions: Dict[str, Dict]):
        gene_graphs = {}
        gene_groups = defaultdict(dict)

        # Track statistics
        stats = {
            'total_junctions': len(junctions),
            'included_junctions': 0,
            'excluded_no_gene': 0,
            'junctions_per_gene': defaultdict(int),
            'single_junction_genes': set()  # Track genes with only one junction
        }

        # First pass - group junctions by gene
        for j_id, j_data in junctions.items():
            if not j_data['gene_ids']:
                stats['excluded_no_gene'] += 1
                continue
            gene_id = j_data['gene_ids'][0]
            gene_groups[gene_id][j_id] = j_data
            stats['included_junctions'] += 1

        # Build graphs and track single-junction genes
        for gene_id, gene_juncs in gene_groups.items():
            G = nx.Graph()

            for j_id, j_data in gene_juncs.items():
                if j_data['strand'] == '+':
                    donor = (j_data['chrom'], j_data['start'], 'donor')
                    acceptor = (j_data['chrom'], j_data['end'], 'acceptor')
                else:
                    donor = (j_data['chrom'], j_data['end'], 'donor')
                    acceptor = (j_data['chrom'], j_data['start'], 'acceptor')

                G.add_edge(donor, acceptor,
                          junction_id=j_id,
                          strand=j_data['strand'],
                          score=j_data['total_score'])

            gene_graphs[gene_id] = G
            num_junctions = len(gene_juncs)
            stats['junctions_per_gene'][gene_id] = num_junctions

            if num_junctions == 1:
                stats['single_junction_genes'].add(gene_id)

        # Calculate additional statistics
        stats['num_genes'] = len(gene_graphs)
        stats['num_single_junction_genes'] = len(stats['single_junction_genes'])
        stats['num_analyzable_genes'] = stats['num_genes'] - stats['num_single_junction_genes']
        stats['single_junction_pairs'] = stats['num_single_junction_genes']
        stats['avg_junctions_per_gene'] = stats['included_junctions'] / len(gene_graphs) if gene_graphs else 0
        stats['max_junctions_in_gene'] = max(stats['junctions_per_gene'].values()) if stats['junctions_per_gene'] else 0
        stats['min_junctions_in_gene'] = min(stats['junctions_per_gene'].values()) if stats['junctions_per_gene'] else 0

        # Print summary automatically
        print(f"""Splice Graph Building Summary:
    Total junctions: {stats['total_junctions']}
    - Included in graphs: {stats['included_junctions']}
    - Excluded (no gene): {stats['excluded_no_gene']}

    Gene Statistics:
    - Total genes: {stats['num_genes']}
    - Genes with single junction (can't analyze): {stats['num_single_junction_genes']} ({stats['num_single_junction_genes']/stats['num_genes']*100:.1f}% of genes)
    - Analyzable genes (>1 junction): {stats['num_analyzable_genes']} ({stats['num_analyzable_genes']/stats['num_genes']*100:.1f}% of genes)

    Junction Distribution:
    - Average junctions per gene: {stats['avg_junctions_per_gene']:.1f}
    - Max junctions in a gene: {stats['max_junctions_in_gene']}
    - Min junctions in a gene: {stats['min_junctions_in_gene']}""")

        return gene_graphs, stats

    def find_connected_junctions(self, G: nx.Graph, start_junction_id: str, visited: set) -> set:
        """
        Find all junctions connected by shared splice sites, including indirect connections.
        Returns empty set if only one junction is found.
        """
        # Get start junction's nodes
        start_nodes = []
        for u, v, data in G.edges(data=True):
            if data['junction_id'] == start_junction_id:
                start_nodes = [u, v]
                break
            
        if not start_nodes:
            return set()

        # Find all nodes in the connected component
        component_nodes = set(nx.node_connected_component(G, start_nodes[0]))

        # Get all junctions in this component
        connected = set()
        for u, v, data in G.edges(data=True):
            if u in component_nodes and v in component_nodes:
                junction_id = data['junction_id']
                if junction_id not in visited:
                    connected.add(junction_id)

        return connected if len(connected) > 1 else set()

    def analyze_splice_sites(self, G: nx.Graph) -> Dict:
        """
        Analyze splice site connectivity in a graph.
        """
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'donor_sites': len([n for n in G.nodes() if n[2] == 'donor']),
            'acceptor_sites': len([n for n in G.nodes() if n[2] == 'acceptor'])
        }

        # Analyze node degrees (how many junctions share each splice site)
        degrees = [d for _, d in G.degree()]
        stats['max_degree'] = max(degrees) if degrees else 0
        stats['isolated_sites'] = degrees.count(1)

        return stats

    def analyze_singleton_junctions(self, G: nx.Graph, junction_id: str) -> Dict:
        """Analyze a singleton junction for its properties."""
        junction_info = {}

        for u, v, data in G.edges(data=True):
            if data['junction_id'] == junction_id:
                junction_info = {
                    'junction_id': junction_id,
                    'donor_site': u[1],
                    'acceptor_site': v[1],
                    'strand': data['strand'],
                    'score': data['score'],  # number of reads
                    'chromosome': u[0]
                }
                break

        return junction_info
    
    def sample_unincluded_junctions(self, graphs: Dict[str, nx.Graph], included_junctions: set, sample_size: int = 500) -> List[Dict]:
        """
        Sample unincluded junctions and verify they don't share splice sites.
        Returns list of any problematic cases found.
        """
        all_unincluded = []
        for gene_id, G in graphs.items():
            gene_junctions = {data['junction_id'] for _, _, data in G.edges(data=True)}
            unincluded = gene_junctions - included_junctions
            all_unincluded.extend((gene_id, j_id) for j_id in unincluded)

        # Sample junctions
        if len(all_unincluded) > sample_size:
            sampled = random.sample(all_unincluded, sample_size)
        else:
            sampled = all_unincluded

        problematic = []

        # Check each sampled junction
        for gene_id, junction_id in sampled:
            G = graphs[gene_id]

            # Get junction's splice sites
            current_donor = None
            current_acceptor = None
            for u, v, data in G.edges(data=True):
                if data['junction_id'] == junction_id:
                    current_donor = u[1]  # Just the coordinate
                    current_acceptor = v[1]
                    break
                
            if current_donor is None or current_acceptor is None:
                print(f"Warning: Could not find coordinates for {junction_id}")
                continue
            
            # Check all other junctions in the same gene
            for u, v, data in G.edges(data=True):
                other_id = data['junction_id']
                if other_id != junction_id:
                    other_donor = u[1]
                    other_acceptor = v[1]

                    # Check for exact coordinate matches
                    if (current_donor == other_donor or 
                        current_donor == other_acceptor or
                        current_acceptor == other_donor or 
                        current_acceptor == other_acceptor):

                        # For debugging, print the exact match found
                        problematic.append({
                            'gene_id': gene_id,
                            'junction_id': junction_id,
                            'junction_coords': (current_donor, current_acceptor),
                            'shares_site_with': other_id,
                            'other_coords': (other_donor, other_acceptor),
                            'shared_coord': current_donor if (current_donor == other_donor or current_donor == other_acceptor) else current_acceptor
                        })

        if problematic:
            print("\nDetailed analysis of problematic cases:")
            for case in problematic[:5]:
                print(f"\nJunction: {case['junction_id']} ({case['junction_coords']})")
                print(f"Shares with: {case['shares_site_with']} ({case['other_coords']})")
                print(f"Shared coordinate: {case['shared_coord']}")

        return problematic
    

    def find_atse_groups(self, graphs: Dict[str, nx.Graph], min_splice_site_usage: float = 0.01) -> Dict[str, Dict]:
        """
        Find alternative transcript splicing events (ATSEs) with splice site usage filtering.
    
        Args:
            graphs: Dictionary of gene graphs
            min_splice_site_usage: Minimum proportion of reads a junction must have at a splice site
                               compared to total reads at that site (default: 0.01 or 1%)
    
        Returns:
            Dictionary of ATSE groups and sorted counts
        """
        atse_groups = {}
        event_counter = 0
        filtered_junctions = 0  # Track filtered junctions
    
        # Statistics tracking
        stats = {
            'total_junctions': sum(G.number_of_edges() for G in graphs.values()),
            'analyzable_junctions': 0,
            'junctions_in_atses': set(),
            'junction_counts': defaultdict(int),
            'singleton_junctions': [],  # Store info about singleton junctions
            'filtered_junctions': []    # Track junctions filtered due to low splice site usage
        }

        # First count analyzable junctions
        for gene_id, G in graphs.items():
            num_junctions = G.number_of_edges()
            if num_junctions >= 2:
                stats['analyzable_junctions'] += num_junctions

        # Find ATSEs
        for gene_id, G in graphs.items():
            visited = set()

            for _, _, data in G.edges(data=True):
                junction_id = data['junction_id']
                if junction_id not in visited:
                    connected = self.find_connected_junctions(G, junction_id, visited)

                    if connected:
                        # Get all splice sites in this connected component
                        splice_sites = set()
                        junction_data = {}

                        # First pass: collect all splice sites and junction data
                        for j_id in connected:
                            for u, v, data in G.edges(data=True):
                                if data['junction_id'] == j_id:
                                    splice_sites.add(u)
                                    splice_sites.add(v)
                                    junction_data[j_id] = {
                                    'donor': u,
                                    'acceptor': v,
                                    'score': data['score'],
                                    'strand': data['strand']
                                    }
                        
                        # Calculate total reads at each splice site
                        site_total_reads = defaultdict(int)
                        for j_id, j_data in junction_data.items():
                            site_total_reads[j_data['donor']] += j_data['score']
                            site_total_reads[j_data['acceptor']] += j_data['score']

                        # Calculate usage proportions and filter junctions with low splice site usage
                        filtered_out = set()
                        junction_usage = {}

                        for j_id, j_data in junction_data.items():
                            donor_usage = j_data['score'] / site_total_reads[j_data['donor']]
                            acceptor_usage = j_data['score'] / site_total_reads[j_data['acceptor']]

                            # store the usage values for each junction 
                            if j_data['strand'] == '+':
                                # For positive strand, donor is 5' and acceptor is 3'
                                five_prime_usage = donor_usage
                                three_prime_usage = acceptor_usage
                            else:
                                # For negative strand, acceptor is 5' and donor is 3'
                                five_prime_usage = acceptor_usage
                                three_prime_usage = donor_usage

                            junction_usage[j_id] = {
                                'five_prime_usage': five_prime_usage,
                                'three_prime_usage': three_prime_usage,
                                'donor_usage': donor_usage,
                                'acceptor_usage': acceptor_usage,
                                'donor_total_reads': site_total_reads[j_data['donor']],
                                'acceptor_total_reads': site_total_reads[j_data['acceptor']]
                            }
                        
                            # If either the donor or acceptor usage is too low, filter out the junction
                            if donor_usage < min_splice_site_usage or acceptor_usage < min_splice_site_usage:
                                filtered_out.add(j_id)
                                stats['filtered_junctions'].append({
                                    'gene_id': gene_id,
                                    'junction_id': j_id,
                                    'five_prime_usage': five_prime_usage,
                                    'three_prime_usage': three_prime_usage,
                                    'donor_usage': donor_usage,
                                    'acceptor_usage': acceptor_usage,
                                    'donor_total_reads': site_total_reads[j_data['donor']],
                                    'acceptor_total_reads': site_total_reads[j_data['acceptor']],
                                    'junction_reads': j_data['score']
                                })
                    
                        # Remove filtered junctions
                        filtered_connected = connected - filtered_out
                        filtered_junctions += len(filtered_out)

                        # Only create an ATSE if at least 2 junctions remain after filtering
                        if len(filtered_connected) >= 2:
                            event_id = f"ATSE_{event_counter}"
                        
                            # Recalculate splice sites based on filtered junctions
                            filtered_splice_sites = set()
                            for j_id in filtered_connected:
                                filtered_splice_sites.add(junction_data[j_id]['donor'])
                                filtered_splice_sites.add(junction_data[j_id]['acceptor'])

                            stats['junction_counts'][len(filtered_connected)] += 1
                            stats['junctions_in_atses'].update(filtered_connected)

                            atse_groups[event_id] = {
                                'gene_id': gene_id,
                                'junction_ids': list(filtered_connected),
                                'num_junctions': len(filtered_connected),
                                'splice_sites': list(filtered_splice_sites),
                                'filtered_junctions': list(filtered_out),
                                'junction_usage': {j_id: junction_usage[j_id] for j_id in filtered_connected}
                            }
                            event_counter += 1
                        else:
                            # Add the remaining junctions as singletons if they don't form an ATSE anymore
                            for j_id in filtered_connected:
                                singleton_info = self.analyze_singleton_junctions(G, j_id)
                                singleton_info['gene_id'] = gene_id
                                stats['singleton_junctions'].append(singleton_info)

                        # Mark all junctions as visited
                        visited.update(connected)
                    else:
                        # This is a singleton junction
                        singleton_info = self.analyze_singleton_junctions(G, junction_id)
                        singleton_info['gene_id'] = gene_id
                        stats['singleton_junctions'].append(singleton_info)
                        visited.add(junction_id)

        # Calculate final statistics
        total_atses = len(atse_groups)
        junctions_used = len(stats['junctions_in_atses'])
        singleton_count = len(stats['singleton_junctions'])
    
        # Perform sanity check on sample of unincluded junctions
        problematic = self.sample_unincluded_junctions(graphs, stats['junctions_in_atses'])

        # Write singleton information to file
        with open('singleton_junctions.tsv', 'w') as f:
            # Write header
            f.write('gene_id\tjunction_id\tchromosome\tdonor_site\tacceptor_site\tstrand\tread_count\n')
            # Write data
            for j in stats['singleton_junctions']:
                f.write(f"{j['gene_id']}\t{j['junction_id']}\t{j['chromosome']}\t{j['donor_site']}\t"
                       f"{j['acceptor_site']}\t{j['strand']}\t{j['score']}\n")

        # Write filtered junction information to file
        with open('filtered_low_usage_junctions.tsv', 'w') as f:
            # Write header
            f.write('gene_id\tjunction_id\tjunction_reads\tdonor_total_reads\tdonor_usage\tacceptor_total_reads\tacceptor_usage\t5prime_usage\t3prime_usage\n')
            # Write data
            for j in stats['filtered_junctions']:
                f.write(f"{j['gene_id']}\t{j['junction_id']}\t{j['junction_reads']}\t"
                       f"{j['donor_total_reads']}\t{j['donor_usage']:.4f}\t"
                       f"{j['acceptor_total_reads']}\t{j['acceptor_usage']:.4f}\t"
                       f"{j['five_prime_usage']:.4f}\t{j['three_prime_usage']:.4f}\n")

        print(f"""
                ATSE Analysis Summary:
                ---------------------
                Total junctions in dataset: {stats['total_junctions']}
                Analyzable junctions (in genes with ≥2 junctions): {stats['analyzable_junctions']}
                Junctions filtered due to low splice site usage (<{min_splice_site_usage*100:.1f}%): {filtered_junctions}
                Junctions included in ATSEs: {junctions_used}
                Singleton junctions: {singleton_count}
                Total ATSEs found: {total_atses}

                ATSE Size Distribution:
                ----------------------""")

        sorted_counts = dict(sorted(stats['junction_counts'].items()))
        for num_junctions, count in sorted_counts.items():
            print(f"ATSEs with {num_junctions} junctions: {count}")

        if problematic:
            print(f"\nWARNING: Found {len(problematic)} potentially problematic cases in random sampling")
            print("First few examples:")
            for case in problematic[:5]:
                print(f"Junction {case['junction_id']} in gene {case['gene_id']} shares site with {case['shares_site_with']}")
        else:
            print("\nRandom sampling verification: OK - no shared splice sites found in sampled junctions")

        print(f"\nDetails of filtered junctions saved to 'filtered_low_usage_junctions.tsv'")

        return atse_groups, sorted_counts
        
    def classify_events(self, graphs: Dict[str, nx.Graph], atse_groups: Dict[str, Dict]):
        # Initialize counter for event types
        event_counts = {
            'alternative_3_prime': 0,
            'alternative_5_prime': 0,
            'exon_skip': 0,
            'complex': 0
        }

        for event_id, group in atse_groups.items():
            G = graphs[group['gene_id']]

            # Count unique donor and acceptor sites
            donor_sites = len([s for s in group['splice_sites'] if s[2] == 'donor'])
            acceptor_sites = len([s for s in group['splice_sites'] if s[2] == 'acceptor'])

            # Need a window on how far the alternative splice sites are 
            if donor_sites == 1 and acceptor_sites > 1:
                group['event_type'] = 'alternative_3_prime'
            elif donor_sites > 1 and acceptor_sites == 1:
                group['event_type'] = 'alternative_5_prime'
            elif len(group['junction_ids']) == 3:
                # Get junction coordinates and strand
                junc_coords = []
                strand = None
                for j_id in group['junction_ids']:
                    for u, v, data in G.edges(data=True):
                        if data['junction_id'] == j_id:
                            coord1, coord2 = u[1], v[1]
                            strand = u[2]  # Get strand from node tuple (assuming format: (gene_id, position, strand))
                            junc_coords.append((coord1, coord2))
                            break
                        
                # Sort junctions based on start position, accounting for strand
                if strand == '+':
                    # For positive strand, smaller coordinate is start
                    junc_coords.sort(key=lambda x: min(x))
                else:
                    # For negative strand, larger coordinate is start
                    junc_coords.sort(key=lambda x: -max(x))

                # Check for exon skipping by comparing starts and ends based on strand
                is_exon_skip = False
                if strand == '+':
                    # Positive strand: compare smallest coordinates for starts, largest for ends
                    if (min(junc_coords[0]) == min(junc_coords[1]) and  # J1 start == J2 start
                        max(junc_coords[1]) == max(junc_coords[2])):    # J2 end == J3 end
                        is_exon_skip = True
                else:
                    # Negative strand: compare largest coordinates for starts, smallest for ends
                    if (max(junc_coords[0]) == max(junc_coords[1]) and  # J1 start == J2 start
                        min(junc_coords[1]) == min(junc_coords[2])):    # J2 end == J3 end
                        is_exon_skip = True

                group['event_type'] = 'exon_skip' if is_exon_skip else 'complex'
            else:
                group['event_type'] = 'complex'

            # Update counter
            event_counts[group['event_type']] += 1

        return atse_groups, event_counts
    
    def save_atse_file(self, atse_groups: Dict[str, Dict], junctions: Dict[str, Dict], file_name: str):
        """
        Save ATSE groups to a tab-delimited file with gzip compression, including junction annotations.

        Args:
            atse_groups: Dictionary of ATSE events
            junctions: Dictionary of junction annotations
            file_name: Output file path (will append .gz if not present)
        """
        import gzip

        required_fields = {'gene_id', 'num_junctions', 'event_type', 'junction_ids'}

        # Ensure file has .gz extension
        if not file_name.endswith('.gz'):
            file_name = file_name + '.gz'

        try:
            with gzip.open(file_name, 'wt') as f:  # 'wt' for write text mode
                # Write header
                f.write("event_id\tgene_id\tgene_name\tgene_types\t"
                   "transcripts\tboth_ends_transcripts\tonly_5_prime_transcripts\tonly_3_prime_transcripts\t"
                   "transcript_types\tnum_junctions\tevent_type\tannotation_status\t"
                   "junction_id\tchrom\tstart\tend\tstrand\tcells\ttotal_score\t"
                   "five_prime_usage\tthree_prime_usage\tdonor_usage\tacceptor_usage\t"
                   "donor_total_reads\tacceptor_total_reads\t"  # New columns
                   "splice_motif\tdonor_seq\tacceptor_seq\t"
                   "position_off_5_prime\tposition_off_3_prime\n")

                # Write data
                for event_id, group in atse_groups.items():
                    # Verify all required fields are present
                    missing_fields = required_fields - set(group.keys())
                    if missing_fields:
                        print(f"Warning: Event {event_id} missing required fields: {missing_fields}")
                        continue
                    
                    # Get junction usage data if available
                    junction_usage = group.get('junction_usage', {})

                    try:
                        # For each junction in the ATSE
                        for junction_id in group['junction_ids']:
                            if junction_id not in junctions:
                                print(f"Warning: Junction {junction_id} not found in annotations")
                                continue

                            j_data = junctions[junction_id]

                            # Get usage values for this junction
                            usage_data = junction_usage.get(junction_id, {})
                            five_prime_usage = usage_data.get('five_prime_usage', 'NA')
                            three_prime_usage = usage_data.get('three_prime_usage', 'NA')
                            donor_usage = usage_data.get('donor_usage', 'NA')
                            acceptor_usage = usage_data.get('acceptor_usage', 'NA')

                            # Get donor and acceptor total reads
                            donor_total_reads = usage_data.get('donor_total_reads', 'NA')
                            acceptor_total_reads = usage_data.get('acceptor_total_reads', 'NA')

                            # Format usage values
                            five_prime_usage_str = f"{five_prime_usage:.4f}" if isinstance(five_prime_usage, float) else 'NA'
                            three_prime_usage_str = f"{three_prime_usage:.4f}" if isinstance(three_prime_usage, float) else 'NA'
                            donor_usage_str = f"{donor_usage:.4f}" if isinstance(donor_usage, float) else 'NA'
                            acceptor_usage_str = f"{acceptor_usage:.4f}" if isinstance(acceptor_usage, float) else 'NA'

                            # Format total reads values
                            donor_total_reads_str = f"{donor_total_reads}" if isinstance(donor_total_reads, (int, float)) else 'NA'
                            acceptor_total_reads_str = f"{acceptor_total_reads}" if isinstance(acceptor_total_reads, (int, float)) else 'NA'

                            # Handle gene names - join with pipe if multiple names exist
                            gene_names = j_data.get('gene_names', [])
                            gene_names_str = '|'.join(str(name) for name in gene_names) if gene_names else 'NA'

                            # Handle gene types - new column
                            gene_types = j_data.get('gene_types', [])
                            gene_types_str = '|'.join(str(gtype) for gtype in gene_types) if gene_types else 'NA'

                            # Handle transcripts - join with comma or return NA if empty
                            transcripts = j_data.get('transcripts', [])
                            transcripts_str = ','.join(str(t) for t in transcripts) if transcripts else 'NA'

                            # Handle the three transcript categories - new columns
                            both_ends = j_data.get('both_ends_transcripts', [])
                            both_ends_str = ','.join(str(t) for t in both_ends) if both_ends else 'NA'

                            only_5_prime = j_data.get('only_5_prime_transcripts', [])
                            only_5_prime_str = ','.join(str(t) for t in only_5_prime) if only_5_prime else 'NA'

                            only_3_prime = j_data.get('only_3_prime_transcripts', [])
                            only_3_prime_str = ','.join(str(t) for t in only_3_prime) if only_3_prime else 'NA'

                            # Handle transcripts types - join with comma or return NA if empty
                            transcript_types = j_data.get('transcript_types', [])
                            transcript_types_str = ','.join(str(t) for t in transcript_types) if transcript_types else 'NA'

                            # Write line with all information, including new columns
                            f.write(f"{event_id}\t"
                               f"{group['gene_id']}\t"
                               f"{gene_names_str}\t"
                               f"{gene_types_str}\t"
                               f"{transcripts_str}\t"
                               f"{both_ends_str}\t"
                               f"{only_5_prime_str}\t"
                               f"{only_3_prime_str}\t"
                               f"{transcript_types_str}\t"
                               f"{group['num_junctions']}\t"
                               f"{group['event_type']}\t"
                               f"{j_data.get('annotation_status', 'NA')}\t"
                               f"{junction_id}\t"
                               f"{j_data.get('chrom', 'NA')}\t"
                               f"{j_data.get('start', 'NA')}\t"
                               f"{j_data.get('end', 'NA')}\t"
                               f"{j_data.get('strand', 'NA')}\t"
                               f"{j_data.get('cells', 'NA')}\t"
                               f"{j_data.get('total_score', 'NA')}\t"
                               f"{five_prime_usage_str}\t"
                               f"{three_prime_usage_str}\t"
                               f"{donor_usage_str}\t"
                               f"{acceptor_usage_str}\t"
                               f"{donor_total_reads_str}\t"  # New column
                               f"{acceptor_total_reads_str}\t"  # New column
                               f"{j_data.get('splice_motif', 'NA')}\t"
                               f"{j_data.get('donor_seq', 'NA')}\t"
                               f"{j_data.get('acceptor_seq', 'NA')}\t"
                               f"{j_data.get('position_off_5_prime', 'NA')}\t"
                               f"{j_data.get('position_off_3_prime', 'NA')}\n"
                            )
                    except Exception as e:
                        print(f"Warning: Error writing event {event_id}: {str(e)}")
                        continue
                    
            print(f"ATSEs successfully saved to {file_name}")
            print(f"Wrote {len(atse_groups)} ATSE events")
        
        except IOError as e:
            print(f"Error: Could not write to file {file_name}: {str(e)}")
            raise
        except Exception as e:
            print(f"Error: Unexpected error while saving ATSEs: {str(e)}")
            raise

def create_analysis_summary(reader, len_junction_files, filtered_junctions):
   summary_data = {
       'Parameter': [
           'Number of input files',
           'Min intron length',
           'Max intron length',
           'Min cells per junction',
           'Min reads per junction',
           'Splice site filter',
           'Total junctions found',
           'Average cells per junction',
           'Number of unique genes', 
       ],
       'Value': [
           len_junction_files,
           reader.min_intron,
           reader.max_intron,
           reader.min_cells,
           reader.min_reads,
           'Canonical only',
           len(filtered_junctions),
           sum(j['cells'] for j in filtered_junctions.values()) / len(filtered_junctions) if filtered_junctions else 0,
        len({g for j in filtered_junctions.values() for g in j['gene_ids']}) if filtered_junctions else 0
       ]
   }
   
   summary_df = pd.DataFrame(summary_data)
   # Add today's and time date to file name 
   file_name = 'ATSE_mapping_junction_summary_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
   summary_df.to_csv(file_name, index=False)
   print(f"Summary saved to {file_name}")
   return summary_df
