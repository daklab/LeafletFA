# process_junctions.py
import os
import math
import pickle
import argparse
from pathlib import Path
from typing import List, Dict
from find_intron_clusters_v3 import JunctionReader
from tqdm import tqdm

def split_file_list(input_file: str, chunks: int, output_dir: str):
    """Split input file list into chunks for Slurm array processing"""
    with open(input_file) as f:
        files = [line.strip() for line in f if line.strip()]
    
    chunk_size = math.ceil(len(files) / chunks)
    os.makedirs(output_dir, exist_ok=True)
    
    # Count actual chunks that will be created
    actual_chunks = math.ceil(len(files) / chunk_size)
    print(f"Will create {actual_chunks} chunks from {len(files)} files")
    
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        chunk_id = i // chunk_size
        # Simple chunk ID without padding to match SLURM_ARRAY_TASK_ID
        chunk_filename = f"{output_dir}/chunk_{chunk_id}.txt"
        print(f"Creating chunk {chunk_id} with {len(chunk)} files: {chunk_filename}")
        with open(chunk_filename, 'w') as f:
            f.write('\n'.join(chunk))
    
    return len(files), actual_chunks

def process_chunk(chunk_file: str, output_dir: str):
    """Process a single chunk of files"""
    print(f"Processing chunk file: {chunk_file}")
    if not os.path.exists(chunk_file):
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    with open(chunk_file) as f:
        file_list = [line.strip() for line in f]
    
    print(f"Found {len(file_list)} files in chunk")
    
    reader = JunctionReader(
        min_intron=50,
        max_intron=500000,
        batch_size=10,
        num_workers=4
    )
    
    junctions = reader.process_files(file_list)
    
    chunk_id = Path(chunk_file).stem
    output_file = f"{output_dir}/{chunk_id}_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(junctions, f)
    
    print(f"Saved results to {output_file}")

def merge_results(results_dir: str, output_file: str):
    """Merge all processed chunks"""
    combined_junctions = {}
    
    results_files = list(Path(results_dir).glob('*_results.pkl'))
    print(f"Found {len(results_files)} result files to merge")
    
    for results_file in results_files:
        print(f"Processing {results_file}")
        with open(results_file, 'rb') as f:
            chunk_results = pickle.load(f)
            
        for j_id, j_data in chunk_results.items():
            if j_id not in combined_junctions:
                combined_junctions[j_id] = j_data
            else:
                combined_junctions[j_id]['cells'] += j_data['cells']
                combined_junctions[j_id]['total_score'] += j_data['total_score']
    
    with open(output_file, 'wb') as f:
        pickle.dump(combined_junctions, f)
    
    return len(combined_junctions)

def main():
    parser = argparse.ArgumentParser(description='Process junction files using Slurm arrays')
    parser.add_argument('--mode', choices=['split', 'process', 'merge'], required=True,
                      help='Operation mode: split files, process chunk, or merge results')
    parser.add_argument('--input-file', help='Text file containing list of junction files')
    parser.add_argument('--chunks', type=int, help='Number of chunks for splitting')
    parser.add_argument('--chunk-file', help='Input chunk file for processing')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--merge-output', help='Output file for merged results')
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        if not args.input_file or not args.chunks:
            parser.error('--input-file and --chunks required for split mode')
        total_files, chunks = split_file_list(args.input_file, args.chunks, args.output_dir)
        print(f'Split {total_files} files into {chunks} chunks')
    
    elif args.mode == 'process':
        if not args.chunk_file:
            parser.error('--chunk-file required for process mode')
        process_chunk(args.chunk_file, args.output_dir)
    
    elif args.mode == 'merge':
        if not args.merge_output:
            parser.error('--merge-output required for merge mode')
        total_junctions = merge_results(args.output_dir, args.merge_output)
        print(f'Found {total_junctions} unique junctions')

if __name__ == '__main__':
    main()