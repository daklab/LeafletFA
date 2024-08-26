import sys
import torch 
import h5py
import argparse
import pdb

# Argument parser for command-line options (moved to the top)
parser = argparse.ArgumentParser(description="Simulate data using real dataset and prepare it for model input.")
parser.add_argument("--input_folder", type=str, required=True, help="Path to the input files folder.")
parser.add_argument("--K", type=int, required=True, help="Number of cell types to simulate.")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output .h5 file.")
args = parser.parse_args()

print(f"Input folder: {args.input_folder}")
print(f"Number of cell types: {args.K}")
print(f"Output file: {args.output_file}")

# Import custom modules after setting up arguments
sys.path.append("/gpfs/commons/home/kisaev/Leaflet-private/src/simulation/")
import simulate_counts as sim 

torch.manual_seed(42)

def save_to_h5(file_path, cell_index_tensor, junc_index_tensor, final_data, simple_data, compression="gzip", compression_opts=9):
    print(f"Saving cell_index_tensor and junc_index_tensor into HDF5 format...")
    with h5py.File(file_path, 'w') as f:
        # Save tensors as compressed datasets with explicit chunks
        f.create_dataset('cell_index_tensor', data=cell_index_tensor.cpu().numpy(), 
                         compression=compression, compression_opts=compression_opts, chunks=True)
        f.create_dataset('junc_index_tensor', data=junc_index_tensor.cpu().numpy(), 
                         compression=compression, compression_opts=compression_opts, chunks=True)

    print(f"Tensors successfully saved to {file_path}.")
    
    # Save DataFrames as compressed CSV files
    final_data_csv_path = file_path.replace('.h5', '_final_data.csv.gz')
    simple_data_csv_path = file_path.replace('.h5', '_simple_data.csv.gz')
    
    print(f"Saving final_data to {final_data_csv_path}...")
    final_data.to_csv(final_data_csv_path, index=False, compression='gzip')
    
    print(f"Saving simple_data to {simple_data_csv_path}...")
    simple_data.to_csv(simple_data_csv_path, index=False, compression='gzip')
    
    print("DataFrames successfully saved as compressed CSV files.")

# Main function --> simulate_and_prepare_data from sim 
if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    float_type = { 
        "device": device, 
        "dtype": torch.float,
    }

    # Simulate data and prepare tensors
    print(f"Simulating data and preparing tensors...")
    
    cell_index_tensor, junc_index_tensor, my_data, final_data, simple_data = sim.simulate_and_prepare_data(
        input_files_folder=args.input_folder, K=args.K, float_type=float_type, max_intron_count=1000)

    # Save the output (tensors to HDF5, DataFrames to compressed CSV)
    print(f"Saving data to {args.output_file}...")
    save_to_h5(args.output_file, cell_index_tensor, junc_index_tensor, final_data, simple_data)

    print("Simulation and data preparation complete. Data saved.")

# ------------------------------------------------------------------------------------------------------------------------------
# How to run? 
# simulate_code=/gpfs/commons/home/kisaev/Leaflet-private/notebooks/simulation_analysis/simulate_data.py
# input_files_folder=/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/TabulaMurisBrain/model_input/Mammary_Gland/
# python $simulate_code --input_folder $input_files_folder/ --K 2 --output_file "two_cell_types_sim.h5"