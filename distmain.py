import argparse
import os
from distutils import extract_dates_and_accessions
from distutils import get_distance
from distutils import calculate_distance_matrix

def distmain():
    parser = argparse.ArgumentParser(description="Extract dates and accessions, and generate distance matrix.")
    
    parser.add_argument('--input_pattern', type=str, help="The pattern to match CSV files (e.g., 'data_*.csv').")
    parser.add_argument('--aligned_sequence_file', type=str, help="The path to the aligned sequence file in FASTA format.")

    args = parser.parse_args()
    
    # Extract dates and accessions
    extract_dates_and_accessions(args.input_pattern, args.aligned_sequence_file)
    
    # Create file names based on the aligned sequence file name
    base_name = os.path.splitext(os.path.basename(args.aligned_sequence_file))[0]
    csv_file = f"dat_Acces_Seq_{base_name}.csv"
    
    # Save the Date alone to a CSV file
    output_date_file = f"dates_{base_name}.csv"
    print(f"Dates saved to {output_date_file}")
    
    # Save the Distance matrix to a .npy file
    dist_output_file = f"dist_{base_name}.npy"
    calculate_distance_matrix(csv_file, dist_output_file)
    print(f"Distance matrix saved to {dist_output_file}")

if __name__ == "__main__":
    distmain()
