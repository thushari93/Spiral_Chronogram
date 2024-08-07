import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from Bio import SeqIO
from Bio import AlignIO
import os
from IPython.display import clear_output


def extract_dates_and_accessions(input_pattern, aligned_sequence_file):
        """
    Extract dates and accessions from CSV files, align them with sequences from a FASTA file,
    and save the results to new CSV files.

    Parameters:
    input_pattern (str): The pattern to match CSV files (e.g., "data/*.csv").
    aligned_sequence_file (str): The path to the aligned sequence file in FASTA format.

    Returns:
    None
    """
    # Find all files matching the input pattern
    files = glob.glob(input_pattern)
    
    # Create an empty list to store results
    result_list = []
    
    for file in files:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check if the required columns exist in the DataFrame
        if 'Collection_Date' in df.columns and 'Accession' in df.columns:
            # Extract the columns and rename them
            extracted_df = df[['Collection_Date', 'Accession']].rename(columns={'Collection_Date': 'Date'})
            
            # Convert the 'Date' column to datetime format, coerce errors to NaT
            extracted_df['Date'] = pd.to_datetime(extracted_df['Date'], format='%Y-%m-%d', errors='coerce')
            
            # Drop rows with NaT in the 'Date' column
            extracted_df = extracted_df.dropna(subset=['Date'])
            
            # Append the extracted data to the result list if it's not empty
            if not extracted_df.empty:
                result_list.append(extracted_df)
    
    # Concatenate all non-empty DataFrames in the result list
    if result_list:
        result_df = pd.concat(result_list, ignore_index=True)
        
        # Sort the DataFrame by the 'Date' column
        result_df = result_df.sort_values(by='Date').reset_index(drop=True)
        
        
        # Load the sequences from the sequence file
        alignment = list(SeqIO.parse(aligned_sequence_file, "fasta"))
        alignment_dict = {record.id.split(':')[0]: record for record in alignment}
        
        # Filter the DataFrame to only include rows with matching sequences
        result_df = result_df[result_df['Accession'].isin(alignment_dict.keys())]
        
        # Order the sequences according to the sorted accessions
        Align_ordered = [alignment_dict[acc] for acc in result_df['Accession']]
        
        # Convert ordered sequences to a list of strings
        Align_ordered_str = [str(record.seq) for record in Align_ordered]
        
        result_df['Seq'] = Align_ordered_str
        output_file = "dat_Acces_Seq_" + os.path.splitext(os.path.basename(aligned_sequence_file))[0] + ".csv"
        
        # Save the result dates and accession numbers to a new CSV file
        result_df.to_csv(output_file, index=False)
        #print(f"Data saved to {output_file}")
        
        # Save the Date alone to a new CSV file
        output_date_file = "dates_" + os.path.splitext(os.path.basename())[0] + ".csv"
        result_df['Date'].to_csv(output_date_file, index=False)
        #print(f"Dates saved to {output_date_file}")
       
        
    else:
        print("No valid data to save.")

def get_distance(x, y):
    return sum(1 for ele_x, ele_y in zip(x, y) if ele_x != 'N' and ele_y != 'N' and ele_x != '-' and ele_y != '-' and ele_x != ele_y)

def calculate_distance_matrix(csv_file, output_file):
    """
    Calculate the pairwise distance matrix for the given alignment data and save the result.

    Parameters:
    csv_file (str): The path to the .csv file containing the dates, accession numbers, and aligned sequences.
    output_file (str): The path to the .npy file where the resulting distance matrix will be saved.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract the sequences
    sequences = df['Seq'].tolist()
    n = len(sequences)

    # Initialize the distance matrix
    dis_array_ordered = np.zeros((n, n))
    
    # Calculate the pairwise distances
    for i, ele_1 in enumerate(sequences):
        t = i / n
        clear_output(wait=True)  
        print(f"Progress: {t*100:.2f}%", flush=True)
        for j in range(i):
            distance = get_distance(ele_1, sequences[j])  
            dis_array_ordered[i, j] = distance
            dis_array_ordered[j, i] = distance

    # Save the resulting distance matrix
    np.save(output_file, dis_array_ordered)
    #print(f"Distance matrix saved to {output_file}")
