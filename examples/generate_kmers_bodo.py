import pandas as pd
from Bio import SeqIO
import glob
import os
import bodo
import time


df_sample = pd.DataFrame({"annotation": ["a"], "PROBE_SEQUENCE": ["p"], "source": ["s"]})
df_type = bodo.typeof(df_sample)


@bodo.wrap_python(df_type)
def fasta4epitope(fasta_file):

    # Save sequence names and sequences in a list
    seq_names = []
    sequences = []

    # Read fasta file containing epitope sequences
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_names.append(record.id)
        sequences.append(str(record.seq))
    
    # Save these in a data frame
    seq_df = pd.DataFrame({'annotation': seq_names, 'PROBE_SEQUENCE': sequences})
    
    # Add a column with epitope name
    seq_df['source'] = os.path.basename(fasta_file).split('.')[0]
    
    # Change all columns to string
    seq_df = seq_df.astype(str)
    return seq_df


@bodo.jit(cache=True)
def process_all_fastas(fasta_files, out_path):
    t0 = time.time()

    combined_df = pd.DataFrame()    
    # Load each fasta file into a data frame and combine them
    for i in bodo.prange(len(fasta_files)):
        combined_df = pd.concat([combined_df, fasta4epitope(fasta_files[i])], ignore_index=True)
        
    # Save the combined data frame to a csv file
    combined_df.to_csv(out_path, index=False)
    print("Execution time: ", time.time() - t0)


if __name__ == '__main__':
    # Directory containing the fasta files
    directory = '../tmp/kmers/'
    # Find all fasta files in the specified directory
    fasta_files = glob.glob(os.path.join(directory, '*.fasta'))
    out_path = os.path.join(directory, 'combined_fasta_sequences.csv')

    # Process all fasta files and combine them into a single data frame
    process_all_fastas(fasta_files, out_path)
